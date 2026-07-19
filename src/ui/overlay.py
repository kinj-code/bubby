"""Enhanced overlay window module for the transparent desktop companion.

Thread-safe: all state shared with background threads is guarded by
a threading.Lock(). Widget mutations only happen on the main Qt thread
via Signal/slot dispatch.

Features:
- Click-through mode for non-interactive states
- Working close button (bottom-right corner)
- Drag-to-move (suppresses autonomy spam during drag)
- Sprite-based frame animation with PNG sprites
- Idle wandering via QPropertyAnimation
"""

import logging
import random
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import QWidget, QApplication, QLabel, QLineEdit
from PySide6.QtCore import (
    Qt, QPoint, QTimer, QSize, Signal, QPropertyAnimation,
    QEasingCurve, QRect,
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QCursor, QPixmap, QFont,
)

from src.brain.decisions import Decision, DecisionType

logger = logging.getLogger(__name__)


# ── Sprite animation framework ────────────────────────────────────

class SpriteAnimation:
    """
    Frame-based sprite animation from PNG images.

    Loads a sequence of PNG frames from a directory and cycles
    through them at a configurable FPS via a QTimer.
    """

    def __init__(self, name: str, frames: List[QPixmap], fps: int = 12,
                 loop: bool = True):
        self.name = name
        self.frames = frames
        self.fps = fps
        self.loop = loop
        self.current_frame = 0
        self.playing = False

    def start(self) -> None:
        self.playing = True
        self.current_frame = 0

    def stop(self) -> None:
        self.playing = False

    def current_pixmap(self) -> Optional[QPixmap]:
        if not self.frames:
            return None
        idx = self.current_frame % len(self.frames)
        return self.frames[idx]

    def advance(self) -> bool:
        """Advance to next frame. Returns True if still playing."""
        if not self.frames:
            self.playing = False
            return False
        self.current_frame += 1
        if self.current_frame >= len(self.frames):
            if self.loop:
                self.current_frame = 0
                return True
            else:
                self.playing = False
                return False
        return True


class SpriteManager:
    """
    Manages multiple sprite animations loaded from disk.

    Directory layout expected:
      sprites/
        idle/
          00.png, 01.png, 02.png, ...
        walk/
          00.png, 01.png, ...
        wave/
          00.png, 01.png, ...
    """

    SPRITE_DIR = Path(__file__).parent.parent.parent / "sprites"

    def __init__(self) -> None:
        self._animations: Dict[str, SpriteAnimation] = {}
        self._current: Optional[str] = None
        self._load_all()

    def _load_all(self) -> None:
        """Load all sprite directories into animations."""
        if not self.SPRITE_DIR.exists():
            logger.info("No sprites/ directory found — using emoji fallback")
            return

        for anim_dir in sorted(self.SPRITE_DIR.iterdir()):
            if not anim_dir.is_dir():
                continue
            name = anim_dir.name.lower()
            frames: List[QPixmap] = []
            for img_file in sorted(anim_dir.glob("*.png")):
                pix = QPixmap(str(img_file))
                if not pix.isNull():
                    frames.append(pix)
            if frames:
                self._animations[name] = SpriteAnimation(
                    name=name, frames=frames, fps=12, loop=True,
                )
                logger.info(f"Sprite loaded: '{name}' ({len(frames)} frames)")
            else:
                logger.debug(f"No PNG frames in sprites/{name}/")

    def has_sprites(self) -> bool:
        return len(self._animations) > 0

    def play(self, name: str) -> Optional[SpriteAnimation]:
        """Start playing a named animation. Returns the SpriteAnimation."""
        if name in self._animations:
            self._current = name
            anim = self._animations[name]
            anim.start()
            return anim
        return None

    def current_animation(self) -> Optional[SpriteAnimation]:
        if self._current:
            return self._animations.get(self._current)
        return None


# ── Overlay Window ────────────────────────────────────────────────

class OverlayWindow(QWidget):
    """
    Frameless, transparent, always-on-top overlay window with
    sprite animation, close button, dragging, and idle wandering.
    """

    # ── Signals ──
    close_requested = Signal()
    drag_started_signal = Signal(QPoint)
    drag_finished_signal = Signal(QPoint)
    click_through_changed = Signal(bool)
    display_message_signal = Signal(str, str, str)
    update_state_signal = Signal(str, str)
    behavior_state_signal = Signal(object)
    # Signal emitted while dragging (so autonomy loop can suppress)
    dragging_changed = Signal(bool)
    # Emitted when user clicks the character body (not close button)
    user_poked = Signal()
    # Emitted when user types a message and presses Enter in the input box
    user_message_submitted = Signal(str)

    CLOSE_ZONE_SIZE = 24
    CLOSE_ZONE_COLOR = QColor(255, 59, 48, 180)
    CLOSE_ZONE_HOVER_COLOR = QColor(255, 80, 60, 220)

    # Wandering
    WANDER_INTERVAL_MS = 12_000  # Wander every 12 seconds when idle
    WANDER_DURATION_MS = 3_500   # Movement takes 3.5 seconds

    def __init__(
        self,
        size: Tuple[int, int] = (400, 400),
        click_through: bool = False,
        close_zone_enabled: bool = True,
    ) -> None:
        super().__init__()

        self._window_size = size
        self._click_through = click_through
        self._close_zone_enabled = close_zone_enabled
        self._drag_start_pos: Optional[QPoint] = None
        self._is_dragging = False
        self._drag_threshold = 5
        self._animation_widget: Optional[QWidget] = None
        self._state_label: Optional[QLabel] = None
        self._current_tint: Optional[QColor] = None
        self._state_lock = threading.Lock()

        # Sprite animation
        self._sprite_manager = SpriteManager()
        self._sprite_timer = QTimer(self)
        self._sprite_timer.timeout.connect(self._advance_sprite)
        self._current_anim: Optional[SpriteAnimation] = None
        self._current_anim_name = "idle"

        # Emoji fallback
        self._emoji_label: Optional[QLabel] = None

        # Idle wandering timer
        self._wander_timer = QTimer(self)
        self._wander_timer.timeout.connect(self._idle_wander)
        self._wander_timer.start(self.WANDER_INTERVAL_MS)

        self._setup_window()
        self._setup_input_box()
        self._setup_close_zone_timer()

        # Start default animation
        self.set_state("idle")
        logger.info(f"OverlayWindow initialized: {size}")

    # ── Window setup ──────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.set_click_through(self._click_through)
        self.setFixedSize(*self._window_size)
        self.setMouseTracking(True)

    def _setup_input_box(self) -> None:
        """Create a text input box for chatting with Bubby."""
        w, h = self._window_size
        input_height = 32
        self._input_box = QLineEdit(self)
        self._input_box.setPlaceholderText("Chat with Bubby...")
        self._input_box.setGeometry(8, h - input_height - 8, w - 16, input_height)
        self._input_box.setStyleSheet("""
            QLineEdit {
                background: rgba(20, 20, 30, 180);
                color: rgba(255, 255, 255, 230);
                border: 1px solid rgba(100, 180, 220, 120);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid rgba(100, 180, 220, 200);
            }
        """)
        self._input_box.returnPressed.connect(self._on_input_submitted)
        # Hide by default — user can type to show
        self._input_box.show()

    def _on_input_submitted(self) -> None:
        """Handle Enter key in the input box."""
        text = self._input_box.text().strip()
        if text:
            logger.debug(f"User submitted message: {text[:60]}")
            self.user_message_submitted.emit(text)
        self._input_box.clear()

    def set_input_visible(self, visible: bool) -> None:
        """Show or hide the text input box."""
        if hasattr(self, '_input_box'):
            self._input_box.setVisible(visible)

    def _setup_close_zone_timer(self) -> None:
        self._close_zone_timer = QTimer(self)
        self._close_zone_timer.timeout.connect(self._update_close_zone)
        self._close_zone_timer.start(16)

    # ── Public API ────────────────────────────────────────────────

    def set_state(self, state_name: str, message_text: str = "") -> None:
        """
        Set the visual animation state.

        Args:
            state_name: One of 'idle', 'wave', 'talk', 'observe',
                        'think', 'confused', 'curious', 'success'.
            message_text: Optional message to display.
        """
        self._current_anim_name = state_name

        # Try sprite first
        if self._sprite_manager.has_sprites():
            anim = self._sprite_manager.play(state_name)
            if anim:
                if not self._sprite_timer.isActive():
                    self._sprite_timer.start(1000 // max(anim.fps, 1))
                self._current_anim = anim
                if self._emoji_label:
                    self._emoji_label.hide()
                self.update()
                return

        # Emoji fallback
        self._sprite_timer.stop()
        self._current_anim = None
        emote_map = {
            "idle": "\U0001F600", "wave": "\U0001F44B", "talk": "\U0001F4AC",
            "observe": "\U0001F440", "think": "\U0001F914",
            "confused": "\U0001F615", "curious": "\U0001F9D0",
            "success": "\u2705", "frustrated": "\U0001F624",
        }
        emoji = emote_map.get(state_name, "\U0001F600")
        if not self._emoji_label:
            self._emoji_label = QLabel(self)
            font = QFont()
            font.setPointSize(64)
            self._emoji_label.setFont(font)
            self._emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._emoji_label.setStyleSheet("background: transparent;")
            self._emoji_label.setGeometry(0, 0, *self._window_size)
        self._emoji_label.setText(emoji)
        self._emoji_label.show()

        if message_text:
            self._update_state_text(message_text)

    def set_click_through(self, enabled: bool) -> None:
        self._click_through = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        self.click_through_changed.emit(enabled)

    def is_click_through(self) -> bool:
        return self._click_through

    def set_animation_widget(self, widget: Optional[QWidget]) -> None:
        if self._animation_widget:
            self._animation_widget.setParent(None)
        self._animation_widget = widget
        if widget:
            widget.setParent(self)
            widget.setGeometry(0, 0, *self._window_size)
            widget.show()

    def show_message(
        self, text: str, animation: str = "idle",
        event_type: str = "observation", duration_ms: int = 5000,
    ) -> None:
        if not text:
            return
        self._update_state_text(text[:100])
        event_tints = {
            "greeting": QColor(255, 200, 200, 60),
            "observation": QColor(200, 180, 255, 60),
            "response": QColor(255, 255, 200, 60),
            "error": QColor(255, 59, 48, 60),
            "status": QColor(200, 200, 200, 30),
        }
        tint = event_tints.get(event_type)
        if tint:
            with self._state_lock:
                self._current_tint = tint
            self.update()
        QTimer.singleShot(duration_ms, self._clear_message)

    def update_behavior_state(self, decision: Decision) -> None:
        logger.debug(f"Behavior state: {decision.decision_type.value}")
        if decision.decision_type == DecisionType.WANDER:
            self._idle_wander()
        self._update_state_tint(decision.decision_type)

    def is_dragging(self) -> bool:
        return self._is_dragging

    def sizeHint(self) -> QSize:
        return QSize(*self._window_size)

    # ── Close button ──────────────────────────────────────────────

    def _close_zone_rect(self) -> QRect:
        """Return the strict 24x24 pixel close button rect in the
        absolute bottom-right corner of the window."""
        w, h = self.width(), self.height()
        return QRect(
            w - self.CLOSE_ZONE_SIZE,
            h - self.CLOSE_ZONE_SIZE,
            self.CLOSE_ZONE_SIZE,
            self.CLOSE_ZONE_SIZE,
        )

    def _is_in_close_zone(self, pos: QPoint) -> bool:
        if not self._close_zone_enabled:
            return False
        return self._close_zone_rect().contains(pos)

    def _update_close_zone(self) -> None:
        if not self._close_zone_enabled or self._click_through:
            self.update()
            return
        if self._is_in_close_zone(self.mapFromGlobal(QCursor.pos())):
            self.update()

    def _clear_message(self) -> None:
        if self._state_label:
            self._state_label.setText("")
        with self._state_lock:
            self._current_tint = None
        self.update()

    # ── Idle wandering ────────────────────────────────────────────

    def _idle_wander(self) -> None:
        """Move window to a random position on screen."""
        if self._is_dragging:
            return
        bounds = self._get_safe_bounds()
        if bounds.width() <= 0 or bounds.height() <= 0:
            return
        tx = random.randint(bounds.left(), max(bounds.left() + 1, bounds.right()))
        ty = random.randint(bounds.top(), max(bounds.top() + 1, bounds.bottom()))
        target = QPoint(tx, ty)
        if target == self.pos():
            return

        logger.debug(f"Idle wander to ({tx}, {ty})")
        # Keep a reference so the animation isn't garbage-collected
        self._wander_anim = QPropertyAnimation(self, b"pos")
        self._wander_anim.setDuration(self.WANDER_DURATION_MS)
        self._wander_anim.setStartValue(self.pos())
        self._wander_anim.setEndValue(target)
        self._wander_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._wander_anim.start()

    def wander_to(self, target: QPoint) -> None:
        """Move window to a specific target position."""
        if self._is_dragging:
            return
        if target == self.pos():
            return
        logger.debug(f"Wander to requested pos ({target.x()}, {target.y()})")
        self._wander_anim = QPropertyAnimation(self, b"pos")
        self._wander_anim.setDuration(self.WANDER_DURATION_MS)
        self._wander_anim.setStartValue(self.pos())
        self._wander_anim.setEndValue(target)
        self._wander_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._wander_anim.start()

    # ── State display helpers ─────────────────────────────────────

    def _update_state_text(self, text: str) -> None:
        if not hasattr(self, '_state_label') or self._state_label is None:
            self._state_label = QLabel(self)
            self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._state_label.setStyleSheet("""
                QLabel {
                    color: rgba(255, 255, 255, 220);
                    font-size: 14px;
                    font-weight: bold;
                    background: rgba(0, 0, 0, 80);
                    border-radius: 8px;
                    padding: 6px 12px;
                }
            """)
            self._state_label.setGeometry(
                20, self._window_size[1] - 60,
                self._window_size[0] - 40, 40,
            )
        self._state_label.setText(text)
        self._state_label.show()

    def _update_state_tint(self, decision_type):
        colors = {
            DecisionType.IDLE: None,
            DecisionType.WANDER: QColor(173, 216, 230, 30),
            DecisionType.PACE: QColor(144, 238, 144, 30),
            DecisionType.SIT: QColor(255, 200, 150, 30),
            DecisionType.OBSERVE_SCREEN: QColor(200, 180, 255, 30),
            DecisionType.INTERACT: QColor(255, 255, 200, 30),
            DecisionType.GREET: QColor(255, 200, 200, 30),
            DecisionType.SLEEP: QColor(100, 100, 150, 50),
        }
        with self._state_lock:
            self._current_tint = colors.get(decision_type)
        self.update()

    def _get_safe_bounds(self) -> QRect:
        screen = QApplication.primaryScreen()
        if not screen:
            return QRect(0, 0, 1920, 1080)
        r = screen.availableGeometry()
        m = 100
        return QRect(
            r.left() + m, r.top() + m,
            r.width() - m * 2 - self._window_size[0],
            r.height() - m * 2 - self._window_size[1],
        )

    # ── Body click interaction ────────────────────────────────────

    def _on_body_click(self) -> None:
        """Handle a click on the character body (not close button).

        Triggers the 'wave' animation and emits user_poked signal
        so the LLM can respond. Reverts to idle after 3 seconds.
        """
        logger.debug("Body clicked — triggering wave + poke")
        self.set_state("wave")
        self.user_poked.emit()
        # Revert to idle after animation plays
        QTimer.singleShot(3000, lambda: self.set_state("idle"))

    # ── Sprite animation ──────────────────────────────────────────

    def _advance_sprite(self) -> None:
        if self._current_anim:
            if not self._current_anim.advance():
                self._sprite_timer.stop()
            self.update()

    # ── Qt events ─────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ARG002
        try:
            # IMPORTANT: Do NOT fill the entire rect with a solid color.
            # The background must remain 100% transparent on X11/Linux.
            # Only draw specific overlays (sprite, close button) on top
            # of the transparent background.

            # State tint — only draw if explicitly set (event-specific highlight)
            with self._state_lock:
                tint = self._current_tint
            if tint is not None:
                p = QPainter(self)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                p.fillRect(self.rect(), tint)
                p.end()

            # Sprite frame
            if self._current_anim and self._current_anim.playing:
                pix = self._current_anim.current_pixmap()
                if pix and not pix.isNull():
                    p = QPainter(self)
                    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                    # Center the sprite in the window, scaled to fit
                    scaled = pix.scaled(
                        self._window_size[0], self._window_size[1],
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    x = (self._window_size[0] - scaled.width()) // 2
                    y = (self._window_size[1] - scaled.height()) // 2
                    p.drawPixmap(x, y, scaled)
                    p.end()

            # Close zone indicator (24x24 rounded rect in bottom-right corner)
            if not self._close_zone_enabled or self._click_through:
                return
            w, h = self.width(), self.height()
            if w <= 0 or h <= 0:
                return
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            hovered = self._is_in_close_zone(mouse_pos)
            if not hovered:
                return

            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            zone_rect = self._close_zone_rect()
            color = self.CLOSE_ZONE_HOVER_COLOR if hovered else self.CLOSE_ZONE_COLOR
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawRoundedRect(zone_rect, 4, 4)
            # X icon
            pen = QPen(QColor(255, 255, 255, 240), 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            margin = 5
            x0, y0 = zone_rect.left() + margin, zone_rect.top() + margin
            x1, y1 = zone_rect.right() - margin, zone_rect.bottom() - margin
            p.drawLine(x0, y0, x1, y1)
            p.drawLine(x1, y0, x0, y1)
            p.end()
        except Exception as e:
            logger.debug(f"paintEvent: {e}")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            # Do NOT close on press — defer decision to release.
            # Only start drag tracking on press.
            self._drag_start_pos = pos
            self._is_dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start_pos and not self._is_dragging:
            if (event.pos() - self._drag_start_pos).manhattanLength() > self._drag_threshold:
                self._is_dragging = True
                self.dragging_changed.emit(True)
                self.drag_started_signal.emit(event.pos())
        if self._is_dragging:
            self.move(event.globalPosition().toPoint() - self._drag_start_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()

            if self._is_dragging:
                # End of drag — do NOT treat as close or interaction
                self.drag_finished_signal.emit(pos)
                self._is_dragging = False
                self._drag_start_pos = None
                self.dragging_changed.emit(False)
            else:
                # No drag occurred — this was a click.
                # Check close zone FIRST (strict 24x24 bottom-right corner).
                if self._is_in_close_zone(pos) and self._close_zone_enabled:
                    logger.info("Close button clicked — shutting down")
                    self.close()
                    QApplication.quit()
                else:
                    # Click on body — trigger interactive animation
                    self._on_body_click()
        super().mouseReleaseEvent(event)

    def enterEvent(self, event) -> None:
        if self._click_through:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._click_through:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        super().leaveEvent(event)

    def closeEvent(self, event) -> None:
        logger.info("OverlayWindow closing")
        self._sprite_timer.stop()
        self._wander_timer.stop()
        if self._animation_widget:
            self._animation_widget.setParent(None)
        super().closeEvent(event)


# ── Testing ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")
    app = QApplication(sys.argv)
    window = OverlayWindow(size=(400, 400))
    window.show()
    sys.exit(app.exec())