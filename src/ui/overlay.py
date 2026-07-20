"""Enhanced overlay window module for the transparent desktop companion.

Thread-safe: all state shared with background threads is guarded by
a threading.Lock(). Widget mutations only happen on the main Qt thread
via Signal/slot dispatch.

Features:
- True transparency via WA_TranslucentBackground + CompositionMode_Source
  (no X11 mask — eliminates the opaque blue box)
- Click-through mode for non-interactive states
- Smooth hover popup chat window (appears on mouse enter, disappears on leave)
- Drag-to-move (suppresses autonomy spam during drag)
- Sprite-based frame animation with PNG sprites
- Idle wandering via QPropertyAnimation with smooth easing
- Stillness rule: ALL movement stops when mouse is over the character
- Centered launch (no top-left snap)
"""

import logging
import random
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import QWidget, QApplication, QLabel, QLineEdit, QTextEdit, QFrame
from PySide6.QtCore import (
    Qt, QPoint, QTimer, QSize, Signal, QPropertyAnimation,
    QEasingCurve, QRect, QParallelAnimationGroup, QPropertyAnimation,
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QCursor, QPixmap, QFont,
    QPainterPath, QFontMetrics,
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


# ── Hover Chat Popup ──────────────────────────────────────────────

class HoverChatPopup(QFrame):
    """
    Smooth pop-out chat window that appears when hovering over the character.

    Features:
    - Slides in from below with opacity fade
    - Contains a read-only chat display and a text input
    - Supports drag-and-drop files
    - Auto-hides when mouse leaves the character area
    """

    message_submitted = Signal(str)
    files_dropped = Signal(list)

    POPUP_WIDTH = 320
    POPUP_HEIGHT = 180
    POPUP_MARGIN = 8

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._visible = False
        self._animating = False
        self._slide_anim: Optional[QPropertyAnimation] = None
        self._opacity_anim: Optional[QPropertyAnimation] = None
        self._opacity = 0.0
        self._slide_offset = 0  # 0 = fully visible, positive = hidden below

        self._setup_ui()
        self.setAcceptDrops(True)
        self.setVisible(False)

    def _setup_ui(self) -> None:
        self.setFixedSize(self.POPUP_WIDTH, self.POPUP_HEIGHT)

        # Container with rounded corners and semi-transparent dark bg
        self.setStyleSheet("""
            HoverChatPopup {
                background: rgba(15, 15, 25, 200);
                border: 1px solid rgba(100, 180, 220, 100);
                border-radius: 12px;
            }
        """)

        layout_rect = QRect(8, 8, self.POPUP_WIDTH - 16, self.POPUP_HEIGHT - 16)

        # Chat display (read-only, shows recent messages)
        self._chat_display = QTextEdit(self)
        self._chat_display.setReadOnly(True)
        self._chat_display.setGeometry(
            layout_rect.left(), layout_rect.top(),
            layout_rect.width(), layout_rect.height() - 44,
        )
        self._chat_display.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: rgba(220, 220, 240, 200);
                font-size: 12px;
                border: none;
                padding: 4px;
            }
            QTextEdit:focus { border: none; }
        """)
        self._chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Input bar at bottom
        self._input_bar = QLineEdit(self)
        self._input_bar.setPlaceholderText("Chat with Bubby...")
        self._input_bar.setGeometry(
            layout_rect.left(), layout_rect.bottom() - 36,
            layout_rect.width(), 32,
        )
        self._input_bar.setStyleSheet("""
            QLineEdit {
                background: rgba(30, 30, 45, 180);
                color: rgba(255, 255, 255, 230);
                border: 1px solid rgba(100, 180, 220, 80);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid rgba(100, 180, 220, 180);
                background: rgba(40, 40, 60, 200);
            }
        """)
        self._input_bar.returnPressed.connect(self._on_input_submitted)

    def _on_input_submitted(self) -> None:
        text = self._input_bar.text().strip()
        if text:
            self.message_submitted.emit(text)
            # Show what the user typed
            self._append_chat(f"<span style='color:#8ab4f8;'>You:</span> {text}")
        self._input_bar.clear()

    def _append_chat(self, html: str) -> None:
        self._chat_display.append(html)
        # Auto-scroll to bottom
        scrollbar = self._chat_display.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def append_message(self, sender: str, message: str) -> None:
        """Add a message to the chat display."""
        color = "#f0c060" if sender == "Bubby" else "#8ab4f8"
        self._append_chat(f"<span style='color:{color};'>{sender}:</span> {message}")

    def show_popup(self, anchor_rect: QRect) -> None:
        """
        Show the popup above the anchor with a slide-up + fade-in animation.
        """
        if self._visible and not self._animating:
            return

        # Position popup centered above the anchor (character window)
        popup_x = anchor_rect.center().x() - self.POPUP_WIDTH // 2
        popup_y = anchor_rect.top() - self.POPUP_HEIGHT - self.POPUP_MARGIN

        # Ensure the popup stays on screen
        screen = QApplication.primaryScreen()
        if screen:
            sg = screen.availableGeometry()
            popup_x = max(sg.left() + 10, min(popup_x, sg.right() - self.POPUP_WIDTH - 10))
            if popup_y < sg.top() + 10:
                # Show below instead if no room above
                popup_y = anchor_rect.bottom() + self.POPUP_MARGIN

        self.move(popup_x, popup_y)
        self.setVisible(True)
        self.raise_()
        self._input_bar.setFocus()

        self._visible = True
        self._animate_in()

    def _animate_in(self) -> None:
        """Slide up + fade in animation."""
        self._animating = True
        self._slide_offset = 40  # Start 40px below final position

        # Opacity animation
        self._opacity_anim = QPropertyAnimation(self, b"opacity")
        self._opacity_anim.setDuration(200)
        self._opacity_anim.setStartValue(0.0)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._opacity_anim.finished.connect(self._on_anim_done)

        # Slide animation - we use the window position for slide effect
        original_pos = self.pos()
        start_pos = QPoint(original_pos.x(), original_pos.y() + self._slide_offset)
        self._slide_anim = QPropertyAnimation(self, b"pos")
        self._slide_anim.setDuration(250)
        self._slide_anim.setStartValue(start_pos)
        self._slide_anim.setEndValue(original_pos)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutBack)

        self._opacity_anim.start()
        self._slide_anim.start()

    def _animate_out(self) -> None:
        """Fade out + slide down animation."""
        self._animating = True

        self._opacity_anim = QPropertyAnimation(self, b"opacity")
        self._opacity_anim.setDuration(150)
        self._opacity_anim.setStartValue(1.0)
        self._opacity_anim.setEndValue(0.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._opacity_anim.finished.connect(self._on_anim_done)

        original_pos = self.pos()
        end_pos = QPoint(original_pos.x(), original_pos.y() + 30)
        self._slide_anim = QPropertyAnimation(self, b"pos")
        self._slide_anim.setDuration(150)
        self._slide_anim.setStartValue(original_pos)
        self._slide_anim.setEndValue(end_pos)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.InCubic)

        self._opacity_anim.start()
        self._slide_anim.start()

    def hide_popup(self) -> None:
        if not self._visible:
            return
        self._visible = False
        self._animate_out()

    def _on_anim_done(self) -> None:
        self._animating = False
        if not self._visible:
            self.setVisible(False)

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, val: float) -> None:
        self._opacity = val
        self.setWindowOpacity(val)
        self.update()

    opacity = property(get_opacity, set_opacity)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                HoverChatPopup {
                    background: rgba(30, 60, 80, 220);
                    border: 2px solid rgba(100, 200, 255, 180);
                    border-radius: 12px;
                }
            """)

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("""
            HoverChatPopup {
                background: rgba(15, 15, 25, 200);
                border: 1px solid rgba(100, 180, 220, 100);
                border-radius: 12px;
            }
        """)

    def dropEvent(self, event) -> None:
        self.setStyleSheet("""
            HoverChatPopup {
                background: rgba(15, 15, 25, 200);
                border: 1px solid rgba(100, 180, 220, 100);
                border-radius: 12px;
            }
        """)
        if event.mimeData().hasUrls():
            urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            if urls:
                self.files_dropped.emit(urls)
                self._append_chat(f"<span style='color:#aaa;'>📎 Dropped {len(urls)} file(s)</span>")
                event.acceptProposedAction()

    def paintEvent(self, event) -> None:
        """Custom paint for rounded corners with the semi-transparent bg."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Clip to rounded rect
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(0, 0, 0, 0), 12, 12)
        p.setClipPath(path)

        # Background
        p.fillRect(self.rect(), QColor(15, 15, 25, 200))

        # Border
        pen = QPen(QColor(100, 180, 220, 100), 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 12, 12)
        p.end()

        # Let child widgets paint
        super().paintEvent(event)

    def closeEvent(self, event) -> None:
        self._visible = False
        self._animating = False
        super().closeEvent(event)


# ── Overlay Window ────────────────────────────────────────────────

class OverlayWindow(QWidget):
    """
    Frameless, transparent, always-on-top overlay window with
    sprite animation, close button, dragging, idle wandering,
    still-on-hover behavior, and a smooth popup chat window.

    True transparency is achieved via:
      - WA_TranslucentBackground + WA_NoSystemBackground
      - paintEvent uses CompositionMode_Source to clear before drawing
      - NO X11 mask (setMask) — eliminates the opaque blue box
    """

    # ── Signals ──
    close_requested = Signal()
    drag_started_signal = Signal(QPoint)
    drag_finished_signal = Signal(QPoint)
    click_through_changed = Signal(bool)
    display_message_signal = Signal(str, str, str)
    update_state_signal = Signal(str, str)
    behavior_state_signal = Signal(object)
    dragging_changed = Signal(bool)
    user_poked = Signal()
    user_message_submitted = Signal(str)
    settings_requested = Signal()

    CLOSE_ZONE_SIZE = 24
    CLOSE_ZONE_COLOR = QColor(255, 59, 48, 180)
    CLOSE_ZONE_HOVER_COLOR = QColor(255, 80, 60, 220)

    # Wandering
    WANDER_INTERVAL_MS = 12_000    # Wander every 12 seconds when idle
    WANDER_DURATION_MS = 3_000     # Movement takes 3 seconds

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

        # Hover state
        self._mouse_over_character = False
        self._hover_timer: Optional[QTimer] = None
        self._hover_popup: Optional[HoverChatPopup] = None

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
        self._setup_hover_popup()
        self._setup_settings_button()
        self._setup_close_zone_timer()

        # Start default animation
        self.set_state("idle")

        # SPRINT 4: Peekaboo timer — check every 2s for overlapping windows
        self._peekaboo_timer = QTimer(self)
        self._peekaboo_timer.timeout.connect(self._update_peekaboo)
        self._peekaboo_timer.start(2000)

        # Launch at center of screen (no top-left snap)
        QTimer.singleShot(50, self._center_on_screen)

        logger.info(f"OverlayWindow initialized: {size}")

    def _center_on_screen(self) -> None:
        """Move window to center of primary screen on launch."""
        screen = QApplication.primaryScreen()
        if screen:
            sg = screen.availableGeometry()
            cx = sg.center().x() - self._window_size[0] // 2
            cy = sg.center().y() - self._window_size[1] // 2
            self.move(cx, cy)
            logger.info(f"Centered on screen: ({cx}, {cy})")

    # ── Window setup ──────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        # True transparency — no X11 mask needed
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.set_click_through(self._click_through)
        self.setFixedSize(*self._window_size)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

    def _setup_hover_popup(self) -> None:
        """Create the hover chat popup window (hidden by default)."""
        self._hover_popup = HoverChatPopup()
        self._hover_popup.message_submitted.connect(self._on_popup_input)
        self._hover_popup.files_dropped.connect(self._on_files_dropped)

        # Timer to detect hover for popup show
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(600)  # 600ms hover delay before showing popup
        self._hover_timer.timeout.connect(self._show_hover_popup)

    def _setup_settings_button(self) -> None:
        """Add a small gear icon button for settings."""
        self._settings_btn = QLabel("⚙", self)
        self._settings_btn.setStyleSheet("""
            QLabel {
                color: rgba(200, 200, 220, 160);
                font-size: 16px;
                background: rgba(0, 0, 0, 40);
                border-radius: 10px;
                padding: 2px 6px;
            }
            QLabel:hover {
                color: rgba(255, 255, 255, 220);
                background: rgba(60, 60, 80, 140);
            }
        """)
        self._settings_btn.setFixedSize(24, 24)
        w, h = self._window_size
        self._settings_btn.move(w - self.CLOSE_ZONE_SIZE - 28, h - self.CLOSE_ZONE_SIZE - 2)
        self._settings_btn.mousePressEvent = lambda e: self.settings_requested.emit()
        self._settings_btn.show()

    # Removed _setup_input_box — input is now in HoverChatPopup

    def _on_popup_input(self, text: str) -> None:
        """Handle Enter key in the hover popup input bar."""
        logger.debug(f"User submitted message from popup: {text[:60]}")
        # Echo user message in popup
        if self._hover_popup:
            self._hover_popup.append_message("You", text)
        self.user_message_submitted.emit(text)

    def _on_files_dropped(self, paths: list) -> None:
        """Handle files dropped on the hover popup."""
        logger.info(f"Files dropped on hover popup: {paths}")
        # Emit as a special user message
        file_list = "\n".join(paths)
        self.user_message_submitted.emit(f"[DROPPED FILES]\n{file_list}")

    def _show_hover_popup(self) -> None:
        """Show the hover chat popup above the character."""
        if self._hover_popup and self._mouse_over_character and not self._is_dragging:
            # Only show if mouse is still over character
            global_pos = self.mapToGlobal(QPoint(0, 0))
            anchor = QRect(global_pos, self.size())
            self._hover_popup.show_popup(anchor)

    def _hide_hover_popup(self) -> None:
        """Hide the hover chat popup."""
        if self._hover_popup:
            self._hover_popup.hide_popup()

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
                        'think', 'confused', 'curious', 'success', etc.
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
            "sleep": "\U0001F634", "excited": "\U0001F929",
            "sad": "\U0001F622", "surprised": "\U0001F62E",
            "love": "\U0001F970", "dance": "\U0001F483",
            "point": "\u261D\uFE0F", "hide": "\U0001F440",
            "stretch": "\U0001F64C", "blush": "\U0001F60A",
            "angry": "\U0001F620", "plead": "\U0001F97A",
            "celebrate": "\U0001F389", "sneeze": "\U0001F927",
            "yawn": "\U0001F971", "reading": "\U0001F4D6",
            "nod": "\U0001F44D", "facepalm": "\U0001F926",
            "victory": "\U0001F3C6",
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

        # Also push message to hover popup if visible
        if self._hover_popup:
            sender = "Bubby"
            self._hover_popup.append_message(sender, text[:200])

        QTimer.singleShot(duration_ms, self._clear_message)

    def update_behavior_state(self, decision: Decision) -> None:
        logger.debug(f"Behavior state: {decision.decision_type.value}")
        if decision.decision_type == DecisionType.WANDER:
            self._idle_wander()
        self._update_state_tint(decision.decision_type)

    def is_dragging(self) -> bool:
        return self._is_dragging

    def is_mouse_over(self) -> bool:
        return self._mouse_over_character

    def sizeHint(self) -> QSize:
        return QSize(*self._window_size)

    # ── Close button ──────────────────────────────────────────────

    def _close_zone_rect(self) -> QRect:
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

    # ── Idle wandering (with stillness rule) ──────────────────────

    def _idle_wander(self) -> None:
        """Move window to a random position on screen.

        Stillness Rule: If the mouse cursor is hovering over the character,
        ALL autonomous movement MUST stop.
        """
        if self._is_dragging:
            return
        if self._mouse_over_character:
            # STILLNESS RULE: No wandering when mouse is over character
            logger.debug("Stillness rule active — suppressing wander (mouse over character)")
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
        if self._mouse_over_character:
            # Stillness rule applies to forced wander too
            logger.debug("Stillness rule active — suppressing forced wander")
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
        """Handle a click on the character body (not close button)."""
        logger.debug("Body clicked — triggering wave + poke")
        self.set_state("wave")
        self.user_poked.emit()
        QTimer.singleShot(3000, lambda: self.set_state("idle"))

    # ── Sprite animation ──────────────────────────────────────────

    def _advance_sprite(self) -> None:
        if self._current_anim:
            if not self._current_anim.advance():
                self._sprite_timer.stop()
        self.update()
        # NO mask update — true transparency via CompositionMode_Source

    # ── SPRINT 4: Resizing via scroll wheel ──────────────────────
    # Shrink down to MIN_SCALE (0.18x), cannot grow larger than 1.0x
    MIN_SCALE = 0.18
    MAX_SCALE = 1.0
    SCALE_STEP = 0.05

    def _current_scale(self) -> float:
        """Get current scale factor relative to original size."""
        return self.width() / self._window_size[0]

    def wheelEvent(self, event) -> None:
        """Handle scroll wheel for resizing the character.

        Ctrl+Scroll UP = grow (up to MAX_SCALE)
        Ctrl+Scroll DOWN = shrink (down to MIN_SCALE)
        Cannot grow larger than the default size (MAX_SCALE = 1.0).
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            current_scale = self._current_scale()
            delta = event.angleDelta().y()
            if delta > 0:
                # Scroll up = grow
                new_scale = min(current_scale + self.SCALE_STEP, self.MAX_SCALE)
            else:
                # Scroll down = shrink
                new_scale = max(current_scale - self.SCALE_STEP, self.MIN_SCALE)

            new_size = int(self._window_size[0] * new_scale)
            # Keep window square
            self.setFixedSize(new_size, new_size)

            # Reposition to keep center stable
            center = self.geometry().center()
            new_rect = self.geometry()
            new_rect.setSize(self.size())
            new_rect.moveCenter(center)
            self.setGeometry(new_rect)

            logger.debug(f"Resized to {new_scale:.2f}x ({new_size}px)")
            event.accept()
        else:
            super().wheelEvent(event)

    # ── SPRINT 4: Peekaboo — Hide behind open windows ───────────
    def _check_window_overlap(self) -> bool:
        """Check if any other window overlaps the character.

        Returns True if the character should hide (peekaboo).
        """
        try:
            from PySide6.QtGui import QWindow
            from PySide6.QtWidgets import QApplication

            my_geo = self.geometry()
            my_center = my_geo.center()

            for widget in QApplication.topLevelWidgets():
                if widget == self or not widget.isVisible() or widget.windowOpacity() < 0.1:
                    continue
                # Skip our own popup
                if hasattr(self, '_hover_popup') and self._hover_popup and widget == self._hover_popup:
                    continue
                other_geo = widget.geometry()
                # Check if other window significantly overlaps the character center
                overlap = other_geo.intersected(my_geo)
                if overlap.isValid() and overlap.width() > 40 and overlap.height() > 40:
                    # Check if we're behind this window (it has focus or is above us)
                    if widget.isActiveWindow() or widget.windowFlags() & Qt.WindowType.WindowStaysOnTopHint:
                        return True
            return False
        except Exception:
            return False

    def _update_peekaboo(self) -> None:
        """Update peekaboo state — hide or show based on window overlap."""
        if self._mouse_over_character or self._is_dragging:
            # Don't peekaboo when user is interacting
            return
        should_hide = self._check_window_overlap()
        if should_hide:
            # Reduce opacity to peekaboo level
            self.setWindowOpacity(0.15)
            self.set_state("hide")
        else:
            # Restore full opacity
            self.setWindowOpacity(1.0)
            if self._current_anim_name == "hide":
                self.set_state("idle")

    # ── SPRINT 3: Physical Guidance ──────────────────────────────
    def physical_guide_to(self, target_screen_pos: QPoint) -> None:
        """Move character to point at a specific screen coordinate.

        Used when the user asks "how do I install this" — the character
        moves to the button coordinates and points at it.

        Args:
            target_screen_pos: Screen coordinates to point at
        """
        my_center = self.geometry().center()
        dx = target_screen_pos.x() - my_center.x()
        dy = target_screen_pos.y() - my_center.y()

        # Move to a position where we can point at the target
        point_x = target_screen_pos.x() - 100  # Stand to the left of target
        point_y = target_screen_pos.y() - 100  # Stand above the target
        target_pos = QPoint(point_x, point_y)

        logger.info(f"Physical guidance to ({target_screen_pos.x()}, {target_screen_pos.y()})")
        self.wander_to(target_pos)

        # Show pointing animation
        self.set_state("point")
        QTimer.singleShot(4000, lambda: self.set_state("idle"))

    # ── SPRINT 4: Synced Animations ──────────────────────────────
    def start_talk_animation(self, tts_duration_ms: int = 3000) -> None:
        """Start talk animation synced with TTS output.

        The talk animation will play for tts_duration_ms then revert
        to idle. Call this when TTS begins speaking.

        Args:
            tts_duration_ms: Expected duration of TTS speech
        """
        self.set_state("talk")
        QTimer.singleShot(tts_duration_ms, lambda: self.set_state("idle"))

    # ── Qt events ─────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        """
        Paint the character with TRUE transparency.

        Uses CompositionMode_Source to ensure the background is fully
        transparent before drawing the sprite. No window mask needed.
        """
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            # ── STEP 1: Clear background to transparent ──
            # CompositionMode_Source replaces the destination pixels entirely,
            # including alpha. This ensures the window background is 100%
            # transparent and NO blue box appears.
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            p.fillRect(self.rect(), QColor(0, 0, 0, 0))

            # ── STEP 2: Draw state tint (if set) ──
            # Use SourceOver so tint blends with sprite
            with self._state_lock:
                tint = self._current_tint
            if tint is not None:
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                p.fillRect(self.rect(), tint)

            # ── STEP 3: Draw sprite frame ──
            # Use SourceOver so sprite alpha blends correctly over transparent bg
            if self._current_anim and self._current_anim.playing:
                pix = self._current_anim.current_pixmap()
                if pix and not pix.isNull():
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                    scaled = pix.scaled(
                        self._window_size[0], self._window_size[1],
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    x = (self._window_size[0] - scaled.width()) // 2
                    y = (self._window_size[1] - scaled.height()) // 2
                    p.drawPixmap(x, y, scaled)

            p.end()

            # ── Close zone indicator (bottom-right X button) ──
            if self._close_zone_enabled and not self._click_through:
                w, h = self.width(), self.height()
                if w > 0 and h > 0:
                    mouse_pos = self.mapFromGlobal(QCursor.pos())
                    hovered = self._is_in_close_zone(mouse_pos)
                    if hovered:
                        p2 = QPainter(self)
                        p2.setRenderHint(QPainter.RenderHint.Antialiasing)
                        zone_rect = self._close_zone_rect()
                        color = self.CLOSE_ZONE_HOVER_COLOR if hovered else self.CLOSE_ZONE_COLOR
                        p2.setPen(Qt.PenStyle.NoPen)
                        p2.setBrush(QBrush(color))
                        p2.drawRoundedRect(zone_rect, 4, 4)
                        # X icon
                        pen = QPen(QColor(255, 255, 255, 240), 2)
                        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                        p2.setPen(pen)
                        margin = 5
                        x0, y0 = zone_rect.left() + margin, zone_rect.top() + margin
                        x1, y1 = zone_rect.right() - margin, zone_rect.bottom() - margin
                        p2.drawLine(x0, y0, x1, y1)
                        p2.drawLine(x1, y0, x0, y1)
                        p2.end()

        except Exception as e:
            logger.debug(f"paintEvent: {e}")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            self._drag_start_pos = pos
            self._is_dragging = False
            # Hide popup on click
            self._hide_hover_popup()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start_pos and not self._is_dragging:
            if (event.pos() - self._drag_start_pos).manhattanLength() > self._drag_threshold:
                self._is_dragging = True
                self.dragging_changed.emit(True)
                self.drag_started_signal.emit(event.pos())
                self._hide_hover_popup()
        if self._is_dragging:
            self.move(event.globalPosition().toPoint() - self._drag_start_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()

            if self._is_dragging:
                self.drag_finished_signal.emit(pos)
                self._is_dragging = False
                self._drag_start_pos = None
                self.dragging_changed.emit(False)
            else:
                if self._is_in_close_zone(pos) and self._close_zone_enabled:
                    logger.info("Close button clicked — shutting down")
                    self.close()
                    QApplication.quit()
                else:
                    self._on_body_click()
        super().mouseReleaseEvent(event)

    def enterEvent(self, event) -> None:
        """Mouse enters the character window."""
        self._mouse_over_character = True

        # Stillness rule: stop wandering when mouse is over character
        if hasattr(self, '_wander_anim') and self._wander_anim:
            try:
                self._wander_anim.stop()
            except Exception:
                pass

        if self._click_through:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Start hover timer for popup (with delay)
        if self._hover_timer and not self._is_dragging:
            self._hover_timer.start()

        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        """Mouse leaves the character window."""
        self._mouse_over_character = False

        # Cancel pending hover popup
        if self._hover_timer:
            self._hover_timer.stop()

        # Hide the popup
        self._hide_hover_popup()

        if self._click_through:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        super().leaveEvent(event)

    def dragEnterEvent(self, event) -> None:
        """Accept drag enter — pass through to popup or handle."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        """Handle file drops directly on the character."""
        if event.mimeData().hasUrls():
            urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            if urls:
                logger.info(f"Files dropped on character: {urls}")
                self._on_files_dropped(urls)
                event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        logger.info("OverlayWindow closing")
        self._sprite_timer.stop()
        self._wander_timer.stop()
        if self._hover_popup:
            self._hover_popup.close()
        if self._animation_widget:
            self._animation_widget.setParent(None)
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        """Handle window resize — reposition settings button and state label."""
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        # Reposition settings button
        if hasattr(self, '_settings_btn'):
            self._settings_btn.move(
                w - self.CLOSE_ZONE_SIZE - 28,
                h - self.CLOSE_ZONE_SIZE - 2,
            )
        # Reposition state label
        if hasattr(self, '_state_label') and self._state_label:
            self._state_label.setGeometry(20, h - 60, w - 40, 40)


# ── Testing ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")
    app = QApplication(sys.argv)
    window = OverlayWindow(size=(400, 400))
    window.show()
    sys.exit(app.exec())