"""Enhanced overlay window module for the transparent desktop companion.

Thread-safe: all state shared with background threads is guarded by
a threading.Lock(). Widget mutations only happen on the main Qt thread
via Signal/slot dispatch.
"""

import logging
import threading
from typing import Optional, Tuple

from PySide6.QtWidgets import QWidget, QApplication, QLabel
from PySide6.QtCore import (
    Qt, QPoint, QTimer, QSize, Signal, QPropertyAnimation,
    QEasingCurve, QRect,
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QCursor,
)

from src.brain.decisions import Decision, DecisionType

logger = logging.getLogger(__name__)


class OverlayWindow(QWidget):
    """
    Frameless, transparent, always-on-top overlay window.

    Features:
    - Click-through mode for non-interactive states
    - Drag-and-drop close zone detection
    - Wayland-compatible transparency
    - Thread-safe state access for cross-thread updates
    """

    # ── Signals (thread-safe dispatch to main thread) ──────────────
    close_requested = Signal()
    drag_started = Signal(QPoint)
    drag_finished = Signal(QPoint)
    click_through_changed = Signal(bool)
    display_message_signal = Signal(str, str, str)  # text, animation, event_type
    update_state_signal = Signal(str, str)           # state_name, message_text
    behavior_state_signal = Signal(object)            # Decision object

    CLOSE_ZONE_SIZE = 80
    CLOSE_ZONE_COLOR = QColor(255, 59, 48, 180)

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

        # Thread-safety lock for state shared with background threads
        self._state_lock = threading.Lock()

        self._setup_window()
        self._setup_close_zone_timer()
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

    def _setup_close_zone_timer(self) -> None:
        self._close_zone_timer = QTimer(self)
        self._close_zone_timer.timeout.connect(self._update_close_zone)
        self._close_zone_timer.start(16)  # ~60 FPS

    # ── Public API ────────────────────────────────────────────────

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
        display_text = text[:100]
        self._update_state_text(display_text)
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
        logger.info(f"Behavior state update: {decision.decision_type.value}")
        if decision.decision_type == DecisionType.WANDER:
            target_x = decision.params.get("target_x", 0)
            target_y = decision.params.get("target_y", 0)
            self.wander_to(QPoint(int(target_x), int(target_y)))
        state_text = decision.decision_type.value.upper()
        self._update_state_text(state_text)
        self._update_state_tint(decision.decision_type)

    def wander_to(self, target: QPoint) -> None:
        safe = self._get_safe_bounds()
        cx = max(safe.left(), min(target.x(), safe.right() - self._window_size[0]))
        cy = max(safe.top(), min(target.y(), safe.bottom() - self._window_size[1]))
        clamped = QPoint(cx, cy)
        if self.pos() == clamped:
            return
        logger.info(f"Wandering to: ({cx}, {cy})")
        anim = QPropertyAnimation(self, b"pos")
        anim.setDuration(2000)
        anim.setStartValue(self.pos())
        anim.setEndValue(clamped)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.start()

    def sizeHint(self) -> QSize:
        return QSize(*self._window_size)

    # ── Internal helpers ──────────────────────────────────────────

    def _is_in_close_zone(self, pos: QPoint) -> bool:
        if not self._close_zone_enabled:
            return False
        z = self.rect().adjusted(-self.CLOSE_ZONE_SIZE, -self.CLOSE_ZONE_SIZE, 0, 0)
        return z.contains(pos)

    def _update_close_zone(self) -> None:
        if not self._close_zone_enabled or self._click_through:
            self.update()
            return
        mouse_pos = self.mapFromGlobal(QCursor.pos())
        if self._is_in_close_zone(mouse_pos):
            self.update()

    def _clear_message(self) -> None:
        if self._state_label:
            self._state_label.setText("")
        with self._state_lock:
            self._current_tint = None
        self.update()

    def _update_state_text(self, text: str) -> None:
        if not hasattr(self, '_state_label') or self._state_label is None:
            self._state_label = QLabel(self)
            self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._state_label.setStyleSheet("""
                QLabel {
                    color: rgba(255, 255, 255, 200);
                    font-size: 24px;
                    font-weight: bold;
                    background: rgba(0, 0, 0, 50);
                    border-radius: 10px;
                    padding: 10px;
                }
            """)
            self._state_label.setGeometry(50, 50, self._window_size[0] - 100, 100)
        self._state_label.setText(text)
        self._state_label.show()

    def _update_state_tint(self, decision_type: DecisionType) -> None:
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
        margin = 100
        return QRect(
            r.left() + margin,
            r.top() + margin,
            r.width() - margin * 2 - self._window_size[0],
            r.height() - margin * 2 - self._window_size[1],
        )

    # ── Qt event overrides ────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Paint close zone indicator and state tint (thread-safe)."""
        try:
            # ── State tint ──
            with self._state_lock:
                tint = self._current_tint
            if tint is not None:
                p = QPainter(self)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.fillRect(self.rect(), tint)
                # Important: end the painter before using another one below
                p.end()

            # ── Close zone ──
            if not self._close_zone_enabled or self._click_through:
                return

            # Guard: window dimensions must be valid
            w = self.width()
            h = self.height()
            if w <= 0 or h <= 0:
                return

            mouse_pos = self.mapFromGlobal(QCursor.pos())
            if not self._is_in_close_zone(mouse_pos):
                return

            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            center = QPoint(w - self.CLOSE_ZONE_SIZE // 2, h - self.CLOSE_ZONE_SIZE // 2)
            radius = self.CLOSE_ZONE_SIZE // 2

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(self.CLOSE_ZONE_COLOR))
            p.drawEllipse(center, radius, radius)

            pen = QPen(QColor(255, 255, 255, 255), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)

            offset = radius // 3
            p.drawLine(
                center.x() - offset, center.y() - offset,
                center.x() + offset, center.y() + offset,
            )
            p.drawLine(
                center.x() + offset, center.y() - offset,
                center.x() - offset, center.y() + offset,
            )
            p.end()
        except Exception as e:
            logger.debug(f"paintEvent suppressed: {e}")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
            self._is_dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start_pos and not self._is_dragging:
            if (event.pos() - self._drag_start_pos).manhattanLength() > self._drag_threshold:
                self._is_dragging = True
                self.drag_started.emit(event.pos())
        if self._is_dragging:
            self.move(event.globalPosition().toPoint() - self._drag_start_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_dragging:
                if self._is_in_close_zone(event.pos()):
                    self.close_requested.emit()
                else:
                    self.drag_finished.emit(event.pos())
                self._is_dragging = False
                self._drag_start_pos = None
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
        if self._animation_widget:
            self._animation_widget.setParent(None)
        super().closeEvent(event)


# ── Testing ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app = QApplication(sys.argv)
    window = OverlayWindow(size=(400, 400), click_through=False)
    window.close_requested.connect(lambda: logger.info("CLOSE REQUESTED"))
    window.show()
    sys.exit(app.exec())