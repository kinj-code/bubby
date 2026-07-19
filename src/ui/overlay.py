"""Enhanced overlay window module for the transparent desktop companion."""

import logging
import random
from typing import Optional, Tuple
from PySide6.QtWidgets import QWidget, QApplication, QLabel
from PySide6.QtCore import (
    Qt, QPoint, QTimer, QSize, Signal, QPropertyAnimation, QEasingCurve, QRect
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QCursor, QFont
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
    - Animation widget support
    """
    
    # Signals for external communication
    close_requested = Signal()
    drag_started = Signal(QPoint)
    drag_finished = Signal(QPoint)
    click_through_changed = Signal(bool)
    # Thread-safe UI update from any thread
    display_message_signal = Signal(str, str, str)  # text, animation, event_type
    update_state_signal = Signal(str, str)           # state_name, message_text
    behavior_state_signal = Signal(object)            # Decision object
    
    # Close zone configuration
    CLOSE_ZONE_SIZE = 80
    CLOSE_ZONE_COLOR = QColor(255, 59, 48, 180)  # Red with alpha
    
    def __init__(
        self,
        size: Tuple[int, int] = (400, 400),
        click_through: bool = False,
        close_zone_enabled: bool = True
    ) -> None:
        super().__init__()
        
        # Window configuration
        self._window_size = size
        self._click_through = click_through
        self._close_zone_enabled = close_zone_enabled
        
        # Drag state
        self._drag_start_pos: Optional[QPoint] = None
        self._is_dragging = False
        self._drag_threshold = 5  # pixels before drag activates
        
        # Animation widget placeholder
        self._animation_widget: Optional[QWidget] = None
        
        # Initialize UI
        self._setup_window()
        self._setup_close_zone_timer()
        
        logger.info(f"OverlayWindow initialized: {size}, click_through={click_through}")
    
    def _setup_window(self) -> None:
        """Configure window flags and attributes for transparent overlay."""
        # Window flags: frameless, always on top, tool window (no taskbar entry)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Transparency attributes
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # Initial click-through state
        self.set_click_through(self._click_through)
        
        # Fixed size
        self.setFixedSize(*self._window_size)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
    
    def _setup_close_zone_timer(self) -> None:
        """Timer for close zone hover detection."""
        self._close_zone_timer = QTimer(self)
        self._close_zone_timer.timeout.connect(self._update_close_zone)
        self._close_zone_timer.start(16)  # ~60 FPS
    
    def set_click_through(self, enabled: bool) -> None:
        """
        Toggle click-through mode.
        
        When enabled, mouse events pass through to windows below.
        When disabled, the window receives mouse events.
        """
        self._click_through = enabled
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            enabled
        )
        self.click_through_changed.emit(enabled)
        logger.debug(f"Click-through set to: {enabled}")
    
    def is_click_through(self) -> bool:
        """Return current click-through state."""
        return self._click_through
    
    def set_animation_widget(self, widget: Optional[QWidget]) -> None:
        """
        Set the animation widget to display in the overlay.
        
        Args:
            widget: QWidget to display (e.g., Lottie animation player)
        """
        if self._animation_widget:
            self._animation_widget.setParent(None)
        
        self._animation_widget = widget
        if widget:
            widget.setParent(self)
            widget.setGeometry(0, 0, *self._window_size)
            widget.show()
        
        logger.debug(f"Animation widget set: {widget is not None}")
    
    def _is_in_close_zone(self, pos: QPoint) -> bool:
        """Check if position is within the close zone (bottom-right corner)."""
        if not self._close_zone_enabled:
            return False
        
        close_zone_rect = self.rect().adjusted(
            -self.CLOSE_ZONE_SIZE,
            -self.CLOSE_ZONE_SIZE,
            0,
            0
        )
        return close_zone_rect.contains(pos)
    
    def _update_close_zone(self) -> None:
        """Update close zone visual feedback."""
        if not self._close_zone_enabled or self._click_through:
            self.update()
            return
        
        # Get current mouse position relative to window
        mouse_pos = self.mapFromGlobal(QCursor.pos())
        
        # Only update if mouse is in close zone
        if self._is_in_close_zone(mouse_pos):
            self.update()
    
    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Paint close zone indicator when hovered."""
        if not self._close_zone_enabled or self._click_through:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get mouse position
        mouse_pos = self.mapFromGlobal(QCursor.pos())
        
        # Draw close zone if hovered
        if self._is_in_close_zone(mouse_pos):
            # Draw circular close button
            center = QPoint(
                self.width() - self.CLOSE_ZONE_SIZE // 2,
                self.height() - self.CLOSE_ZONE_SIZE // 2
            )
            radius = self.CLOSE_ZONE_SIZE // 2
            
            # Background circle
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.CLOSE_ZONE_COLOR))
            painter.drawEllipse(center, radius, radius)
            
            # X mark
            pen = QPen(QColor(255, 255, 255, 255), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            
            offset = radius // 3
            painter.drawLine(
                center.x() - offset, center.y() - offset,
                center.x() + offset, center.y() + offset
            )
            painter.drawLine(
                center.x() + offset, center.y() - offset,
                center.x() - offset, center.y() + offset
            )
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press for drag initiation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
            self._is_dragging = False
            logger.debug(f"Mouse press at: {event.pos()}")
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move for drag detection."""
        if self._drag_start_pos and not self._is_dragging:
            # Check if drag threshold exceeded
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            
            if distance > self._drag_threshold:
                self._is_dragging = True
                self.drag_started.emit(event.pos())
                logger.info("Drag started")
        
        if self._is_dragging:
            # Move window with mouse
            new_pos = event.globalPosition().toPoint() - self._drag_start_pos
            self.move(new_pos)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release for drop detection."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_dragging:
                # Check if dropped in close zone
                if self._is_in_close_zone(event.pos()):
                    logger.info("Dropped in close zone - closing")
                    self.close_requested.emit()
                else:
                    self.drag_finished.emit(event.pos())
                
                self._is_dragging = False
                self._drag_start_pos = None
            else:
                # Click (not drag) - log for testing
                logger.info(f"Click at: {event.pos()}, in_close_zone={self._is_in_close_zone(event.pos())}")
        
        super().mouseReleaseEvent(event)
    
    def enterEvent(self, event) -> None:
        """Handle mouse enter - temporarily disable click-through."""
        if self._click_through:
            # Temporarily enable interaction when mouse enters
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            logger.debug("Mouse entered - click-through disabled")
        
        super().enterEvent(event)
    
    def leaveEvent(self, event) -> None:
        """Handle mouse leave - restore click-through."""
        if self._click_through:
            # Restore click-through when mouse leaves
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            logger.debug("Mouse left - click-through restored")
        
        super().leaveEvent(event)
    
    def closeEvent(self, event) -> None:
        """Clean up on close."""
        logger.info("OverlayWindow closing")
        if self._animation_widget:
            self._animation_widget.setParent(None)
        super().closeEvent(event)
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(*self._window_size)
    
    def update_behavior_state(self, decision: Decision) -> None:
        """
        Update visual state based on brain decision.
        
        Args:
            decision: Decision from autonomy loop
        """
        logger.info(f"Behavior state update: {decision.decision_type.value}")
        
        # Handle movement for WANDER
        if decision.decision_type == DecisionType.WANDER:
            target_x = decision.params.get("target_x", 0)
            target_y = decision.params.get("target_y", 0)
            self.wander_to(QPoint(int(target_x), int(target_y)))
        
        # Update state text
        state_text = decision.decision_type.value.upper()
        self._update_state_text(state_text)
        
        # Update window tint based on state
        self._update_state_tint(decision.decision_type)
    
    def show_message(
        self,
        text: str,
        animation: str = "idle",
        event_type: str = "observation",
        duration_ms: int = 5000,
    ) -> None:
        """
        Display a verbal/text message on the overlay (thread-safe).
        
        This is the main entry point for the verbal pipeline. It receives
        messages from InteractionHandler.display_callback and displays
        them on the overlay without stealing window focus.
        
        Args:
            text: The message text to display
            animation: Animation style (wave/observe/talk/idle/confused)
            event_type: Type of event (greeting/observation/response/status/error)
            duration_ms: How long to show the message (default 5 seconds)
        """
        if not text:
            return
        
        # Truncate for overlay display
        display_text = text[:100]
        
        logger.debug(
            f"Overlay message: [{event_type}] '{display_text}' "
            f"(anim={animation}, duration={duration_ms}ms)"
        )
        
        # Update the state label with the message
        self._update_state_text(display_text)
        
        # Map event type to tint color for visual feedback
        event_tints = {
            "greeting": QColor(255, 200, 200, 60),   # Light pink
            "observation": QColor(200, 180, 255, 60), # Light purple  
            "response": QColor(255, 255, 200, 60),    # Light yellow
            "error": QColor(255, 59, 48, 60),         # Light red
            "status": QColor(200, 200, 200, 30),      # Subtle grey
        }
        tint = event_tints.get(event_type)
        if tint:
            self._current_tint = tint
            self.update()
        
        # Auto-clear message after duration
        QTimer.singleShot(duration_ms, self._clear_message)
    
    def _clear_message(self) -> None:
        """Clear the message display and restore idle state."""
        if hasattr(self, '_state_label'):
            self._state_label.setText("")
        self._current_tint = None
        self.update()
    
    def _update_state_text(self, text: str) -> None:
        """Update the state text displayed in the window."""
        if not hasattr(self, '_state_label'):
            # Create state label on first call
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
        """Update window background tint based on decision type."""
        # Color mapping for different states
        colors = {
            DecisionType.IDLE: None,  # No tint
            DecisionType.WANDER: QColor(173, 216, 230, 30),  # Light blue
            DecisionType.PACE: QColor(144, 238, 144, 30),  # Light green
            DecisionType.SIT: QColor(255, 200, 150, 30),  # Light orange
            DecisionType.OBSERVE_SCREEN: QColor(200, 180, 255, 30),  # Light purple
            DecisionType.INTERACT: QColor(255, 255, 200, 30),  # Light yellow
            DecisionType.GREET: QColor(255, 200, 200, 30),  # Light pink
            DecisionType.SLEEP: QColor(100, 100, 150, 50),  # Dark blue
        }
        
        self._current_tint = colors.get(decision_type)
        self.update()  # Trigger repaint
    
    def wander_to(self, target: QPoint) -> None:
        """
        Safely move window to new position with animation.
        
        Args:
            target: Target position (will be clamped to screen bounds)
        """
        # Get safe bounds
        safe_bounds = self._get_safe_bounds()
        
        # Clamp target to safe bounds
        clamped_x = max(safe_bounds.left(), min(target.x(), safe_bounds.right() - self._window_size[0]))
        clamped_y = max(safe_bounds.top(), min(target.y(), safe_bounds.bottom() - self._window_size[1]))
        clamped_target = QPoint(clamped_x, clamped_y)
        
        # Don't move if already at target
        if self.pos() == clamped_target:
            return
        
        logger.info(f"Wandering to: ({clamped_x}, {clamped_y})")
        
        # Create animation
        animation = QPropertyAnimation(self, b"pos")
        animation.setDuration(2000)  # 2 seconds
        animation.setStartValue(self.pos())
        animation.setEndValue(clamped_target)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Start animation
        animation.start()
    
    def _get_safe_bounds(self) -> QRect:
        """
        Get safe movement boundaries within the screen.
        
        Returns:
            QRect defining safe movement area
        """
        screen = QApplication.primaryScreen()
        if not screen:
            # Fallback if no screen available
            return QRect(0, 0, 1920, 1080)
        
        # Get available screen geometry (excludes taskbars, etc.)
        screen_rect = screen.availableGeometry()
        
        # Add margin to keep window away from edges
        margin = 100
        
        safe_rect = QRect(
            screen_rect.left() + margin,
            screen_rect.top() + margin,
            screen_rect.width() - margin * 2 - self._window_size[0],
            screen_rect.height() - margin * 2 - self._window_size[1]
        )
        
        return safe_rect
    
    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Paint close zone indicator and state tint."""
        # Draw state tint if set
        if hasattr(self, '_current_tint') and self._current_tint:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), self._current_tint)
        
        # Draw close zone (original behavior)
        if not self._close_zone_enabled or self._click_through:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get mouse position
        mouse_pos = self.mapFromGlobal(QCursor.pos())
        
        # Draw close zone if hovered
        if self._is_in_close_zone(mouse_pos):
            # Draw circular close button
            center = QPoint(
                self.width() - self.CLOSE_ZONE_SIZE // 2,
                self.height() - self.CLOSE_ZONE_SIZE // 2
            )
            radius = self.CLOSE_ZONE_SIZE // 2
            
            # Background circle
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.CLOSE_ZONE_COLOR))
            painter.drawEllipse(center, radius, radius)
            
            # X mark
            pen = QPen(QColor(255, 255, 255, 255), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            
            offset = radius // 3
            painter.drawLine(
                center.x() - offset, center.y() - offset,
                center.x() + offset, center.y() + offset
            )
            painter.drawLine(
                center.x() + offset, center.y() - offset,
                center.x() - offset, center.y() + offset
            )


# Testing helper
if __name__ == "__main__":
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(*self._window_size)
    
    def update_behavior_state(self, decision: Decision) -> None:
        """
        Update visual state based on brain decision.
        
        Args:
            decision: Decision from autonomy loop
        """
        logger.info(f"Behavior state update: {decision.decision_type.value}")
        
        # Handle movement for WANDER
        if decision.decision_type == DecisionType.WANDER:
            target_x = decision.params.get("target_x", 0)
            target_y = decision.params.get("target_y", 0)
            self.wander_to(QPoint(int(target_x), int(target_y)))
        
        # Update state text
        state_text = decision.decision_type.value.upper()
        self._update_state_text(state_text)
        
        # Update window tint based on state
        self._update_state_tint(decision.decision_type)
    
    def _update_state_text(self, text: str) -> None:
        """Update the state text displayed in the window."""
        if not hasattr(self, '_state_label'):
            # Create state label on first call
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
        """Update window background tint based on decision type."""
        # Color mapping for different states
        colors = {
            DecisionType.IDLE: None,  # No tint
            DecisionType.WANDER: QColor(173, 216, 230, 30),  # Light blue
            DecisionType.PACE: QColor(144, 238, 144, 30),  # Light green
            DecisionType.SIT: QColor(255, 200, 150, 30),  # Light orange
            DecisionType.OBSERVE_SCREEN: QColor(200, 180, 255, 30),  # Light purple
            DecisionType.INTERACT: QColor(255, 255, 200, 30),  # Light yellow
            DecisionType.GREET: QColor(255, 200, 200, 30),  # Light pink
            DecisionType.SLEEP: QColor(100, 100, 150, 50),  # Dark blue
        }
        
        self._current_tint = colors.get(decision_type)
        self.update()  # Trigger repaint
    
    def wander_to(self, target: QPoint) -> None:
        """
        Safely move window to new position with animation.
        
        Args:
            target: Target position (will be clamped to screen bounds)
        """
        # Get safe bounds
        safe_bounds = self._get_safe_bounds()
        
        # Clamp target to safe bounds
        clamped_x = max(safe_bounds.left(), min(target.x(), safe_bounds.right() - self._window_size[0]))
        clamped_y = max(safe_bounds.top(), min(target.y(), safe_bounds.bottom() - self._window_size[1]))
        clamped_target = QPoint(clamped_x, clamped_y)
        
        # Don't move if already at target
        if self.pos() == clamped_target:
            return
        
        logger.info(f"Wandering to: ({clamped_x}, {clamped_y})")
        
        # Create animation
        animation = QPropertyAnimation(self, b"pos")
        animation.setDuration(2000)  # 2 seconds
        animation.setStartValue(self.pos())
        animation.setEndValue(clamped_target)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Start animation
        animation.start()
    
    def _get_safe_bounds(self) -> QRect:
        """
        Get safe movement boundaries within the screen.
        
        Returns:
            QRect defining safe movement area
        """
        screen = QApplication.primaryScreen()
        if not screen:
            # Fallback if no screen available
            return QRect(0, 0, 1920, 1080)
        
        # Get available screen geometry (excludes taskbars, etc.)
        screen_rect = screen.availableGeometry()
        
        # Add margin to keep window away from edges
        margin = 100
        
        safe_rect = QRect(
            screen_rect.left() + margin,
            screen_rect.top() + margin,
            screen_rect.width() - margin * 2 - self._window_size[0],
            screen_rect.height() - margin * 2 - self._window_size[1]
        )
        
        return safe_rect


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    app = QApplication(sys.argv)
    
    # Create overlay with click-through disabled for testing
    window = OverlayWindow(
        size=(400, 400),
        click_through=False,
        close_zone_enabled=True
    )
    
    # Connect signals
    window.close_requested.connect(lambda: logger.info("CLOSE REQUESTED"))
    window.drag_started.connect(lambda pos: logger.info(f"DRAG STARTED: {pos}"))
    window.drag_finished.connect(lambda pos: logger.info(f"DRAG FINISHED: {pos}"))
    window.click_through_changed.connect(lambda ct: logger.info(f"CLICK-THROUGH: {ct}"))
    
    window.show()
    logger.info("OverlayWindow shown - test with mouse interactions")
    
    sys.exit(app.exec())