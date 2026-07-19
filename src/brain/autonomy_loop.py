"""Autonomy loop for background decision-making with Active Perception."""

import logging
import time
from typing import Optional, Callable, TYPE_CHECKING
from threading import Event

import numpy as np

from PySide6.QtCore import QThread, Signal, QTimer

from src.brain.decisions import Decision, ScreenContext, DecisionType
from src.brain.behavior_tree import BehaviorTree
from src.brain.context_manager import ContextManager
from src.brain.reasoning import ReasoningBridge, VisualReasoning
from src.vision.change_detector import ScreenChangeDetector, ChangeDetectionResult

logger = logging.getLogger(__name__)


class AutonomyLoop(QThread):
    """
    Background thread for autonomous decision-making.
    
    This runs the behavior tree periodically and emits decisions
    to the UI thread via Qt signals.
    
    Signals:
        decision_made: Emitted when a decision is made
        loop_started: Emitted when the loop starts
        loop_stopped: Emitted when the loop stops
        error_occurred: Emitted when an error occurs
    """
    
    decision_made = Signal(Decision)
    loop_started = Signal()
    loop_stopped = Signal()
    error_occurred = Signal(str)
    
    def __init__(
        self,
        behavior_tree: BehaviorTree,
        context_manager: ContextManager,
        reasoning_bridge: Optional[ReasoningBridge] = None,
        change_detector: Optional[ScreenChangeDetector] = None,
        decision_interval: float = 2.0
    ) -> None:
        """
        Initialize autonomy loop with Smart Sampling.
        
        Args:
            behavior_tree: BehaviorTree instance for decision-making
            context_manager: ContextManager for screen state
            reasoning_bridge: ReasoningBridge for vision integration (optional)
            change_detector: ScreenChangeDetector for event-driven vision (optional)
            decision_interval: Seconds between decisions (default: 2.0)
        """
        super().__init__()
        
        self._tree = behavior_tree
        self._context_manager = context_manager
        self._reasoning_bridge = reasoning_bridge
        self._change_detector = change_detector
        self._decision_interval = decision_interval
        
        # Control flags
        self._stop_event = Event()
        self._is_running = False
        
        # Performance tracking
        self._decisions_made = 0
        self._last_decision_time: Optional[float] = None
        self._errors = 0
        self._vision_checks = 0
        
        # Smart Sampling: Track vision check state
        self._last_vision_check_time: Optional[float] = None
        self._last_change_result: Optional[ChangeDetectionResult] = None
        self._current_vision_mode = "idle"  # "idle", "normal", "alert"
        
        # Smart sampling intervals (seconds)
        self._vision_intervals = {
            "idle": 30.0,      # No activity - check every 30s
            "normal": 5.0,     # Normal activity - check every 5s
            "alert": 2.0       # High activity - check every 2s
        }
        
        logger.info(f"AutonomyLoop initialized (interval={decision_interval}s)")
        logger.info(f"Smart Sampling: enabled (modes: idle={self._vision_intervals['idle']}s, "
                   f"normal={self._vision_intervals['normal']}s, "
                   f"alert={self._vision_intervals['alert']}s)")
    
    def run(self) -> None:
        """Main loop - runs in background thread."""
        logger.info("Autonomy loop started")
        self._is_running = True
        self.loop_started.emit()
        
        while not self._stop_event.is_set():
            try:
                # Make a decision
                self._make_decision()
                
                # Sleep for interval (check stop event periodically)
                self._stop_event.wait(timeout=self._decision_interval)
                
            except Exception as e:
                logger.error(f"Error in autonomy loop: {e}", exc_info=True)
                self._errors += 1
                self.error_occurred.emit(str(e))
                
                # Brief pause on error to prevent tight error loops
                time.sleep(1.0)
        
        self._is_running = False
        logger.info("Autonomy loop stopped")
        self.loop_stopped.emit()
    
    def _make_decision(self) -> None:
        """Evaluate behavior tree with Active Perception and emit decision."""
        try:
            # Build current context
            context = self._context_manager.build_context()
            
            # Active Perception: Only check vision when needed
            if self._should_check_vision(context):
                self._perform_vision_check(context)
            
            # Evaluate behavior tree
            decision = self._tree.evaluate(context)
            
            # "Wait-and-See" Protocol: Block actions if vision is uncertain
            if self._reasoning_bridge and self._is_action_blocked(decision, context):
                logger.info(f"Action blocked by Wait-and-See protocol: {decision}")
                decision = self._create_safe_decision(decision)
            
            # Update stats
            self._decisions_made += 1
            self._last_decision_time = time.time()
            
            # Emit decision to UI thread (thread-safe)
            self.decision_made.emit(decision)
            
            logger.debug(f"Decision #{self._decisions_made}: {decision}")
            
        except Exception as e:
            logger.error(f"Failed to make decision: {e}", exc_info=True)
            self._errors += 1
            self.error_occurred.emit(str(e))
    
    def _should_check_vision(self, context: ScreenContext) -> bool:
        """
        Determine if vision check should be performed (Smart Sampling).
        
        Args:
            context: Current screen context
            
        Returns:
            True if vision check should be performed
        """
        # Always check if we have a reasoning bridge
        if not self._reasoning_bridge:
            return False
        
        # Get current vision interval based on mode
        current_interval = self._vision_intervals.get(self._current_vision_mode, 5.0)
        
        # Check if enough time has passed since last vision check
        if self._last_vision_check_time:
            time_since_last_check = time.time() - self._last_vision_check_time
            if time_since_last_check < current_interval:
                return False
        
        # Use reasoning bridge to determine if check is needed
        return self._reasoning_bridge.should_trigger_vision_check(context)
    
    def _perform_vision_check(self, context: ScreenContext) -> None:
        """
        Perform vision check with Smart Sampling.
        
        Uses reasoning confidence to adjust vision mode (idle/normal/alert),
        replacing the frame-based change detector with confidence-based
        sampling rate control.
        """
        try:
            logger.debug(f"Performing vision check (mode={self._current_vision_mode})")
            
            # Get reasoning from latest observation
            reasoning = self._reasoning_bridge.reason(context)
            
            if reasoning:
                # Update context with vision results
                context.content_type = reasoning.content_type
                context.content_confidence = reasoning.confidence
                
                # ── Smart Sampling: adjust mode based on confidence ──
                if reasoning.confidence >= 0.7:
                    if self._current_vision_mode != "alert":
                        logger.debug(f"Vision mode: {self._current_vision_mode} → alert (high confidence)")
                        self._current_vision_mode = "alert"
                elif reasoning.confidence >= 0.4:
                    if self._current_vision_mode == "alert":
                        logger.debug(f"Vision mode: alert → normal")
                        self._current_vision_mode = "normal"
                else:
                    if self._current_vision_mode != "idle":
                        logger.debug(f"Vision mode: {self._current_vision_mode} → idle (low confidence)")
                        self._current_vision_mode = "idle"
                
                logger.debug(f"Vision check: {reasoning.content_type} "
                           f"(confidence={reasoning.confidence:.2f}, mode={self._current_vision_mode})")
            
            # Update last check time
            self._last_vision_check_time = time.time()
            self._vision_checks += 1
            
        except Exception as e:
            logger.error(f"Vision check failed: {e}", exc_info=True)
    
    def update_change_detection(self, frame: np.ndarray, force_check: bool = False) -> ChangeDetectionResult:
        """
        Update change detection and adjust vision mode.
        
        Args:
            frame: Current screen frame
            force_check: If True, skip min_frame_interval cache
            
        Returns:
            ChangeDetectionResult
        """
        if not self._change_detector:
            # No change detector - use normal mode
            self._current_vision_mode = "normal"
            return ChangeDetectionResult(has_change=False, change_percentage=0.0, avg_mse=0.0, timestamp=time.time())
        
        # Detect change
        change_result = self._change_detector.detect_change(frame, force_check=force_check)
        self._last_change_result = change_result
        
        # Update vision mode based on change
        if change_result.has_change:
            # High activity - switch to alert mode
            if self._current_vision_mode != "alert":
                logger.info(f"Vision mode: {self._current_vision_mode} → alert (screen changed {change_result.change_percentage:.1f}%)")
                self._current_vision_mode = "alert"
        else:
            # No change - gradually return to idle
            if self._current_vision_mode == "alert":
                logger.info(f"Vision mode: alert → normal (screen stable)")
                self._current_vision_mode = "normal"
            elif self._current_vision_mode == "normal":
                logger.info(f"Vision mode: normal → idle (screen stable)")
                self._current_vision_mode = "idle"
        
        return change_result
    
    def get_vision_mode(self) -> str:
        """Get current vision mode."""
        return self._current_vision_mode
    
    def get_vision_interval(self) -> float:
        """Get current vision check interval."""
        return self._vision_intervals.get(self._current_vision_mode, 5.0)
    
    def _is_action_blocked(self, decision: Decision, context: ScreenContext) -> bool:
        """
        Check if decision should be blocked by "Wait-and-See" protocol.
        
        Args:
            decision: Proposed decision
            context: Current screen context
            
        Returns:
            True if action should be blocked
        """
        # Only block INTERACT and OBSERVE_SCREEN actions
        if decision.decision_type not in [DecisionType.INTERACT, DecisionType.OBSERVE_SCREEN]:
            return False
        
        # Block if no reasoning bridge
        if not self._reasoning_bridge:
            return False
        
        # Get latest reasoning
        reasoning = self._reasoning_bridge.get_last_reasoning()
        
        if not reasoning:
            return False
        
        # Block if confidence is low
        if reasoning.confidence < 0.5:
            logger.warning(f"Blocking {decision.decision_type.value} - low confidence "
                          f"({reasoning.confidence:.2f})")
            return True
        
        # Block if content is unknown
        if reasoning.content_type == "unknown":
            logger.warning(f"Blocking {decision.decision_type.value} - unknown content")
            return True
        
        return False
    
    def _create_safe_decision(self, blocked_decision: Decision) -> Decision:
        """
        Create a safe alternative decision when action is blocked.
        
        Args:
            blocked_decision: Original decision that was blocked
            
        Returns:
            Safe alternative decision (IDLE or OBSERVE_SCREEN)
        """
        # Replace INTERACT with OBSERVE_SCREEN
        if blocked_decision.decision_type == DecisionType.INTERACT:
            return Decision(
                decision_type=DecisionType.OBSERVE_SCREEN,
                priority=blocked_decision.priority,
                params={"reason": "wait_and_see"},
                confidence=0.5
            )
        
        # Replace OBSERVE_SCREEN with IDLE
        elif blocked_decision.decision_type == DecisionType.OBSERVE_SCREEN:
            return Decision(
                decision_type=DecisionType.IDLE,
                priority=blocked_decision.priority,
                params={"animation": "idle", "reason": "wait_and_see"},
                confidence=0.5
            )
        
        # Default: return original
        return blocked_decision
    
    def stop(self, timeout: float = 2.0) -> None:
        """
        Stop the autonomy loop.
        
        Args:
            timeout: Maximum time to wait for loop to stop (seconds)
        """
        if not self._is_running:
            logger.warning("Loop not running")
            return
        
        logger.info("Stopping autonomy loop...")
        self._stop_event.set()
        
        # Wait for thread to finish (PySide6 wait takes milliseconds as positional arg)
        if self.isRunning():
            self.wait(int(timeout * 1000))
        
        logger.info("Autonomy loop stopped")
    
    def is_running(self) -> bool:
        """Check if loop is currently running."""
        return self._is_running
    
    def get_stats(self) -> dict:
        """
        Get loop statistics.
        
        Returns:
            Dictionary with loop stats
        """
        return {
            "is_running": self._is_running,
            "decisions_made": self._decisions_made,
            "errors": self._errors,
            "vision_checks": self._vision_checks,
            "last_decision_time": (
                time.time() - self._last_decision_time
                if self._last_decision_time else None
            ),
            "last_vision_check_time": (
                time.time() - self._last_vision_check_time
                if self._last_vision_check_time else None
            ),
            "decision_interval": self._decision_interval,
            "vision_check_interval": self._vision_check_interval
        }
    
    def set_decision_interval(self, interval: float) -> None:
        """
        Change the decision interval.
        
        Args:
            interval: New interval in seconds
        """
        self._decision_interval = max(0.5, min(10.0, interval))
        logger.info(f"Decision interval set to {self._decision_interval}s")


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    from PySide6.QtWidgets import QApplication
    
    from src.brain.behavior_tree import (
        BehaviorTree, Selector, Sequence, Condition, Action
    )
    from src.brain.context_manager import ContextManager
    from src.brain.decisions import NodeStatus, ScreenContext
    
    logger.info("=" * 60)
    logger.info("AUTONOMY LOOP TEST")
    logger.info("=" * 60)
    
    app = QApplication(sys.argv)
    
    # Create a simple behavior tree
    def idle_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    def wander_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    idle_node = Action("Idle", idle_action)
    wander_node = Action("Wander", wander_action)
    
    root = Selector("Root")
    root.add_child(wander_node)
    root.add_child(idle_node)
    
    tree = BehaviorTree(root)
    context_manager = ContextManager()
    
    # Create autonomy loop
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        decision_interval=1.0
    )
    
    # Connect signals
    def on_decision(decision):
        logger.info(f"→ Decision received: {decision}")
    
    def on_started():
        logger.info("→ Loop started")
    
    def on_stopped():
        logger.info("→ Loop stopped")
        app.quit()
    
    def on_error(error):
        logger.error(f"→ Error: {error}")
    
    loop.decision_made.connect(on_decision)
    loop.loop_started.connect(on_started)
    loop.loop_stopped.connect(on_stopped)
    loop.error_occurred.connect(on_error)
    
    # Start loop
    logger.info("Starting autonomy loop...")
    loop.start()
    
    # Run for 5 seconds
    logger.info("Running for 5 seconds...")
    
    # Stop after 5 seconds
    QTimer.singleShot(5000, loop.stop)
    
    sys.exit(app.exec())