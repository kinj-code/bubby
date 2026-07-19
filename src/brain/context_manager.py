"""Context manager for tracking screen state and user presence."""

import logging
import time
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.brain.decisions import ScreenContext

logger = logging.getLogger(__name__)


@dataclass
class UserActivity:
    """Tracks user activity timestamps."""
    last_input_time: float = 0.0
    last_mouse_move: float = 0.0
    last_key_press: float = 0.0
    last_click: float = 0.0
    
    def update_input(self) -> None:
        """Update all input timestamps to now."""
        now = time.time()
        self.last_input_time = now
        self.last_mouse_move = now
        self.last_key_press = now
        self.last_click = now
    
    def get_idle_time(self) -> float:
        """Get seconds since last user input."""
        if self.last_input_time == 0.0:
            return 0.0
        return time.time() - self.last_input_time


class ContextManager:
    """
    Manages screen context and user presence detection.
    
    This provides the behavior tree with current state information.
    In future phases, this will integrate with:
    - Wayland capture for screen content analysis
    - VLM for content classification
    - DBus for active window tracking
    """
    
    # Idle time thresholds (seconds)
    IDLE_THRESHOLD_SHORT = 60.0      # 1 minute
    IDLE_THRESHOLD_MEDIUM = 180.0    # 3 minutes
    IDLE_THRESHOLD_LONG = 300.0      # 5 minutes
    
    # User presence detection
    USER_PRESENT_CHECK_INTERVAL = 5.0  # seconds
    
    def __init__(self) -> None:
        """Initialize context manager."""
        self._user_activity = UserActivity()
        self._last_context: Optional[ScreenContext] = None
        self._last_update_time: float = 0.0
        
        # Window tracking (stub for now)
        self._active_window: str = ""
        self._active_window_class: str = ""
        
        # Screen content (stub for now)
        self._content_type: str = "unknown"
        self._content_confidence: float = 0.0
        
        # System monitoring
        self._cpu_usage: float = 0.0
        self._memory_usage: float = 0.0
        
        logger.info("ContextManager initialized")
    
    def update_user_activity(self) -> None:
        """Call this when user input is detected."""
        self._user_activity.update_input()
        logger.debug("User activity updated")
    
    def get_idle_time(self) -> float:
        """
        Get time since last user input in seconds.
        
        Returns:
            Idle time in seconds
        """
        return self._user_activity.get_idle_time()
    
    def is_user_present(self) -> bool:
        """
        Determine if user is currently present.
        
        Returns:
            True if user has interacted recently
        """
        idle_time = self.get_idle_time()
        return idle_time < self.IDLE_THRESHOLD_MEDIUM
    
    def get_active_window(self) -> str:
        """
        Get currently active window title.
        
        Returns:
            Window title or empty string if unavailable
        """
        # TODO: Implement via DBus or xdotool
        # For now, return stub
        return self._active_window
    
    def get_active_window_class(self) -> str:
        """
        Get currently active window class/application.
        
        Returns:
            Window class or empty string if unavailable
        """
        # TODO: Implement via DBus
        return self._active_window_class
    
    def get_screen_content_type(self) -> tuple[str, float]:
        """
        Classify what's on screen.
        
        Returns:
            Tuple of (content_type, confidence)
            Examples: ("browser", 0.9), ("code", 0.8), ("video", 0.7)
        """
        # TODO: Integrate with VLM in Phase 3
        # For now, return stub
        return self._content_type, self._content_confidence
    
    def get_system_usage(self) -> tuple[float, float]:
        """
        Get current CPU and memory usage.
        
        Returns:
            Tuple of (cpu_percent, memory_percent)
        """
        if PSUTIL_AVAILABLE:
            try:
                self._cpu_usage = psutil.cpu_percent(interval=0.1)
                self._memory_usage = psutil.virtual_memory().percent
            except Exception as e:
                logger.warning(f"Failed to get system usage: {e}")
        
        return self._cpu_usage, self._memory_usage
    
    def build_context(self) -> ScreenContext:
        """
        Build current screen context.
        
        This is called by the autonomy loop before tree evaluation.
        
        Returns:
            ScreenContext with current state
        """
        # Update system usage
        cpu, mem = self.get_system_usage()
        
        # Get screen content
        content_type, content_confidence = self.get_screen_content_type()
        
        # Build context
        context = ScreenContext(
            user_present=self.is_user_present(),
            user_idle_time=self.get_idle_time(),
            active_window=self.get_active_window(),
            active_window_class=self.get_active_window_class(),
            content_type=content_type,
            content_confidence=content_confidence,
            cpu_usage=cpu,
            memory_usage=mem,
            timestamp=time.time()
        )
        
        self._last_context = context
        self._last_update_time = time.time()
        
        logger.debug(f"Context built: {context.to_dict()}")
        
        return context
    
    def get_last_context(self) -> Optional[ScreenContext]:
        """Get the most recent context."""
        return self._last_context
    
    def get_stats(self) -> dict:
        """Get context manager statistics."""
        return {
            "idle_time": f"{self.get_idle_time():.1f}s",
            "user_present": self.is_user_present(),
            "active_window": self._active_window or "unknown",
            "content_type": self._content_type,
            "cpu_usage": f"{self._cpu_usage:.1f}%",
            "memory_usage": f"{self._memory_usage:.1f}%"
        }


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("CONTEXT MANAGER TEST")
    logger.info("=" * 60)
    
    manager = ContextManager()
    
    # Test 1: Initial state
    logger.info("\n--- Test 1: Initial State ---")
    context = manager.build_context()
    logger.info(f"Context: {context.to_dict()}")
    
    # Test 2: After user activity
    logger.info("\n--- Test 2: After User Activity ---")
    time.sleep(1)
    manager.update_user_activity()
    context = manager.build_context()
    logger.info(f"Idle time: {manager.get_idle_time():.1f}s")
    logger.info(f"User present: {manager.is_user_present()}")
    
    # Test 3: After idle period
    logger.info("\n--- Test 3: After Idle Period ---")
    logger.info("Simulating 2 minutes of idle time...")
    manager._user_activity.last_input_time = time.time() - 120
    context = manager.build_context()
    logger.info(f"Idle time: {manager.get_idle_time():.1f}s")
    logger.info(f"User present: {manager.is_user_present()}")
    
    # Test 4: System usage
    logger.info("\n--- Test 4: System Usage ---")
    cpu, mem = manager.get_system_usage()
    logger.info(f"CPU: {cpu:.1f}%, Memory: {mem:.1f}%")
    
    # Test 5: Stats
    logger.info("\n--- Test 5: Statistics ---")
    stats = manager.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Context manager test complete")
    logger.info("=" * 60)