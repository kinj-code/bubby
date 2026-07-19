"""Screen change detection for smart sampling."""

import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionResult:
    """Result of change detection analysis."""
    has_change: bool  # True if significant change detected
    change_percentage: float  # Percentage of frame that changed (0-100)
    avg_mse: float  # Mean squared error between frames
    timestamp: float  # When detection was performed
    
    def __str__(self) -> str:
        """Human-readable string."""
        return (f"ChangeDetection(change={self.has_change}, "
                f"pct={self.change_percentage:.1f}%, "
                f"mse={self.avg_mse:.2f})")


class ScreenChangeDetector:
    """
    Detects screen changes using image differencing.
    
    This enables smart sampling:
    - Static screen → Low frequency checks (30s)
    - Active screen → High frequency checks (2s)
    
    Uses Mean Squared Error (MSE) for fast, lightweight comparison.
    """
    
    def __init__(
        self,
        change_threshold: float = 10.0,  # % change to trigger detection
        mse_threshold: float = 100.0,  # MSE threshold for change
        min_frame_interval: float = 1.0  # Minimum time between checks
    ) -> None:
        """
        Initialize change detector.
        
        Args:
            change_threshold: Minimum % change to consider as screen change (0-100)
            mse_threshold: MSE threshold for pixel-wise change detection
            min_frame_interval: Minimum seconds between frame comparisons
        """
        self._change_threshold = change_threshold
        self._mse_threshold = mse_threshold
        self._min_frame_interval = min_frame_interval
        
        # State
        self._last_frame: Optional[np.ndarray] = None
        self._last_analyzed_frame: Optional[np.ndarray] = None
        self._last_check_time: float = 0.0
        self._last_result: Optional[ChangeDetectionResult] = None
        self._total_checks = 0
        self._changes_detected = 0
        
        logger.info(f"ScreenChangeDetector initialized (threshold={change_threshold}%)")
    
    def detect_change(
        self,
        current_frame: np.ndarray,
        force_check: bool = False
    ) -> ChangeDetectionResult:
        """
        Detect if screen has changed significantly.
        
        Args:
            current_frame: Current screen frame (numpy array, RGB)
            force_check: If True, perform check regardless of timing
            
        Returns:
            ChangeDetectionResult with change analysis
        """
        current_time = time.time()
        
        # Check timing (throttle checks)
        if not force_check:
            time_since_last = current_time - self._last_check_time
            if time_since_last < self._min_frame_interval:
                # Return cached result (skip actual comparison)
                return self._create_cached_result()
        
        # Store frame for future comparison
        self._last_frame = current_frame.copy()
        self._last_check_time = current_time
        
        # First frame - no previous to compare
        if self._last_analyzed_frame is None:
            self._last_analyzed_frame = current_frame.copy()
            logger.debug("First frame - storing as baseline")
            result = ChangeDetectionResult(
                has_change=False,
                change_percentage=0.0,
                avg_mse=0.0,
                timestamp=current_time
            )
            self._last_result = result
            return result
        
        # Calculate change
        has_change, change_pct, avg_mse = self._calculate_change(
            self._last_analyzed_frame,
            current_frame
        )
        
        result = ChangeDetectionResult(
            has_change=has_change,
            change_percentage=change_pct,
            avg_mse=avg_mse,
            timestamp=current_time
        )
        
        # Store result for caching
        self._last_result = result
        
        # Update analyzed frame if change detected
        if has_change:
            self._last_analyzed_frame = current_frame.copy()
            self._changes_detected += 1
            logger.debug(f"Screen change detected: {change_pct:.1f}% change")
        
        self._total_checks += 1
        return result
    
    def _calculate_change(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Calculate change between two frames using MSE.
        
        Args:
            prev_frame: Previous analyzed frame
            curr_frame: Current frame
            
        Returns:
            Tuple of (has_change, change_percentage, avg_mse)
        """
        # Ensure frames are same size
        if prev_frame.shape != curr_frame.shape:
            logger.warning(f"Frame size mismatch: {prev_frame.shape} vs {curr_frame.shape}")
            return True, 100.0, float('inf')
        
        # Convert to grayscale for faster comparison
        if len(prev_frame.shape) == 3:
            # RGB to grayscale
            prev_gray = np.mean(prev_frame, axis=2)
            curr_gray = np.mean(curr_frame, axis=2)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((prev_gray - curr_gray) ** 2)
        
        # Normalize MSE to percentage (0-100)
        # MSE range: 0 (identical) to 255*255 (completely different)
        max_mse = 255.0 * 255.0
        change_percentage = min(100.0, (mse / max_mse) * 100.0)
        
        # Determine if change is significant
        has_change = (
            change_percentage > self._change_threshold or
            mse > self._mse_threshold
        )
        
        return has_change, change_percentage, mse
    
    def _create_cached_result(self) -> ChangeDetectionResult:
        """Create cached result from last detection."""
        # Return the last actual result (preserves change detection)
        if self._last_result:
            return self._last_result
        
        # No previous result - return no change
        return ChangeDetectionResult(
            has_change=False,
            change_percentage=0.0,
            avg_mse=0.0,
            timestamp=self._last_check_time
        )
    
    def reset(self) -> None:
        """Reset detector state."""
        self._last_frame = None
        self._last_analyzed_frame = None
        self._last_check_time = 0.0
        self._last_result = None
        self._total_checks = 0
        self._changes_detected = 0
        logger.info("Change detector reset")
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        change_rate = (
            (self._changes_detected / self._total_checks * 100)
            if self._total_checks > 0 else 0.0
        )
        
        return {
            "total_checks": self._total_checks,
            "changes_detected": self._changes_detected,
            "change_rate_pct": f"{change_rate:.1f}%",
            "change_threshold": f"{self._change_threshold}%",
            "mse_threshold": self._mse_threshold
        }
    
    def should_trigger_vision_check(
        self,
        change_result: ChangeDetectionResult,
        time_since_last_vision: float
    ) -> Tuple[bool, str]:
        """
        Determine if vision check should be triggered based on change detection.
        
        Args:
            change_result: Result from detect_change()
            time_since_last_vision: Seconds since last vision check
            
        Returns:
            Tuple of (should_check, reason)
        """
        # High activity: Screen changed
        if change_result.has_change:
            return True, "screen_changed"
        
        # Low activity: No change, use time-based logic
        if time_since_last_vision > 30.0:
            return True, "periodic_check_idle"
        
        # Medium activity: Check periodically
        if time_since_last_vision > 5.0:
            return True, "periodic_check_normal"
        
        # Very recent check, skip
        return False, "too_recent"


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("SCREEN CHANGE DETECTOR TEST")
    logger.info("=" * 60)
    
    # Create detector
    detector = ScreenChangeDetector(
        change_threshold=10.0,  # 10% change
        mse_threshold=100.0,
        min_frame_interval=0.5
    )
    
    # Test 1: Static screen (no change)
    logger.info("\n--- Test 1: Static Screen ---")
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result1 = detector.detect_change(frame1)
    logger.info(f"First frame: {result1}")
    
    result2 = detector.detect_change(frame1)  # Same frame
    logger.info(f"Second frame (identical): {result2}")
    assert result2.has_change == False
    
    # Test 2: Slight change (below threshold)
    logger.info("\n--- Test 2: Slight Change (< 10%) ---")
    frame2 = frame1.copy()
    frame2[0:10, 0:10] = 0  # Small change
    
    result3 = detector.detect_change(frame2)
    logger.info(f"Slight change: {result3}")
    
    # Test 3: Major change (above threshold)
    logger.info("\n--- Test 3: Major Change (> 10%) ---")
    frame3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result4 = detector.detect_change(frame3)
    logger.info(f"Major change: {result4}")
    assert result4.has_change == True
    assert result4.change_percentage > 10.0
    
    # Test 4: Stats
    logger.info("\n--- Test 4: Statistics ---")
    stats = detector.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("SCREEN CHANGE DETECTOR TEST COMPLETE")
    logger.info("=" * 60)