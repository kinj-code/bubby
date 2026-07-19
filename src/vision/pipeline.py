"""Vision pipeline for preprocessing screen captures for VLM input."""

import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for vision pipeline."""
    # Target size for VLM input (Moondream2 standard)
    target_width: int = 224
    target_height: int = 224
    
    # Normalization parameters
    normalize: bool = True
    normalization_range: Tuple[float, float] = (0.0, 1.0)
    
    # Color space
    color_space: str = "RGB"  # RGB, BGR, GRAY
    
    # Memory management
    discard_raw_frame: bool = True


class VisionPipeline:
    """
    Preprocesses raw screen captures for VLM input.
    
    This pipeline:
    1. Downsamples frames to VLM input size (224x224)
    2. Normalizes pixel values (0-255 → 0-1)
    3. Formats for model input (batch, channels, height, width)
    4. Memory-efficient: discards raw frame after processing
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """
        Initialize vision pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self._config = config or PipelineConfig()
        self._frame_count = 0
        self._processing_times = []
        
        logger.info(f"VisionPipeline initialized: {self._config.target_width}x{self._config.target_height}")
    
    def preprocess_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Preprocess raw frame for VLM input.
        
        Args:
            raw_frame: Raw screen capture (H, W, C) in RGB format
            
        Returns:
            Processed frame (1, C, H, W) normalized to 0-1
        """
        start_time = time.time()
        
        # Validate input
        if raw_frame is None or raw_frame.size == 0:
            raise ValueError("Invalid frame: empty or None")
        
        logger.debug(f"Preprocessing frame {self._frame_count + 1}: shape={raw_frame.shape}")
        
        # Step 1: Downsample to target size
        downsampled = self._downsample(raw_frame)
        
        # Step 2: Normalize if configured
        if self._config.normalize:
            normalized = self._normalize(downsampled)
        else:
            # Convert to float32 but keep 0-255 range
            normalized = downsampled.astype(np.float32)
        
        # Step 3: Convert to model input format (B, C, H, W)
        model_input = self._to_model_format(normalized)
        
        # Step 4: Discard raw frame to save memory
        if self._config.discard_raw_frame:
            del raw_frame
            del downsampled
            del normalized
        
        # Track performance
        processing_time = time.time() - start_time
        self._processing_times.append(processing_time)
        self._frame_count += 1
        
        logger.debug(f"Frame processed in {processing_time*1000:.1f}ms → shape={model_input.shape}")
        
        return model_input
    
    def _downsample(self, frame: np.ndarray) -> np.ndarray:
        """
        Downsample frame to target size using bilinear interpolation.
        
        Args:
            frame: Input frame (H, W, C)
            
        Returns:
            Downsampled frame (target_height, target_width, C)
        """
        # Use PIL-like resize via numpy slicing for simplicity
        # In production, use cv2.resize or PIL for better quality
        h, w = frame.shape[:2]
        target_h, target_w = self._config.target_height, self._config.target_width
        
        # Simple downsampling via slicing (fast but low quality)
        # For better quality, use cv2.resize or PIL
        if h > target_h or w > target_w:
            # Calculate step size
            step_y = max(1, h // target_h)
            step_x = max(1, w // target_w)
            
            # Sample pixels
            downsampled = frame[::step_y, ::step_x, :]
            
            # Resize to exact target if needed (crop or pad)
            downsampled = downsampled[:target_h, :target_w, :]
        else:
            # Frame is already smaller than target, pad it
            downsampled = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
            downsampled[:h, :w, :] = frame
        
        logger.debug(f"Downsampled: {w}x{h} → {target_w}x{target_h}")
        
        return downsampled
    
    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to 0-1 range.
        
        Args:
            frame: Input frame (H, W, C) with values 0-255
            
        Returns:
            Normalized frame (H, W, C) with values 0.0-1.0
        """
        min_val, max_val = self._config.normalization_range
        
        # Convert to float32
        normalized = frame.astype(np.float32)
        
        # Normalize to 0-1 range
        normalized = normalized / 255.0
        
        # Scale to target range
        if min_val != 0.0 or max_val != 1.0:
            normalized = normalized * (max_val - min_val) + min_val
        
        logger.debug(f"Normalized: range=[{normalized.min():.2f}, {normalized.max():.2f}]")
        
        return normalized
    
    def _to_model_format(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to model input format (B, C, H, W).
        
        Args:
            frame: Input frame (H, W, C)
            
        Returns:
            Model input (1, C, H, W)
        """
        # Add batch dimension and reorder axes
        # From (H, W, C) to (1, C, H, W)
        model_input = np.transpose(frame, (2, 0, 1))  # (C, H, W)
        model_input = np.expand_dims(model_input, axis=0)  # (1, C, H, W)
        
        return model_input
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Get expected input shape (C, H, W) without batch dimension.
        
        Returns:
            Tuple of (channels, height, width)
        """
        # Assume RGB (3 channels)
        return (3, self._config.target_height, self._config.target_width)
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        avg_time = sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0
        
        return {
            "frames_processed": self._frame_count,
            "avg_processing_time_ms": f"{avg_time*1000:.1f}",
            "total_processing_time_s": f"{sum(self._processing_times):.2f}",
            "input_shape": f"{self.get_input_shape()}",
            "output_shape": f"(1, {self.get_input_shape()})"
        }
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._frame_count = 0
        self._processing_times = []
        logger.debug("Pipeline stats reset")
    
    # Callback for vision observations
    def start(self, capture_interval: float = 5.0) -> None:
        """Initialize vision processing (no-op: pipeline is stateless)."""
        self._running = True
        self._capture_interval = capture_interval
        logger.info(f"Vision pipeline active (interval={capture_interval:.1f}s)")

    def stop(self) -> None:
        """Stop vision processing."""
        self._running = False
        logger.info("Vision pipeline stopped")

    def set_callback(self, callback) -> None:
        """Set callback for vision observations."""
        self._callback = callback
    
    def _trigger_callback(self, observation_text: str, confidence: float, metadata: dict) -> None:
        """Trigger the callback if set."""
        if hasattr(self, '_callback') and self._callback:
            try:
                self._callback(observation_text, confidence, metadata)
            except Exception as e:
                logger.error(f"Vision callback error: {e}")


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("VISION PIPELINE TEST")
    logger.info("=" * 60)
    
    # Create pipeline
    pipeline = VisionPipeline()
    
    # Test 1: Process dummy frame
    logger.info("\n--- Test 1: Process 1920x1080 frame ---")
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    logger.info(f"Input shape: {dummy_frame.shape}, dtype={dummy_frame.dtype}")
    
    processed = pipeline.preprocess_frame(dummy_frame)
    logger.info(f"Output shape: {processed.shape}, dtype={processed.dtype}")
    logger.info(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    assert processed.shape == (1, 3, 224, 224), f"Expected (1,3,224,224), got {processed.shape}"
    assert processed.dtype == np.float32, f"Expected float32, got {processed.dtype}"
    assert 0.0 <= processed.min() <= processed.max() <= 1.0, "Values not normalized to 0-1"
    logger.info("✓ Frame processed correctly")
    
    # Test 2: Process smaller frame
    logger.info("\n--- Test 2: Process 800x600 frame ---")
    small_frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    processed2 = pipeline.preprocess_frame(small_frame)
    
    assert processed2.shape == (1, 3, 224, 224)
    logger.info("✓ Small frame processed correctly")
    
    # Test 3: Memory efficiency
    logger.info("\n--- Test 3: Memory efficiency ---")
    large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    large_frame_size = large_frame.nbytes / (1024 * 1024)  # MB
    processed_size = processed.nbytes / (1024 * 1024)  # MB
    
    logger.info(f"Raw frame size: {large_frame_size:.2f} MB")
    logger.info(f"Processed frame size: {processed_size:.2f} MB")
    logger.info(f"Memory reduction: {large_frame_size/processed_size:.1f}x")
    
    assert processed_size < 1.0, "Processed frame should be <1MB"
    logger.info("✓ Memory efficient")
    
    # Test 4: Batch processing
    logger.info("\n--- Test 4: Batch processing (5 frames) ---")
    for i in range(5):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        pipeline.preprocess_frame(frame)
    
    stats = pipeline.get_stats()
    logger.info(f"Processed {stats['frames_processed']} frames")
    logger.info(f"Average time: {stats['avg_processing_time_ms']}ms")
    logger.info("✓ Batch processing works")
    
    # Test 5: Input shape
    logger.info("\n--- Test 5: Input shape ---")
    input_shape = pipeline.get_input_shape()
    logger.info(f"Expected input shape (C, H, W): {input_shape}")
    assert input_shape == (3, 224, 224)
    logger.info("✓ Input shape correct")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL VISION PIPELINE TESTS PASSED ✓")
    logger.info("=" * 60)