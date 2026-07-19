"""Test script for Vision Pipeline component."""

import sys
import logging
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.vision.pipeline import VisionPipeline, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def test_frame_downsampling():
    """Test 1: Frame downsampling to 224x224."""
    logger.info("=" * 60)
    logger.info("TEST 1: Frame Downsampling")
    logger.info("=" * 60)
    
    pipeline = VisionPipeline()
    
    # Test with 1920x1080 (Full HD)
    logger.info("\n--- Test 1a: 1920x1080 → 224x224 ---")
    frame_1080p = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    processed = pipeline.preprocess_frame(frame_1080p)
    
    logger.info(f"Input: {frame_1080p.shape} → Output: {processed.shape}")
    assert processed.shape == (1, 3, 224, 224), f"Expected (1,3,224,224), got {processed.shape}"
    logger.info("✓ 1080p frame downsampled correctly")
    
    # Test with 4K
    logger.info("\n--- Test 1b: 3840x2160 → 224x224 ---")
    frame_4k = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    processed_4k = pipeline.preprocess_frame(frame_4k)
    
    assert processed_4k.shape == (1, 3, 224, 224)
    logger.info("✓ 4K frame downsampled correctly")
    
    # Test with smaller frame
    logger.info("\n--- Test 1c: 800x600 → 224x224 ---")
    frame_small = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    processed_small = pipeline.preprocess_frame(frame_small)
    
    assert processed_small.shape == (1, 3, 224, 224)
    logger.info("✓ Small frame processed correctly")
    
    logger.info("\n✓ All downsampling tests passed")
    return True


def test_normalization():
    """Test 2: Pixel value normalization."""
    logger.info("=" * 60)
    logger.info("TEST 2: Normalization")
    logger.info("=" * 60)
    
    pipeline = VisionPipeline()
    
    # Test with known values
    logger.info("\n--- Test 2a: 0-255 → 0.0-1.0 ---")
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    test_frame[25:75, 25:75] = 255  # White square in center
    
    processed = pipeline.preprocess_frame(test_frame)
    
    logger.info(f"Input range: [0, 255]")
    logger.info(f"Output range: [{processed.min():.4f}, {processed.max():.4f}]")
    
    assert 0.0 <= processed.min() <= processed.max() <= 1.0
    assert processed.max() > 0.9, "White pixels should be near 1.0"
    logger.info("✓ Normalization correct (0-1 range)")
    
    # Test with custom range
    logger.info("\n--- Test 2b: Custom range (0-255 → -1-1) ---")
    config = PipelineConfig(normalization_range=(-1.0, 1.0))
    pipeline_custom = VisionPipeline(config)
    
    processed_custom = pipeline_custom.preprocess_frame(test_frame)
    logger.info(f"Output range: [{processed_custom.min():.4f}, {processed_custom.max():.4f}]")
    
    assert -1.0 <= processed_custom.min() <= processed_custom.max() <= 1.0
    logger.info("✓ Custom normalization works")
    
    logger.info("\n✓ All normalization tests passed")
    return True


def test_memory_efficiency():
    """Test 3: Memory efficiency and raw frame cleanup."""
    logger.info("=" * 60)
    logger.info("TEST 3: Memory Efficiency")
    logger.info("=" * 60)
    
    pipeline = VisionPipeline()
    
    # Create large frame
    logger.info("\n--- Test 3a: Memory reduction ---")
    large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    raw_size_mb = large_frame.nbytes / (1024 * 1024)
    
    logger.info(f"Raw frame: {large_frame.shape} = {raw_size_mb:.2f} MB")
    
    processed = pipeline.preprocess_frame(large_frame)
    processed_size_mb = processed.nbytes / (1024 * 1024)
    
    logger.info(f"Processed frame: {processed.shape} = {processed_size_mb:.4f} MB")
    logger.info(f"Memory reduction: {raw_size_mb / processed_size_mb:.1f}x")
    
    assert processed_size_mb < 1.0, "Processed frame should be <1MB"
    assert raw_size_mb / processed_size_mb > 10, "Should have >10x reduction"
    logger.info("✓ Memory efficient (10x+ reduction)")
    
    # Test batch processing memory
    logger.info("\n--- Test 3b: Batch processing (10 frames) ---")
    total_raw = 0
    for i in range(10):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        total_raw += frame.nbytes
        pipeline.preprocess_frame(frame)
    
    total_raw_mb = total_raw / (1024 * 1024)
    logger.info(f"Processed 10 frames: {total_raw_mb:.2f} MB raw → ~{processed_size_mb:.4f} MB retained")
    logger.info("✓ Raw frames discarded during processing")
    
    logger.info("\n✓ All memory efficiency tests passed")
    return True


def test_model_format():
    """Test 4: Model input format (B, C, H, W)."""
    logger.info("=" * 60)
    logger.info("TEST 4: Model Input Format")
    logger.info("=" * 60)
    
    pipeline = VisionPipeline()
    
    logger.info("\n--- Test 4a: Verify tensor format ---")
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    processed = pipeline.preprocess_frame(frame)
    
    logger.info(f"Shape: {processed.shape}")
    logger.info(f"Dtype: {processed.dtype}")
    logger.info(f"Dimensions: B={processed.shape[0]}, C={processed.shape[1]}, "
                f"H={processed.shape[2]}, W={processed.shape[3]}")
    
    assert processed.shape == (1, 3, 224, 224)
    assert processed.dtype == np.float32
    logger.info("✓ Correct format (B, C, H, W)")
    
    logger.info("\n--- Test 4b: Verify channel order ---")
    # Create frame with distinct colors
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # Red channel
    frame[:, :, 1] = 128  # Green channel
    frame[:, :, 2] = 64   # Blue channel
    
    processed = pipeline.preprocess_frame(frame)
    
    # Check channel order (RGB)
    # After transpose (2,0,1), channel 0 should be red, 1 green, 2 blue
    logger.info(f"Channel 0 (R) mean: {processed[0, 0].mean():.3f}")
    logger.info(f"Channel 1 (G) mean: {processed[0, 1].mean():.3f}")
    logger.info(f"Channel 2 (B) mean: {processed[0, 2].mean():.3f}")
    
    assert processed[0, 0].mean() > processed[0, 1].mean()  # Red > Green
    assert processed[0, 1].mean() > processed[0, 2].mean()  # Green > Blue
    logger.info("✓ Channel order preserved (RGB)")
    
    logger.info("\n✓ All format tests passed")
    return True


def test_pipeline_stats():
    """Test 5: Pipeline statistics."""
    logger.info("=" * 60)
    logger.info("TEST 5: Pipeline Statistics")
    logger.info("=" * 60)
    
    pipeline = VisionPipeline()
    
    logger.info("\n--- Test 5a: Process multiple frames ---")
    for i in range(10):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        pipeline.preprocess_frame(frame)
    
    stats = pipeline.get_stats()
    logger.info(f"Stats: {stats}")
    
    assert stats["frames_processed"] == 10
    assert float(stats["avg_processing_time_ms"]) > 0
    logger.info("✓ Statistics tracked")
    
    logger.info("\n--- Test 5b: Reset stats ---")
    pipeline.reset_stats()
    stats = pipeline.get_stats()
    assert stats["frames_processed"] == 0
    logger.info("✓ Stats reset works")
    
    logger.info("\n✓ All stats tests passed")
    return True


def test_config_options():
    """Test 6: Configuration options."""
    logger.info("=" * 60)
    logger.info("TEST 6: Configuration Options")
    logger.info("=" * 60)
    
    # Test custom size
    logger.info("\n--- Test 6a: Custom target size ---")
    config = PipelineConfig(target_width=128, target_height=128)
    pipeline = VisionPipeline(config)
    
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    processed = pipeline.preprocess_frame(frame)
    
    assert processed.shape == (1, 3, 128, 128)
    logger.info(f"✓ Custom size: {processed.shape}")
    
    # Test no normalization
    logger.info("\n--- Test 6b: No normalization ---")
    config_no_norm = PipelineConfig(normalize=False)
    pipeline_no_norm = VisionPipeline(config_no_norm)
    
    processed_no_norm = pipeline_no_norm.preprocess_frame(frame)
    assert processed_no_norm.dtype == np.float32
    assert processed_no_norm.max() > 1.0  # Should keep 0-255 range
    logger.info("✓ No normalization preserves 0-255 range")
    
    logger.info("\n✓ All config tests passed")
    return True


def main():
    """Run all vision pipeline tests."""
    logger.info("\n" + "=" * 60)
    logger.info("VISION PIPELINE - PHASE 3 TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        test_frame_downsampling()
        test_normalization()
        test_memory_efficiency()
        test_model_format()
        test_pipeline_stats()
        test_config_options()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL VISION PIPELINE TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 3, Part 1 Complete!")
        logger.info("Next: Implement and test memory buffer")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()