"""Test script for WaylandCapture Phase 1."""

import sys
import logging
import time
import psutil
import os
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

from src.capture.wayland_capture import WaylandCapture, Frame

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def test_capture_basic():
    """Test 1: Basic capture initialization and start/stop."""
    logger.info("=" * 60)
    logger.info("TEST 1: Basic Capture Initialization")
    logger.info("=" * 60)
    
    mem_before = get_process_memory_mb()
    logger.info(f"Memory before capture: {mem_before:.2f} MB")
    
    # Create capture instance
    capture = WaylandCapture(
        source="desktop",
        width=640,
        height=480,
        fps=1
    )
    
    logger.info(f"✓ Capture created: {capture.width}x{capture.height} @ {capture.fps}fps")
    logger.info(f"✓ Source: {capture.source}")
    logger.info(f"✓ Frame queue size: {capture.FRAME_QUEUE_SIZE}")
    
    # Start capture
    logger.info("\nStarting capture...")
    success = capture.start()
    assert success, "Capture should start successfully"
    logger.info("✓ Capture started")
    
    # Let it run for 3 seconds
    logger.info("Running for 3 seconds...")
    time.sleep(3)
    
    mem_during = get_process_memory_mb()
    logger.info(f"Memory during capture: {mem_during:.2f} MB")
    logger.info(f"Memory delta: {mem_during - mem_before:.2f} MB")
    
    # Stop capture
    logger.info("\nStopping capture...")
    capture.stop()
    logger.info("✓ Capture stopped")
    
    mem_after = get_process_memory_mb()
    logger.info(f"Memory after capture: {mem_after:.2f} MB")
    
    return capture


def test_capture_frame_retrieval():
    """Test 2: Frame retrieval and validation."""
    logger.info("=" * 60)
    logger.info("TEST 2: Frame Retrieval")
    logger.info("=" * 60)
    
    capture = WaylandCapture(
        source="desktop",
        width=320,  # Very low res for quick test
        height=240,
        fps=2
    )
    
    capture.start()
    
    frames_received = []
    logger.info("Attempting to grab 5 frames...")
    
    for i in range(5):
        frame = capture.grab_frame(timeout=2.0)
        
        if frame:
            frames_received.append(frame)
            logger.info(
                f"✓ Frame {i+1}: "
                f"{frame.width}x{frame.height} "
                f"({frame.size_mb:.3f}MB) "
                f"frame#{frame.frame_number}"
            )
            
            # Validate frame
            assert isinstance(frame, Frame), "Should return Frame object"
            assert frame.data.shape == (240, 320, 3), f"Wrong shape: {frame.data.shape}"
            assert frame.data.dtype == 'uint8', "Should be uint8"
            assert frame.width == 320, "Width mismatch"
            assert frame.height == 240, "Height mismatch"
        else:
            logger.warning(f"✗ No frame received at attempt {i+1}")
    
    capture.stop()
    
    logger.info(f"\n✓ Successfully retrieved {len(frames_received)}/5 frames")
    assert len(frames_received) > 0, "Should receive at least one frame"
    
    return capture, frames_received


def test_capture_performance():
    """Test 3: Performance and resource usage."""
    logger.info("=" * 60)
    logger.info("TEST 3: Performance & Resource Usage")
    logger.info("=" * 60)
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    capture = WaylandCapture(
        source="desktop",
        width=1280,
        height=720,
        fps=1
    )
    
    logger.info(f"Starting 10-second capture test...")
    logger.info(f"Memory before: {mem_before:.2f} MB")
    
    capture.start()
    start_time = time.time()
    
    frame_sizes = []
    frame_timestamps = []
    
    # Capture for 10 seconds
    try:
        while time.time() - start_time < 10:
            frame = capture.grab_frame(timeout=1.0)
            
            if frame:
                frame_sizes.append(frame.size_mb)
                frame_timestamps.append(frame.timestamp)
                
                if len(frame_sizes) % 5 == 0:
                    logger.info(
                        f"  Captured {len(frame_sizes)} frames, "
                        f"avg size: {sum(frame_sizes)/len(frame_sizes):.2f}MB"
                    )
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        capture.stop()
    
    # Calculate stats
    elapsed = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)
    stats = capture.get_stats()
    
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test duration: {elapsed:.2f}s")
    logger.info(f"Frames captured: {stats['frames_captured']}")
    logger.info(f"Actual FPS: {stats['actual_fps']:.2f}")
    logger.info(f"Target FPS: {stats['target_fps']}")
    logger.info(f"Memory before: {mem_before:.2f} MB")
    logger.info(f"Memory after: {mem_after:.2f} MB")
    logger.info(f"Memory delta: {mem_after - mem_before:.2f} MB")
    
    if frame_sizes:
        logger.info(f"Avg frame size: {sum(frame_sizes)/len(frame_sizes):.2f} MB")
        logger.info(f"Max frame size: {max(frame_sizes):.2f} MB")
    
    # Performance assertions
    assert stats['actual_fps'] > 0, "Should capture at least some frames"
    assert mem_after - mem_before < 100, "Memory usage should be reasonable (<100MB delta)"
    
    logger.info("\n✓ Performance test passed")


def test_capture_queue_overflow():
    """Test 4: Queue overflow handling."""
    logger.info("=" * 60)
    logger.info("TEST 4: Queue Overflow Handling")
    logger.info("=" * 60)
    
    # Create capture with very slow consumer
    capture = WaylandCapture(
        source="desktop",
        width=320,
        height=240,
        fps=10  # High FPS
    )
    
    capture.start()
    logger.info("Started high-FPS capture (10fps)")
    logger.info("Not consuming frames to test queue overflow...")
    
    # Let queue fill up
    time.sleep(2)
    
    stats = capture.get_stats()
    logger.info(f"Queue size: {stats['queue_size']}/{capture.FRAME_QUEUE_SIZE}")
    logger.info(f"Frames captured: {stats['frames_captured']}")
    
    # Should see warnings about full queue in logs
    capture.stop()
    
    logger.info("✓ Queue overflow test complete (check logs for warnings)")


def test_capture_stats():
    """Test 5: Statistics reporting."""
    logger.info("=" * 60)
    logger.info("TEST 5: Statistics Reporting")
    logger.info("=" * 60)
    
    capture = WaylandCapture(
        source="window",
        width=800,
        height=600,
        fps=2
    )
    
    capture.start()
    time.sleep(2)
    
    # Grab a few frames
    for _ in range(3):
        frame = capture.grab_frame(timeout=1.0)
        if frame:
            logger.info(f"Grabbed frame: {frame.frame_number}")
    
    # Get stats
    stats = capture.get_stats()
    
    logger.info("\nCapture Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Validate stats
    assert stats['is_capturing'] == True, "Should be capturing"
    assert stats['frames_captured'] > 0, "Should have captured frames"
    assert stats['target_fps'] == 2, "FPS should match"
    assert stats['source'] == "window", "Source should match"
    assert stats['resolution'] == "800x600", "Resolution should match"
    
    capture.stop()
    
    logger.info("\n✓ Statistics test passed")


def save_sample_frame(capture: WaylandCapture, filename: str = "sample_frame.png") -> None:
    """
    Save a sample frame to disk for visual inspection.
    
    Args:
        capture: WaylandCapture instance
        filename: Output filename
    """
    from PIL import Image
    
    logger.info(f"\nAttempting to save sample frame to {filename}...")
    
    frame = capture.grab_frame(timeout=2.0)
    if frame:
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame.data, 'RGB')
        img.save(filename)
        
        file_size = Path(filename).stat().st_size / 1024
        logger.info(f"✓ Saved frame: {filename} ({file_size:.1f} KB)")
    else:
        logger.warning("No frame available to save")


def main():
    """Run all capture tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BUBBY WAYLAND CAPTURE - PHASE 1 TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        # Test 1: Basic initialization
        capture1 = test_capture_basic()
        
        # Test 2: Frame retrieval
        capture2, frames = test_capture_frame_retrieval()
        
        # Save a sample frame for visual inspection
        if frames:
            save_sample_frame(capture2, "test_frame.png")
        
        # Test 3: Performance
        test_capture_performance()
        
        # Test 4: Queue overflow
        test_capture_queue_overflow()
        
        # Test 5: Statistics
        test_capture_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Check test_frame.png for visual verification")
        logger.info("2. Review logs for any warnings")
        logger.info("3. Verify memory usage is acceptable")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()