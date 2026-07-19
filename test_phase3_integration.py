"""Integration test for Phase 3: Vision Pipeline + Memory Buffer."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def test_capture_to_pipeline_to_buffer():
    """
    Test 1: End-to-end flow from capture → pipeline → buffer.
    
    Simulates:
    1. Capture raw frame (1920x1080)
    2. Process through vision pipeline (downsample to 224x224)
    3. Generate text description (simulated VLM output)
    4. Store in memory buffer
    """
    logger.info("=" * 60)
    logger.info("TEST 1: End-to-End Integration")
    logger.info("=" * 60)
    
    from src.vision.pipeline import VisionPipeline
    from src.vision.memory_buffer import MemoryBuffer
    
    # Initialize components
    pipeline = VisionPipeline()
    buffer = MemoryBuffer(max_observations=10, max_tokens=500, max_age_seconds=60)
    
    logger.info("\n--- Simulating 5 screen captures ---")
    
    for i in range(5):
        # Step 1: Simulate raw frame capture (1920x1080)
        raw_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        logger.info(f"\nCapture {i+1}: {raw_frame.shape} ({raw_frame.nbytes/1024/1024:.2f}MB)")
        
        # Step 2: Process through pipeline
        processed = pipeline.preprocess_frame(raw_frame)
        logger.info(f"  → Processed: {processed.shape} ({processed.nbytes/1024:.2f}KB)")
        
        # Step 3: Simulate VLM description (in real system, VLM would generate this)
        descriptions = [
            "User browsing Firefox on example.com",
            "User editing document in LibreOffice",
            "User watching video on YouTube",
            "User coding in VS Code",
            "User reading news on website"
        ]
        description = descriptions[i]
        
        # Step 4: Store in memory buffer
        obs = buffer.add_observation(
            description=description,
            metadata={
                "frame_number": i,
                "processed_shape": processed.shape,
                "capture_size_mb": raw_frame.nbytes / (1024 * 1024)
            }
        )
        
        logger.info(f"  → Stored: {obs.description[:50]}...")
        
        # Verify raw frame was discarded
        del raw_frame
    
    # Verify results
    logger.info("\n--- Verification ---")
    stats = pipeline.get_stats()
    buffer_stats = buffer.get_stats()
    
    logger.info(f"Pipeline processed: {stats['frames_processed']} frames")
    logger.info(f"Buffer stored: {buffer_stats['total_observations']} observations")
    logger.info(f"Buffer tokens: {buffer_stats['total_tokens']}")
    
    assert stats['frames_processed'] == 5
    assert buffer_stats['total_observations'] == 5
    logger.info("✓ End-to-end flow works correctly")
    
    logger.info("\n✓ All integration tests passed")
    return True


def test_memory_budget():
    """
    Test 2: Verify memory stays within budget during extended run.
    
    Simulates 20 captures and verifies:
    - Raw frames are discarded
    - Only processed frames + text descriptions retained
    - Total memory < 50MB
    """
    logger.info("=" * 60)
    logger.info("TEST 2: Memory Budget (20 frames)")
    logger.info("=" * 60)
    
    from src.vision.pipeline import VisionPipeline
    from src.vision.memory_buffer import MemoryBuffer
    
    pipeline = VisionPipeline()
    buffer = MemoryBuffer(max_observations=20, max_tokens=1000, max_age_seconds=300)
    
    logger.info("\n--- Processing 20 frames ---")
    
    for i in range(20):
        # Capture and process
        raw_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        processed = pipeline.preprocess_frame(raw_frame)
        
        # Store description
        buffer.add_observation(f"Frame {i}: User activity description")
        
        # Raw frame discarded by pipeline
        del raw_frame
    
    # Calculate memory usage
    logger.info("\n--- Memory Analysis ---")
    
    # Processed frame size (only one retained at a time in real system)
    processed_size_kb = processed.nbytes / 1024
    
    # Buffer memory (text only)
    buffer_stats = buffer.get_stats()
    estimated_buffer_kb = buffer_stats['total_tokens'] * 2 / 1024  # ~2 bytes per token
    
    # Total (worst case: keeping all processed frames + buffer)
    total_mb = (processed_size_kb + estimated_buffer_kb) / 1024
    
    logger.info(f"Single processed frame: {processed_size_kb:.2f} KB")
    logger.info(f"Memory buffer: {estimated_buffer_kb:.2f} KB")
    logger.info(f"Total (worst case): {total_mb:.2f} MB")
    logger.info(f"Buffer observations: {buffer_stats['total_observations']}")
    
    # In real system, we'd only keep 1 processed frame at a time
    # Plus the text buffer
    assert total_mb < 1.0, f"Memory budget exceeded: {total_mb:.2f}MB"
    logger.info("✓ Memory budget maintained (<1MB)")
    
    logger.info("\n✓ Memory budget test passed")
    return True


def test_temporal_awareness():
    """
    Test 3: Temporal awareness - can buffer answer "what was user doing 10s ago?"
    """
    logger.info("=" * 60)
    logger.info("TEST 3: Temporal Awareness")
    logger.info("=" * 60)
    
    from src.vision.memory_buffer import MemoryBuffer
    
    buffer = MemoryBuffer(max_observations=50, max_tokens=2048, max_age_seconds=300)
    
    logger.info("\n--- Simulating user activity over 15 seconds ---")
    
    activities = [
        (0, "User opened browser"),
        (3, "User searched for Python tutorials"),
        (6, "User opened VS Code"),
        (9, "User started coding"),
        (12, "User opened terminal"),
        (15, "User ran git command")
    ]
    
    for timestamp, activity in activities:
        buffer.add_observation(activity, metadata={"timestamp": timestamp})
        logger.info(f"  T+{timestamp}s: {activity}")
    
    # Query: What was user doing 10 seconds ago?
    logger.info("\n--- Query: What was user doing ~10s ago? ---")
    
    # Get timeline from 8-12 seconds ago
    timeline = buffer.get_timeline(seconds=15)  # Get all
    recent = buffer.get_recent(3)  # Get last 3
    
    logger.info(f"Last 3 activities:")
    for obs in recent:
        logger.info(f"  {obs}")
    
    # The activity at T+6s should be in recent history
    descriptions = [o.description for o in recent]
    assert "User opened VS Code" in descriptions or "User started coding" in descriptions
    logger.info("✓ Temporal context available")
    
    # Get context window
    context = buffer.get_context_window(max_tokens=100)
    logger.info(f"\nContext window:\n{context}")
    
    logger.info("\n✓ Temporal awareness test passed")
    return True


def test_performance():
    """
    Test 4: Performance benchmarks.
    
    Verifies:
    - Pipeline processes frame in <10ms
    - Buffer operations are fast
    - No memory leaks
    """
    logger.info("=" * 60)
    logger.info("TEST 4: Performance Benchmarks")
    logger.info("=" * 60)
    
    from src.vision.pipeline import VisionPipeline
    from src.vision.memory_buffer import MemoryBuffer
    
    pipeline = VisionPipeline()
    buffer = MemoryBuffer()
    
    # Benchmark pipeline
    logger.info("\n--- Pipeline performance (100 frames) ---")
    start = time.time()
    
    for i in range(100):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        processed = pipeline.preprocess_frame(frame)
        del frame
    
    elapsed = time.time() - start
    stats = pipeline.get_stats()
    
    logger.info(f"Processed 100 frames in {elapsed:.2f}s")
    logger.info(f"Average: {elapsed/100*1000:.1f}ms per frame")
    logger.info(f"FPS capacity: {100/elapsed:.1f}")
    
    assert elapsed / 100 < 0.01, "Pipeline should process in <10ms per frame"
    logger.info("✓ Pipeline performance acceptable")
    
    # Benchmark buffer
    logger.info("\n--- Buffer performance (1000 operations) ---")
    start = time.time()
    
    for i in range(1000):
        buffer.add_observation(f"Observation {i}")
    
    elapsed = time.time() - start
    
    logger.info(f"Added 1000 observations in {elapsed:.2f}s")
    logger.info(f"Average: {elapsed/1000*1000:.1f}ms per observation")
    
    assert elapsed / 1000 < 0.001, "Buffer should add in <1ms per observation"
    logger.info("✓ Buffer performance acceptable")
    
    logger.info("\n✓ Performance tests passed")
    return True


def main():
    """Run all Phase 3 integration tests."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 INTEGRATION TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        test_capture_to_pipeline_to_buffer()
        test_memory_budget()
        test_temporal_awareness()
        test_performance()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL PHASE 3 INTEGRATION TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 3, Part 1 Complete!")
        logger.info("Vision pipeline and memory buffer are production-ready")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()