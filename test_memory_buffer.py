"""Test script for Memory Buffer component."""

import sys
import logging
import time
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.vision.memory_buffer import MemoryBuffer, Observation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def test_add_observations():
    """Test 1: Adding observations to buffer."""
    logger.info("=" * 60)
    logger.info("TEST 1: Adding Observations")
    logger.info("=" * 60)
    
    buffer = MemoryBuffer(max_observations=10, max_tokens=100, max_age_seconds=60)
    
    logger.info("\n--- Test 1a: Add single observation ---")
    obs1 = buffer.add_observation("User opened Firefox browser")
    assert len(buffer) == 1
    assert obs1.description == "User opened Firefox browser"
    logger.info(f"✓ Added observation: {obs1}")
    
    logger.info("\n--- Test 1b: Add multiple observations ---")
    for i in range(5):
        buffer.add_observation(f"Observation {i}", metadata={"index": i})
    
    assert len(buffer) == 6
    logger.info(f"✓ Buffer contains {len(buffer)} observations")
    
    logger.info("\n--- Test 1c: Auto token estimation ---")
    buffer2 = MemoryBuffer(max_observations=10, max_tokens=100)
    obs_short = buffer2.add_observation("Short")
    obs_long = buffer2.add_observation("This is a much longer description with more words")
    
    logger.info(f"Short description tokens: {obs_short.tokens}")
    logger.info(f"Long description tokens: {obs_long.tokens}")
    assert obs_long.tokens > obs_short.tokens
    logger.info("✓ Token estimation works")
    
    logger.info("\n✓ All add observation tests passed")
    return True


def test_get_recent():
    """Test 2: Getting recent observations."""
    logger.info("=" * 60)
    logger.info("TEST 2: Get Recent Observations")
    logger.info("=" * 60)
    
    buffer = MemoryBuffer()
    
    logger.info("\n--- Test 2a: Add and retrieve recent ---")
    for i in range(10):
        buffer.add_observation(f"Observation {i}")
    
    recent_3 = buffer.get_recent(3)
    assert len(recent_3) == 3
    # get_recent returns newest first
    assert recent_3[0].description == "Observation 9"
    assert recent_3[1].description == "Observation 8"
    assert recent_3[2].description == "Observation 7"
    logger.info(f"✓ Got 3 most recent: {[o.description for o in recent_3]}")
    
    logger.info("\n--- Test 2b: Get more than available ---")
    recent_20 = buffer.get_recent(20)
    assert len(recent_20) == 10
    logger.info(f"✓ Got all {len(recent_20)} observations when requesting 20")
    
    logger.info("\n--- Test 2c: Verify order (newest first) ---")
    recent_1 = buffer.get_recent(1)
    assert recent_1[0].description == "Observation 9"
    logger.info("✓ Most recent observation is first (newest-first order)")
    
    logger.info("\n✓ All get recent tests passed")
    return True


def test_context_window():
    """Test 3: Context window generation."""
    logger.info("=" * 60)
    logger.info("TEST 3: Context Window")
    logger.info("=" * 60)
    
    buffer = MemoryBuffer(max_observations=50, max_tokens=200)
    
    logger.info("\n--- Test 3a: Generate context window ---")
    for i in range(10):
        buffer.add_observation(f"User performed action {i} at time {i}")
    
    context = buffer.get_context_window(max_tokens=50)
    logger.info(f"Context window ({len(context)} chars):")
    logger.info(context)
    
    assert len(context) > 0
    assert "User performed action" in context
    logger.info("✓ Context window generated")
    
    logger.info("\n--- Test 3b: Token limit enforcement ---")
    buffer2 = MemoryBuffer(max_observations=50, max_tokens=20)
    
    # Add observations with known token counts
    buffer2.add_observation("Short one")
    buffer2.add_observation("This is a longer description with more tokens")
    buffer2.add_observation("Another short one")
    
    context2 = buffer2.get_context_window(max_tokens=10)
    logger.info(f"Context with 10 token limit ({len(context2)} chars):")
    logger.info(context2)
    
    # Should only include most recent observations that fit
    assert "Another short one" in context2
    logger.info("✓ Token limit respected")
    
    logger.info("\n--- Test 3c: Empty buffer ---")
    buffer3 = MemoryBuffer()
    context3 = buffer3.get_context_window()
    assert context3 == "No observations yet."
    logger.info("✓ Empty buffer handled correctly")
    
    logger.info("\n✓ All context window tests passed")
    return True


def test_timeline():
    """Test 4: Timeline queries."""
    logger.info("=" * 60)
    logger.info("TEST 4: Timeline Queries")
    logger.info("=" * 60)
    
    buffer = MemoryBuffer()
    
    logger.info("\n--- Test 4a: Get timeline (last 30s) ---")
    for i in range(5):
        buffer.add_observation(f"Observation {i}")
        time.sleep(0.1)
    
    timeline = buffer.get_timeline(seconds=30)
    assert len(timeline) == 5
    logger.info(f"✓ Got {len(timeline)} observations from last 30s")
    
    logger.info("\n--- Test 4b: Timeline with time filter ---")
    timeline_short = buffer.get_timeline(seconds=0.2)
    assert len(timeline_short) < 5
    logger.info(f"✓ Got {len(timeline_short)} observations from last 0.2s")
    
    logger.info("\n--- Test 4c: Timeline order (newest first) ---")
    if timeline:
        assert timeline[0].description == "Observation 4"
        logger.info("✓ Timeline is ordered newest first")
    
    logger.info("\n✓ All timeline tests passed")
    return True


def test_pruning():
    """Test 5: Automatic pruning."""
    logger.info("=" * 60)
    logger.info("TEST 5: Automatic Pruning")
    logger.info("=" * 60)
    
    logger.info("\n--- Test 5a: Max observations limit ---")
    buffer = MemoryBuffer(max_observations=5, max_tokens=1000, max_age_seconds=300)
    
    for i in range(10):
        buffer.add_observation(f"Observation {i}")
    
    assert len(buffer) == 5
    logger.info(f"✓ Buffer limited to {len(buffer)} observations (max=5)")
    
    # Verify oldest were pruned
    descriptions = [obs.description for obs in buffer]
    assert "Observation 0" not in descriptions
    assert "Observation 5" in descriptions
    assert "Observation 9" in descriptions
    logger.info("✓ Oldest observations pruned correctly")
    
    logger.info("\n--- Test 5b: Token limit ---")
    buffer2 = MemoryBuffer(max_observations=50, max_tokens=20, max_age_seconds=300)
    
    # Add observations that exceed token limit
    buffer2.add_observation("One")
    buffer2.add_observation("Two")
    buffer2.add_observation("This is a very long description with many tokens")
    
    stats = buffer2.get_stats()
    assert stats["total_tokens"] <= 20
    logger.info(f"✓ Token limit respected: {stats['total_tokens']} tokens (max=20)")
    
    logger.info("\n--- Test 5c: Age limit ---")
    buffer3 = MemoryBuffer(max_observations=10, max_tokens=100, max_age_seconds=2)
    
    buffer3.add_observation("Old observation")
    logger.info("Added observation, waiting 3 seconds...")
    time.sleep(3)
    
    # Add new observation to trigger pruning
    buffer3.add_observation("New observation")
    
    assert len(buffer3) == 1
    assert buffer3._observations[0].description == "New observation"
    logger.info("✓ Old observation pruned by age")
    
    logger.info("\n✓ All pruning tests passed")
    return True


def test_buffer_operations():
    """Test 6: Buffer operations."""
    logger.info("=" * 60)
    logger.info("TEST 6: Buffer Operations")
    logger.info("=" * 60)
    
    buffer = MemoryBuffer()
    
    logger.info("\n--- Test 6a: Clear buffer ---")
    for i in range(5):
        buffer.add_observation(f"Observation {i}")
    
    assert len(buffer) == 5
    buffer.clear()
    assert len(buffer) == 0
    assert buffer._total_tokens == 0
    logger.info("✓ Buffer cleared successfully")
    
    logger.info("\n--- Test 6b: Buffer statistics ---")
    for i in range(5):
        buffer.add_observation(f"Observation {i}")
    
    stats = buffer.get_stats()
    logger.info(f"Stats: {stats}")
    
    assert stats["total_observations"] == 5
    assert stats["total_tokens"] > 0
    assert stats["max_observations"] == 50
    logger.info("✓ Statistics correct")
    
    logger.info("\n--- Test 6c: Iteration ---")
    count = 0
    for obs in buffer:
        count += 1
    
    assert count == 5
    logger.info("✓ Buffer is iterable")
    
    logger.info("\n--- Test 6d: Length ---")
    assert len(buffer) == 5
    logger.info("✓ Buffer length correct")
    
    logger.info("\n✓ All buffer operation tests passed")
    return True


def test_memory_efficiency():
    """Test 7: Memory efficiency."""
    logger.info("=" * 60)
    logger.info("TEST 7: Memory Efficiency")
    logger.info("=" * 60)
    
    buffer = MemoryBuffer(max_observations=50, max_tokens=2048, max_age_seconds=300)
    
    logger.info("\n--- Test 7a: Text-only storage ---")
    # Add 50 observations
    for i in range(50):
        buffer.add_observation(f"Observation {i} with some description text")
    
    # Estimate memory usage (rough)
    total_chars = sum(len(obs.description) for obs in buffer)
    estimated_bytes = total_chars * 2  # Unicode ~2 bytes per char
    estimated_kb = estimated_bytes / 1024
    
    logger.info(f"50 observations stored")
    logger.info(f"Estimated memory: {estimated_kb:.1f} KB")
    logger.info(f"Buffer stats: {buffer.get_stats()}")
    
    assert estimated_kb < 100, "Buffer should use <100KB for 50 observations"
    logger.info("✓ Memory efficient (<100KB for 50 observations)")
    
    logger.info("\n--- Test 7b: No image storage ---")
    # Verify buffer only stores text, not images
    for obs in buffer:
        assert isinstance(obs.description, str)
        assert not isinstance(obs.description, np.ndarray)
    
    logger.info("✓ Buffer stores only text, not images")
    
    logger.info("\n✓ All memory efficiency tests passed")
    return True


def test_metadata():
    """Test 8: Observation metadata."""
    logger.info("=" * 60)
    logger.info("TEST 8: Observation Metadata")
    logger.info("=" * 60)
    
    buffer = MemoryBuffer()
    
    logger.info("\n--- Test 8a: Add observation with metadata ---")
    metadata = {
        "window": "Firefox",
        "app": "browser",
        "confidence": 0.95,
        "url": "https://example.com"
    }
    
    obs = buffer.add_observation(
        "User browsing example.com",
        metadata=metadata
    )
    
    assert obs.metadata["window"] == "Firefox"
    assert obs.metadata["confidence"] == 0.95
    logger.info(f"✓ Metadata stored: {obs.metadata}")
    
    logger.info("\n--- Test 8b: Serialization ---")
    obs_dict = obs.to_dict()
    assert "timestamp" in obs_dict
    assert "datetime" in obs_dict
    assert "description" in obs_dict
    assert "metadata" in obs_dict
    assert "tokens" in obs_dict
    logger.info(f"✓ Serialized to dict: {list(obs_dict.keys())}")
    
    logger.info("\n--- Test 8c: String representation ---")
    obs_str = str(obs)
    assert "User browsing example.com" in obs_str
    logger.info(f"✓ String representation: {obs_str}")
    
    logger.info("\n✓ All metadata tests passed")
    return True


def main():
    """Run all memory buffer tests."""
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY BUFFER - PHASE 3 TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        test_add_observations()
        test_get_recent()
        test_context_window()
        test_timeline()
        test_pruning()
        test_buffer_operations()
        test_memory_efficiency()
        test_metadata()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL MEMORY BUFFER TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 3, Part 1 Complete!")
        logger.info("Vision pipeline and memory buffer are ready")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()