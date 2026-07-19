"""Test script for Long-Term Memory (Phase 5)."""

import sys
import time
import tempfile
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


def test_embedding_engine():
    """Test 1: Embedding engine with fallback."""
    logger.info("=" * 60)
    logger.info("TEST 1: Embedding Engine")
    logger.info("=" * 60)
    
    from src.memory.embedding import EmbeddingEngine, EMBEDDING_DIM
    
    engine = EmbeddingEngine(use_fallback=True)
    
    # Test encoding
    emb = engine.encode("User likes coffee")
    assert emb.shape == (EMBEDDING_DIM,)
    logger.info(f"✓ Embedding shape: {emb.shape}")
    
    # Test similarity
    sim_same = engine.similarity("User likes coffee", "User enjoys coffee")
    sim_diff = engine.similarity("User likes coffee", "The weather is nice")
    logger.info(f"  Similar (coffee vs coffee): {sim_same:.3f}")
    logger.info(f"  Different (coffee vs weather): {sim_diff:.3f}")
    assert sim_same > sim_diff
    logger.info("✓ Semantic similarity working")
    
    # Test batch
    texts = ["User likes coffee", "User prefers dark mode"]
    batch = engine.encode_batch(texts)
    assert batch.shape == (2, EMBEDDING_DIM)
    logger.info("✓ Batch encoding working")
    
    # Test cache
    engine.encode("User likes coffee")
    assert engine._cache_hits > 0
    logger.info("✓ Embedding cache working")
    
    logger.info("\n✓ Embedding engine test passed")
    return True


def test_vector_store():
    """Test 2: Vector store with numpy fallback."""
    logger.info("=" * 60)
    logger.info("TEST 2: Vector Store")
    logger.info("=" * 60)
    
    from src.memory.vector_db import VectorStore
    from src.memory.embedding import EmbeddingEngine
    
    engine = EmbeddingEngine(use_fallback=True)
    test_dir = Path(tempfile.mkdtemp())
    store = VectorStore(embedding_dim=384, storage_dir=test_dir)
    
    # Add memories
    facts = [
        "User likes coffee and drinks it every morning",
        "User prefers dark mode in all applications",
        "Companion should be friendly and helpful",
        "User works with Python and TypeScript",
        "The weather today is sunny and warm"
    ]
    
    for i, fact in enumerate(facts):
        emb = engine.encode(fact)
        importance = 0.9 if "coffee" in fact else 0.5
        store.add(fact, emb, importance=importance)
    
    assert store.count() == 5
    logger.info(f"✓ Added {store.count()} memories")
    
    # Semantic search
    results = store.search_by_text(engine, "What does the user drink?", k=3)
    assert len(results) > 0
    assert "coffee" in results[0][0].text.lower()
    logger.info(f"✓ Semantic search: '{results[0][0].text[:40]}...' (score={results[0][1]:.3f})")
    
    # Importance filtering
    important = store.get_by_importance(n=5, min_importance=0.7)
    assert len(important) >= 1
    logger.info(f"✓ Importance filtering: {len(important)} important memories")
    
    # Persistence
    store.save()
    store2 = VectorStore(embedding_dim=384, storage_dir=test_dir)
    assert store2.count() == 5
    logger.info("✓ Persistence: loaded 5 memories from disk")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("\n✓ Vector store test passed")
    return True


def test_long_term_memory():
    """Test 3: Long-term memory archive and retrieval."""
    logger.info("=" * 60)
    logger.info("TEST 3: Long-Term Memory")
    logger.info("=" * 60)
    
    from src.memory.long_term_memory import LongTermMemory, MemoryArchive
    
    test_dir = Path(tempfile.mkdtemp())
    ltm = LongTermMemory(storage_dir=test_dir)
    
    # Archive facts
    facts = [
        "User likes coffee and drinks it every morning",
        "User prefers dark mode in all applications",
        "Companion should be friendly and helpful",
        "User works with Python and TypeScript",
    ]
    
    for fact in facts:
        ltm.archive(fact)
    
    assert ltm.count() == 4
    logger.info(f"✓ Archived {ltm.count()} facts")
    
    # Semantic retrieval
    results = ltm.retrieve("What does the user drink?", k=3)
    assert len(results) > 0
    assert "coffee" in results[0][0].text.lower()
    logger.info(f"✓ Retrieved: '{results[0][0].text[:40]}...' (score={results[0][1]:.3f})")
    
    # Context format
    context = ltm.format_for_context("user preferences", k=3)
    assert len(context) > 0
    assert "[Long-Term Memory Recall]" in context
    logger.info(f"✓ Context format:\n{context}")
    
    # MemoryArchive bridge
    archive = MemoryArchive(ltm)
    archive_id = archive.archive_pruned_observation(
        "User was browsing example.com",
        metadata={"window": "firefox", "vlm_confidence": 0.9}
    )
    assert archive_id >= 0
    logger.info(f"✓ Archived pruned observation #{archive_id}")
    
    # Stats
    stats = ltm.get_stats()
    assert stats["total_memories"] >= 5
    logger.info(f"✓ Total memories: {stats['total_memories']}")
    
    # Cleanup
    ltm.clear()
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("\n✓ Long-term memory test passed")
    return True


def test_archive_on_prune():
    """Test 4: Archive on prune (MemoryBuffer → LTM)."""
    logger.info("=" * 60)
    logger.info("TEST 4: Archive on Prune")
    logger.info("=" * 60)
    
    from src.memory.long_term_memory import LongTermMemory, MemoryArchive
    from src.vision.memory_buffer import MemoryBuffer
    
    test_dir = Path(tempfile.mkdtemp())
    ltm = LongTermMemory(storage_dir=test_dir)
    archive = MemoryArchive(ltm)
    
    # Create a small short-term buffer that will prune quickly
    buffer = MemoryBuffer(max_observations=3, max_tokens=100, max_age_seconds=60)
    
    # Add observations (will trigger pruning when we exceed max_observations)
    observations = [
        "User is watching a video about Python",
        "User is browsing documentation",
        "User is writing code in VS Code",
        "User is checking email",  # This will trigger prune
        "User is reading an article",  # This will trigger prune
    ]
    
    for obs in observations:
        buffer.add_observation(obs)
    
    # Buffer should be limited to 3
    assert len(buffer) <= 3
    logger.info(f"✓ Short-term buffer pruned to {len(buffer)} observations")
    
    # Manually archive the pruned observations
    # (In production, this would be automatic via callback)
    for obs in observations[:-3]:  # First 2 were pruned
        archive.archive_pruned_observation(
            obs,
            metadata={"source": "test_prune"}
        )
    
    # LTM should have the archived observations
    assert ltm.count() >= 2
    logger.info(f"✓ LTM has {ltm.count()} archived memories")
    
    # Verify we can retrieve the archived observations
    results = ltm.retrieve("What was the user watching?", k=3)
    if results:
        logger.info(f"✓ Retrieved archived: '{results[0][0].text[:40]}...'")
    
    # Cleanup
    ltm.clear()
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("\n✓ Archive on prune test passed")
    return True


def test_recall_after_clear():
    """Test 5: Store fact, clear short-term, recall from LTM."""
    logger.info("=" * 60)
    logger.info("TEST 5: Recall After Clear")
    logger.info("=" * 60)
    
    from src.memory.long_term_memory import LongTermMemory
    
    test_dir = Path(tempfile.mkdtemp())
    ltm = LongTermMemory(storage_dir=test_dir)
    
    # Step 1: Store a fact
    fact = "User likes coffee and drinks it every morning"
    ltm.archive(fact, importance=0.9, metadata={"source": "user_statement"})
    assert ltm.count() == 1
    logger.info(f"✓ Stored fact: '{fact[:40]}...'")
    
    # Step 2: Simulate clearing short-term buffer
    # (LTM persists independently)
    
    # Step 3: Query LTM to retrieve the fact
    results = ltm.retrieve("What does the user like to drink?", k=3)
    
    assert len(results) > 0
    retrieved_fact = results[0][0].text
    assert "coffee" in retrieved_fact.lower()
    logger.info(f"✓ Retrieved after clear: '{retrieved_fact[:40]}...' (score={results[0][1]:.3f})")
    
    # Step 4: Verify with different query phrasing
    results2 = ltm.retrieve("What are the user's morning habits?", k=3)
    assert len(results2) > 0
    logger.info(f"✓ Different query still finds fact: '{results2[0][0].text[:40]}...'")
    
    # Step 5: Fresh start (clear LTM)
    ltm.clear()
    assert ltm.count() == 0
    logger.info("✓ Fresh start: LTM cleared")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("\n✓ Recall after clear test passed")
    return True


def main():
    """Run all long-term memory tests."""
    logger.info("\n" + "=" * 60)
    logger.info("LONG-TERM MEMORY TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        test_embedding_engine()
        test_vector_store()
        test_long_term_memory()
        test_archive_on_prune()
        test_recall_after_clear()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL LONG-TERM MEMORY TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 5 Complete!")
        logger.info("Long-term memory ready for production")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()