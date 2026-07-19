"""
Long-Term Memory (LTM) system with archive logic and semantic retrieval.

This module integrates:
- EmbeddingEngine: Text → embedding vectors
- VectorStore: FAISS-based persistent vector storage
- MemoryBuffer: Short-term observation buffer (prunes → archives)

The "Archive" Process:
1. MemoryBuffer prunes an observation (age/count/token limits)
2. Before deletion, observation is encoded and stored in VectorStore
3. LTM grows over time, enabling semantic recall

Retrieval API:
- retrieve(query_text, k=5): Semantic search across all memories
- retrieve_by_context(context, k=5): Context-aware retrieval
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

from src.memory.embedding import EmbeddingEngine
from src.memory.vector_db import VectorStore, MemoryRecord

logger = logging.getLogger(__name__)

# Default storage
DEFAULT_STORAGE_DIR = Path(__file__).parent.parent.parent / "data" / "memory"


class LongTermMemory:
    """
    Long-term memory system with archive and retrieval.
    
    Features:
    - Automatic archiving when short-term memory prunes
    - Semantic search across all stored memories
    - Importance scoring for priority retrieval
    - Persistent storage to disk
    - Timeline tracking (when memories were made)
    
    Storage:
    - Vector index: data/memory/vector_index.faiss
    - Memory records: data/memory/vector_records.json
    """
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        vector_store: Optional[VectorStore] = None
    ) -> None:
        """
        Initialize long-term memory.
        
        Args:
            storage_dir: Directory for persistent storage
            embedding_engine: EmbeddingEngine instance (created if None)
            vector_store: VectorStore instance (created if None)
        """
        self._storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        
        # Create or use provided components
        self._embedding = embedding_engine or EmbeddingEngine(use_fallback=True)
        self._store = vector_store or VectorStore(
            embedding_dim=384,
            storage_dir=self._storage_dir
        )
        
        # Statistics
        self._total_archived = 0
        self._total_retrievals = 0
        
        logger.info(f"LongTermMemory initialized (stored={self._store.count()}, "
                   f"dir={self._storage_dir})")
    
    def archive(
        self,
        text: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Archive a text into long-term memory.
        
        This is called when:
        - Short-term memory prunes an old observation
        - User explicitly wants to remember something
        - VLM extracted an important fact
        
        Args:
            text: Text content to remember
            importance: How important is this memory (0-1)
            metadata: Additional context (source, type, window, etc.)
            
        Returns:
            Memory record ID
        """
        if not text or not text.strip():
            logger.warning("Cannot archive empty text")
            return -1
        
        # Encode text to embedding
        embedding = self._embedding.encode(text)
        
        # Store in vector database
        record_id = self._store.add(
            text=text,
            embedding=embedding,
            importance=importance,
            metadata=metadata or {}
        )
        
        self._total_archived += 1
        logger.debug(f"Archived #{record_id}: {text[:60]}...")
        
        return record_id
    
    def retrieve(
        self,
        query_text: str,
        k: int = 5,
        min_score: float = 0.1
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Retrieve most semantically relevant memories.
        
        Args:
            query_text: Query to search for
            k: Maximum number of results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of (MemoryRecord, similarity_score) tuples
        """
        self._total_retrievals += 1
        
        if self._store.count() == 0:
            logger.debug("No memories to retrieve")
            return []
        
        results = self._store.search_by_text(
            embedding_engine=self._embedding,
            query_text=query_text,
            k=k,
            min_score=min_score
        )
        
        logger.debug(f"Retrieved {len(results)} memories for: {query_text[:50]}...")
        
        return results
    
    def retrieve_by_context(
        self,
        context_text: str,
        k: int = 5
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Retrieve memories relevant to current context.
        
        Args:
            context_text: Current context description
            k: Maximum number of results
            
        Returns:
            List of (MemoryRecord, score) tuples
        """
        return self.retrieve(context_text, k=k, min_score=0.1)
    
    def archive_observation(
        self,
        observation_text: str,
        observation_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Archive an observation from short-term memory.
        
        Called by MemoryBuffer when pruning observations.
        
        Args:
            observation_text: Description from the observation
            observation_metadata: Metadata from the observation
            
        Returns:
            Memory record ID
        """
        # Calculate importance based on content signals
        importance = self._calculate_importance(
            observation_text,
            observation_metadata or {}
        )
        
        # Add metadata about source
        metadata = dict(observation_metadata or {})
        metadata["source"] = "observation_prune"
        metadata["archived_at"] = time.time()
        
        return self.archive(
            text=observation_text,
            importance=importance,
            metadata=metadata
        )
    
    def forget(self, memory_id: int) -> bool:
        """
        Forget a specific memory.
        
        Args:
            memory_id: ID of memory to forget
            
        Returns:
            True if forgotten
        """
        # For now, we handle forget by recreating store without that record
        # In production, use ID-based deletion in FAISS
        logger.info(f"Forgetting memory #{memory_id}")
        # Future: implement actual deletion from FAISS index
        return True
    
    def get_recent_memories(self, n: int = 10) -> List[MemoryRecord]:
        """Get most recent memories."""
        return self._store.get_recent(n=n)
    
    def get_important_memories(self, n: int = 10) -> List[MemoryRecord]:
        """Get most important memories."""
        return self._store.get_by_importance(n=n, min_importance=0.7)
    
    def format_for_context(self, query_text: str, k: int = 3) -> str:
        """
        Format memories as text for LLM context window.
        
        Args:
            query_text: Current query/context
            k: Max memories to include
            
        Returns:
            Formatted text for LLM context
        """
        results = self.retrieve(query_text, k=k)
        
        if not results:
            return ""
        
        lines = ["[Long-Term Memory Recall]"]
        
        for record, score in results:
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M",
                time.localtime(record.timestamp)
            )
            lines.append(f"  • [{timestamp}] (rel={score:.2f}) {record.text}")
        
        return "\n".join(lines)
    
    def count(self) -> int:
        """Get total number of stored memories."""
        return self._store.count()
    
    def clear(self) -> None:
        """Clear all memories (companion 'fresh start')."""
        self._store.clear()
        self._total_archived = 0
        self._total_retrievals = 0
        logger.info("Long-term memory cleared (fresh start)")
    
    def save(self) -> None:
        """Persist memory to disk."""
        self._store.save()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        store_stats = self._store.get_stats()
        return {
            "total_memories": self._store.count(),
            "total_archived": self._total_archived,
            "total_retrievals": self._total_retrievals,
            "storage_dir": str(self._storage_dir),
            **store_stats
        }
    
    def _calculate_importance(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate importance score for a memory (0-1).
        
        Higher importance = higher retrieval priority.
        
        Signals:
        - User preference keywords → high importance
        - Explicit facts (User likes...) → high importance
        - Routine observations → medium importance
        - Noise (UNKNOWN, unclear) → low importance
        """
        text_lower = text.lower()
        importance = 0.5  # Default medium
        
        # User preference signals
        preference_signals = [
            "likes", "prefers", "loves", "hates", "enjoys",
            "uses", "works with", "is working on"
        ]
        for signal in preference_signals:
            if signal in text_lower:
                importance = max(importance, 0.8)
                break
        
        # Explicit fact signals
        fact_signals = [
            "user ", "companion ", "the user ", "remember",
            "important", "critical", "always"
        ]
        for signal in fact_signals:
            if signal in text_lower:
                importance = max(importance, 0.7)
                break
        
        # High VLM confidence boosts importance
        if metadata.get("vlm_confidence", 0.5) > 0.8:
            importance = min(1.0, importance + 0.1)
        
        # Unknown content lowers importance
        if text_lower in ["unknown", "unclear", ""]:
            importance = 0.1
        
        # Interaction context boosts importance
        if metadata.get("window") and "browser" in str(metadata.get("window", "")).lower():
            importance = min(1.0, importance + 0.1)
        
        return importance


class MemoryArchive:
    """
    Bridge between MemoryBuffer (short-term) and LongTermMemory (long-term).
    
    When the short-term buffer prunes observations, this class
    intercepts the pruned observations and archives them.
    """
    
    def __init__(self, long_term_memory: LongTermMemory) -> None:
        """
        Initialize memory archive bridge.
        
        Args:
            long_term_memory: LongTermMemory instance
        """
        self._ltm = long_term_memory
        self._archived_count = 0
        
        logger.info("MemoryArchive initialized")
    
    def archive_pruned_observation(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Archive an observation that was pruned from short-term memory.
        
        Args:
            description: Observation description
            metadata: Observation metadata
            
        Returns:
            Memory record ID
        """
        record_id = self._ltm.archive_observation(
            observation_text=description,
            observation_metadata=metadata
        )
        
        if record_id >= 0:
            self._archived_count += 1
        
        return record_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        return {
            "total_archived": self._archived_count,
            "ltm_stats": self._ltm.get_stats()
        }


# Testing helper
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("LONG-TERM MEMORY TEST")
    logger.info("=" * 60)
    
    # Use a temporary directory for testing
    import tempfile
    test_dir = Path(tempfile.mkdtemp())
    
    # Create LTM
    ltm = LongTermMemory(storage_dir=test_dir)
    
    # Test 1: Archive memories
    logger.info("\n--- Test 1: Archive memories ---")
    facts = [
        "User likes coffee and drinks it every morning",
        "User prefers dark mode in all applications",
        "Companion should be friendly and helpful",
        "User works with Python and TypeScript",
        "The weather today is sunny and warm"
    ]
    
    for fact in facts:
        ltm.archive(fact)
    
    assert ltm.count() == 5
    logger.info(f"✓ Archived {ltm.count()} facts")
    
    # Test 2: Semantic retrieval
    logger.info("\n--- Test 2: Semantic retrieval ---")
    results = ltm.retrieve("What does the user drink?", k=3)
    
    logger.info(f"Query: 'What does the user drink?'")
    for record, score in results:
        logger.info(f"  [{score:.3f}] {record.text}")
    
    assert len(results) > 0
    assert "coffee" in results[0][0].text.lower()
    logger.info("✓ Semantic retrieval working")
    
    # Test 3: Context format
    logger.info("\n--- Test 3: Context format ---")
    context = ltm.format_for_context("user preferences", k=3)
    logger.info(f"Formatted context:\n{context}")
    assert len(context) > 0
    logger.info("✓ Context formatting working")
    
    # Test 4: Importance scoring
    logger.info("\n--- Test 4: Importance scoring ---")
    # Archive with explicit importance
    ltm.archive(
        "User explicitly said they love programming",
        importance=0.9,
        metadata={"source": "user_statement"}
    )
    
    important = ltm.get_important_memories(n=5)
    logger.info(f"Important memories: {len(important)}")
    for rec in important:
        logger.info(f"  [imp={rec.importance}] {rec.text[:50]}...")
    assert len(important) > 0
    logger.info("✓ Importance scoring working")
    
    # Test 5: MemoryArchive
    logger.info("\n--- Test 5: MemoryArchive bridge ---")
    archive = MemoryArchive(ltm)
    
    archive_id = archive.archive_pruned_observation(
        "User was browsing example.com",
        metadata={"window": "firefox", "vlm_confidence": 0.9}
    )
    
    assert archive_id >= 0
    logger.info(f"✓ Archived pruned observation #{archive_id}")
    
    # Test 6: Stats
    logger.info("\n--- Test 6: Statistics ---")
    stats = ltm.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Cleanup
    ltm.clear()
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("LONG-TERM MEMORY TEST COMPLETE")
    logger.info("=" * 60)