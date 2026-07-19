"""Lightweight vector database for memory retrieval (with numpy fallback)."""

import logging
import threading
import time
import json
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Default storage path
DEFAULT_STORAGE_DIR = Path(__file__).parent.parent.parent / "data" / "memory"


@dataclass
class MemoryRecord:
    """
    Single memory record in the vector store.
    
    Attributes:
        id: Unique identifier
        text: Original text content
        timestamp: Unix timestamp when stored
        importance: Importance score (0-1) for prioritization
        metadata: Additional context (source, type, etc.)
    """
    id: int
    text: str
    timestamp: float
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryRecord':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            timestamp=data["timestamp"],
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {})
        )


class VectorStore:
    """
    Lightweight vector database (FAISS with numpy fallback).
    
    Features:
    - Primary: FAISS index for fast similarity search
    - Fallback: numpy-based brute force search (zero extra deps)
    - Persistent storage to disk (data/memory/)
    - Importance-weighted retrieval
    - Automatic index saving/loading
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        storage_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            storage_dir: Directory for persistent storage
        """
        self._embedding_dim = embedding_dim
        self._storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        
        # State
        self._faiss_available = False
        # Use list for O(1) append; batch-convert to ndarray for search
        self._embeddings_list: List[np.ndarray] = []
        self._embeddings: np.ndarray = np.zeros((0, embedding_dim), dtype=np.float32)
        self._embeddings_dirty = False  # True when list and ndarray are out of sync
        self._records: List[MemoryRecord] = []
        self._next_id = 0
        self._total_queries = 0
        self._total_adds = 0
        self._lock = threading.RLock()  # Thread safety for concurrent add/search/save
        self._batch_size = 100  # Flush list to ndarray every N adds
        
        # File paths
        self._embeddings_file = self._storage_dir / "vector_embeddings.npy"
        self._records_file = self._storage_dir / "vector_records.json"
        
        # Ensure storage directory exists
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load FAISS
        self._load_faiss()
        
        # Load existing data
        self._load()
        
        logger.info(f"VectorStore initialized (dim={embedding_dim}, "
                   f"faiss={self._faiss_available}, records={len(self._records)})")
    
    def _load_faiss(self) -> None:
        """Try to load FAISS library."""
        try:
            import faiss
            self._faiss = faiss
            self._faiss_available = True
            logger.info("FAISS available for fast vector search")
        except ImportError:
            logger.info("FAISS not available, using numpy fallback")
            self._faiss_available = False
    
    def add(
        self,
        text: str,
        embedding: np.ndarray,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a memory to the vector store.
        
        Args:
            text: Text content to store
            embedding: Embedding vector (384-dim)
            importance: Importance score (0-1)
            metadata: Additional context
            
        Returns:
            Record ID
        """
        # Create record
        record_id = self._next_id
        self._next_id += 1
        
        record = MemoryRecord(
            id=record_id,
            text=text,
            timestamp=time.time(),
            importance=importance,
            metadata=metadata or {}
        )
        
        # Normalize embedding
        emb_norm = embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(emb_norm)
        if norm > 0:
            emb_norm = emb_norm / norm
        
        with self._lock:
            # O(1) append to list — no vstack reallocation
            self._embeddings_list.append(emb_norm)
            self._embeddings_dirty = True
            self._records.append(record)
            self._total_adds += 1

            # Flush list to contiguous ndarray periodically
            if self._total_adds % self._batch_size == 0:
                self._flush_embeddings()

            # Auto-save periodically (now at batch boundary)
            if self._total_adds % 10 == 0:
                self.save()
        
        return record_id
    
    def _flush_embeddings(self) -> None:
        """Batch-convert embeddings list to contiguous ndarray (O(N) once per batch)."""
        if not self._embeddings_dirty:
            return
        if self._embeddings_list:
            self._embeddings = np.vstack(self._embeddings_list)
        else:
            self._embeddings = np.zeros((0, self._embedding_dim), dtype=np.float32)
        self._embeddings_dirty = False

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Search for most similar memories.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of (record, similarity_score) tuples, sorted by score
        """
        self._total_queries += 1

        with self._lock:
            if len(self._records) == 0:
                return []
            # Ensure ndarray is current before search
            self._flush_embeddings()
            if self._embeddings.shape[0] == 0:
                return []

            n_results = min(k, len(self._records))

            # Normalize query
            query_norm = query_embedding.reshape(1, -1).astype(np.float32)
            norm = np.linalg.norm(query_norm)
            if norm > 0:
                query_norm = query_norm / norm

            if self._faiss_available:
                return self._search_faiss(query_norm, n_results, min_score)
            else:
                return self._search_numpy(query_norm, n_results, min_score)
    
    def _search_faiss(
        self,
        query_norm: np.ndarray,
        k: int,
        min_score: float
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search using FAISS (fast inner product search)."""
        index = self._faiss.IndexFlatIP(self._embedding_dim)
        index.add(self._embeddings)
        distances, indices = index.search(query_norm, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._records):
                continue
            similarity = max(0.0, float(distances[0][i]))
            if similarity < min_score:
                continue
            record = self._records[idx]
            weighted_score = similarity * (0.5 + 0.5 * record.importance)
            results.append((record, weighted_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _search_numpy(
        self,
        query_norm: np.ndarray,
        k: int,
        min_score: float
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search using numpy brute force (zero-dependency fallback)."""
        # Cosine similarity via dot product (both normalized)
        similarities = np.dot(self._embeddings, query_norm.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            similarity = max(0.0, float(similarities[idx]))
            if similarity < min_score:
                continue
            record = self._records[idx]
            weighted_score = similarity * (0.5 + 0.5 * record.importance)
            results.append((record, weighted_score))
        
        return results
    
    def search_by_text(
        self,
        embedding_engine: 'EmbeddingEngine',
        query_text: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search for memories by text query."""
        query_emb = embedding_engine.encode(query_text)
        return self.search(query_emb, k=k, min_score=min_score)
    
    def get_recent(self, n: int = 10) -> List[MemoryRecord]:
        """Get most recent memories (newest first)."""
        sorted_records = sorted(
            self._records,
            key=lambda r: r.timestamp,
            reverse=True
        )
        return sorted_records[:n]
    
    def get_by_importance(self, n: int = 10, min_importance: float = 0.0) -> List[MemoryRecord]:
        """Get most important memories."""
        filtered = [r for r in self._records if r.importance >= min_importance]
        sorted_records = sorted(
            filtered,
            key=lambda r: r.importance,
            reverse=True
        )
        return sorted_records[:n]
    
    def count(self) -> int:
        """Get total number of stored memories."""
        return len(self._records)
    
    def clear(self) -> None:
        """Clear all memories (companion fresh start)."""
        with self._lock:
            self._embeddings_list.clear()
            self._embeddings = np.zeros((0, self._embedding_dim), dtype=np.float32)
            self._embeddings_dirty = False
            self._records.clear()
            self._next_id = 0
            self._total_adds = 0
            self._total_queries = 0
        # Remove persisted files
        if self._embeddings_file.exists():
            self._embeddings_file.unlink()
        if self._records_file.exists():
            self._records_file.unlink()
        logger.info("Vector store cleared")
    
    def save(self) -> None:
        """Save embeddings and records to disk (thread-safe)."""
        with self._lock:
            self._flush_embeddings()
            try:
                np.save(str(self._embeddings_file), self._embeddings)
                records_data = [r.to_dict() for r in self._records]
                with open(self._records_file, 'w') as f:
                    json.dump({
                        "next_id": self._next_id,
                        "records": records_data,
                        "total_adds": self._total_adds,
                        "total_queries": self._total_queries
                    }, f, indent=2)
                logger.debug(f"Saved {len(self._records)} memories to {self._storage_dir}")
            except Exception as e:
                logger.error(f"Failed to save: {e}")
    
    def _load(self) -> None:
        """Load embeddings and records from disk."""
        try:
            if self._embeddings_file.exists():
                self._embeddings = np.load(str(self._embeddings_file))
                logger.info(f"Loaded embeddings ({self._embeddings.shape[0]} vectors)")
            if self._records_file.exists():
                with open(self._records_file, 'r') as f:
                    data = json.load(f)
                self._next_id = data.get("next_id", 0)
                self._total_adds = data.get("total_adds", 0)
                self._total_queries = data.get("total_queries", 0)
                self._records = [MemoryRecord.from_dict(r) for r in data.get("records", [])]
                logger.info(f"Loaded {len(self._records)} memory records")
        except Exception as e:
            logger.warning(f"Failed to load: {e}")
            self._embeddings = np.zeros((0, self._embedding_dim), dtype=np.float32)
            self._records = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_records": len(self._records),
            "total_adds": self._total_adds,
            "total_queries": self._total_queries,
            "faiss_available": self._faiss_available,
            "embedding_dim": self._embedding_dim,
            "storage_dir": str(self._storage_dir),
            "memory_size_kb": self._embeddings.nbytes / 1024 if self._embeddings.size > 0 else 0
        }


# Testing helper
if __name__ == "__main__":
    import sys
    import tempfile
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    from src.memory.embedding import EmbeddingEngine
    
    logger.info("=" * 60)
    logger.info("VECTOR STORE TEST")
    logger.info("=" * 60)
    
    engine = EmbeddingEngine(use_fallback=True)
    test_dir = Path(tempfile.mkdtemp())
    store = VectorStore(embedding_dim=384, storage_dir=test_dir)
    
    # Test 1: Add memories
    logger.info("--- Test 1: Add memories ---")
    memories = [
        "User likes coffee and drinks it every morning",
        "User prefers dark mode in all applications",
        "Companion should be friendly and helpful",
    ]
    for i, mem in enumerate(memories):
        emb = engine.encode(mem)
        store.add(mem, emb, importance=0.9 if "coffee" in mem else 0.5)
    assert store.count() == 3
    logger.info(f"✓ Added {store.count()} memories")
    
    # Test 2: Semantic search
    logger.info("--- Test 2: Semantic search ---")
    results = store.search_by_text(engine, "What does the user drink?", k=3)
    assert len(results) > 0 and "coffee" in results[0][0].text.lower()
    logger.info("✓ Semantic search working")
    
    # Test 3: Persistence
    logger.info("--- Test 3: Persistence ---")
    store.save()
    store2 = VectorStore(embedding_dim=384, storage_dir=test_dir)
    assert store2.count() == 3
    logger.info("✓ Persistence working")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("VECTOR STORE TEST COMPLETE ✓")
    logger.info("=" * 60)