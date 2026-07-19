"""Local embedding engine for semantic text encoding.

Uses sentence-transformers/all-MiniLM-L6-v2 for 100% offline operation.
Falls back to a lightweight numpy-based hash embedding if model unavailable.
"""

import logging
import time
import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Embedding dimension for MiniLM-L6-v2
EMBEDDING_DIM = 384

# Cache file for pre-computed embeddings
EMBEDDING_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "memory"
EMBEDDING_CACHE_FILE = EMBEDDING_CACHE_DIR / "embedding_cache.json"


class EmbeddingEngine:
    """
    Local embedding engine for semantic text encoding.
    
    Primary: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
    Fallback: Lightweight numpy-based hash embedding
    
    Features:
    - 100% offline operation
    - Automatic model download on first use
    - Graceful fallback if model unavailable
    - Embedding caching for frequently seen text
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_fallback: bool = True,
        cache_size: int = 1000
    ) -> None:
        """
        Initialize embedding engine.
        
        Args:
            model_name: Sentence-transformers model name
            use_fallback: If True, use hash embedding when model unavailable
            cache_size: Maximum number of cached embeddings
        """
        self._model_name = model_name
        self._use_fallback = use_fallback
        self._cache_size = cache_size
        
        # State
        self._model = None
        self._model_loaded = False
        self._fallback_active = False
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._total_embeddings = 0
        self._cache_hits = 0
        
        # Ensure cache directory exists
        EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Try to load model
        self._load_model()
        
        logger.info(f"EmbeddingEngine initialized (model={model_name}, "
                   f"fallback={use_fallback}, dim={EMBEDDING_DIM})")
    
    def _load_model(self) -> None:
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self._model_name}")
            start_time = time.time()
            
            self._model = SentenceTransformer(self._model_name)
            self._model_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.1f}s (dim={EMBEDDING_DIM})")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. "
                          "Install with: pip install sentence-transformers")
            if self._use_fallback:
                logger.info("Using fallback hash embedding")
                self._fallback_active = True
            else:
                raise
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            if self._use_fallback:
                logger.info("Using fallback hash embedding")
                self._fallback_active = True
            else:
                raise
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Text to encode
            normalize: If True, L2-normalize the embedding
            
        Returns:
            Embedding vector (384-dim numpy array)
        """
        # Check cache first
        cache_key = text.strip().lower()
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[cache_key].copy()
        
        # Generate embedding
        if self._model_loaded:
            embedding = self._encode_with_model(text)
        elif self._fallback_active:
            embedding = self._encode_fallback(text)
        else:
            raise RuntimeError("No embedding model available")
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # Cache embedding
        if len(self._embedding_cache) < self._cache_size:
            self._embedding_cache[cache_key] = embedding.copy()
        
        self._total_embeddings += 1
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode multiple texts to embedding vectors.
        
        Args:
            texts: List of texts to encode
            normalize: If True, L2-normalize embeddings
            
        Returns:
            Embedding matrix (N x 384)
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        
        if self._model_loaded:
            try:
                embeddings = self._model.encode(texts, normalize_embeddings=normalize)
                self._total_embeddings += len(texts)
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
        
        # Fallback: encode one by one
        embeddings = []
        for text in texts:
            embeddings.append(self.encode(text, normalize=normalize))
        
        return np.array(embeddings, dtype=np.float32)
    
    def _encode_with_model(self, text: str) -> np.ndarray:
        """Encode text using sentence-transformers model."""
        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Model encoding failed: {e}")
            if self._use_fallback:
                return self._encode_fallback(text)
            raise
    
    def _encode_fallback(self, text: str) -> np.ndarray:
        """
        Fallback embedding using feature hashing.
        
        Produces deterministic embeddings based on character n-grams.
        Not as semantically rich as transformer embeddings, but
        provides reasonable similarity for short texts.
        """
        # Normalize text
        text = text.lower().strip()
        
        if not text:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        # Generate embedding using character n-gram hashing
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        # Unigram features
        for i, char in enumerate(text):
            idx = hash(f"1_{char}") % EMBEDDING_DIM
            embedding[idx] += 1.0
        
        # Bigram features
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            idx = hash(f"2_{bigram}") % EMBEDDING_DIM
            embedding[idx] += 1.0
        
        # Trigram features
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            idx = hash(f"3_{trigram}") % EMBEDDING_DIM
            embedding[idx] += 1.0
        
        # Word-level features
        words = text.split()
        for word in words:
            idx = hash(f"w_{word}") % EMBEDDING_DIM
            embedding[idx] += 2.0  # Higher weight for words
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity (0-1)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        similarity = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, similarity))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model": self._model_name,
            "model_loaded": self._model_loaded,
            "fallback_active": self._fallback_active,
            "total_embeddings": self._total_embeddings,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._embedding_cache),
            "embedding_dim": EMBEDDING_DIM
        }


# Testing helper
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("EMBEDDING ENGINE TEST")
    logger.info("=" * 60)
    
    # Create engine
    engine = EmbeddingEngine(use_fallback=True)
    
    # Test 1: Basic encoding
    logger.info("\n--- Test 1: Basic encoding ---")
    emb1 = engine.encode("User likes coffee")
    logger.info(f"Embedding shape: {emb1.shape}")
    assert emb1.shape == (EMBEDDING_DIM,)
    logger.info(f"✓ Embedding generated (dim={EMBEDDING_DIM})")
    
    # Test 2: Similar texts
    logger.info("\n--- Test 2: Similar texts ---")
    sim1 = engine.similarity("User likes coffee", "User enjoys coffee")
    sim2 = engine.similarity("User likes coffee", "The weather is nice")
    logger.info(f"Similar (coffee vs coffee): {sim1:.3f}")
    logger.info(f"Different (coffee vs weather): {sim2:.3f}")
    assert sim1 > sim2, "Similar texts should have higher similarity"
    logger.info("✓ Semantic similarity working")
    
    # Test 3: Batch encoding
    logger.info("\n--- Test 3: Batch encoding ---")
    texts = [
        "User likes coffee",
        "User prefers dark mode",
        "Companion should be friendly"
    ]
    embeddings = engine.encode_batch(texts)
    logger.info(f"Batch shape: {embeddings.shape}")
    assert embeddings.shape == (3, EMBEDDING_DIM)
    logger.info("✓ Batch encoding working")
    
    # Test 4: Cache
    logger.info("\n--- Test 4: Cache ---")
    emb_cached = engine.encode("User likes coffee")
    logger.info(f"Cache hits: {engine._cache_hits}")
    assert engine._cache_hits > 0
    logger.info("✓ Cache working")
    
    # Test 5: Stats
    logger.info("\n--- Test 5: Statistics ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING ENGINE TEST COMPLETE")
    logger.info("=" * 60)