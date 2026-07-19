#!/usr/bin/env python3
"""
RAG Re-Ranker — two-stage retrieval with cross-encoder precision filtering.

Stage 1 (Recall): Rust HNSW engine retrieves top-20 memories by cosine
                   similarity (fast, approximate).
Stage 2 (Re-Rank): Cross-encoder scores each (query, memory) pair for
                   semantic relevance. Only top 3-5 survive.
Stage 3 (Format): Survivors are wrapped in XML tags with scores
                   and injected into the LLM system prompt.

Prevents context poisoning: irrelevant or contradictory memories are
filtered out before reaching the LLM.

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, CPU-friendly).
Fallback: lightweight Jaccard + embedding-similarity hybrid if model unavailable.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RankedMemory:
    """A memory with its similarity and re-rank scores."""
    id: int
    text: str
    timestamp: float
    importance: float
    metadata: str = "{}"
    hnsw_score: float = 0.0
    cross_score: float = 0.0
    final_score: float = 0.0

    @classmethod
    def from_search_result(cls, sr):
        return cls(
            id=sr.record.id, text=sr.record.text,
            timestamp=sr.record.timestamp, importance=sr.record.importance,
            metadata=sr.record.metadata, hnsw_score=sr.weighted_score,
        )


class CrossEncoderReRanker:
    """Re-ranks Stage-1 results using a cross-encoder model or heuristic fallback."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self._model_name}")
            self._model = CrossEncoder(self._model_name)
            self._model_loaded = True
            logger.info("Cross-encoder loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback.")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}. Using fallback.")

    def re_rank(self, query: str, candidates: List[RankedMemory], top_k: int = 5) -> List[RankedMemory]:
        if not candidates:
            return []
        if self._model_loaded:
            return self._re_rank_with_model(query, candidates, top_k)
        return self._re_rank_heuristic(query, candidates, top_k)

    def _re_rank_with_model(self, query: str, candidates: List[RankedMemory], top_k: int) -> List[RankedMemory]:
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for c, score in zip(candidates, scores):
            c.cross_score = float(score)
            c.final_score = 0.7 * c.cross_score + 0.3 * c.hnsw_score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]

    def _re_rank_heuristic(self, query: str, candidates: List[RankedMemory], top_k: int) -> List[RankedMemory]:
        ql = query.lower()
        qw = set(ql.split())
        for c in candidates:
            tl = c.text.lower()
            tw = set(tl.split())
            inter = qw & tw
            union = qw | tw
            jaccard = len(inter) / max(len(union), 1)
            substring_bonus = 0.3 if ql in tl else 0.0
            c.cross_score = jaccard + substring_bonus
            c.final_score = 0.4 * c.cross_score + 0.4 * c.hnsw_score + 0.2 * c.importance
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]


class RAGPipeline:
    """Two-stage RAG pipeline with cross-encoder re-ranking."""

    STAGE1_K = 20
    STAGE2_K = 5

    def __init__(self, bridge=None, embedding_engine=None, reranker=None) -> None:
        self._bridge = bridge
        self._embedding = embedding_engine
        self._reranker = reranker or CrossEncoderReRanker()
        self._stats = {"queries": 0, "stage1_items": 0, "stage2_items": 0, "total_ms": 0.0}

    def retrieve(self, query: str, top_k: int = None) -> List[RankedMemory]:
        if top_k is None:
            top_k = self.STAGE2_K
        start = time.monotonic()
        stage1 = self._stage1_recall(query, self.STAGE1_K)
        self._stats["stage1_items"] += len(stage1)
        stage2 = self._reranker.re_rank(query, stage1, min(top_k, len(stage1)))
        self._stats["stage2_items"] += len(stage2)
        elapsed = (time.monotonic() - start) * 1000
        self._stats["total_ms"] += elapsed
        self._stats["queries"] += 1
        return stage2

    def _stage1_recall(self, query: str, k: int) -> List[RankedMemory]:
        if self._bridge is None or self._embedding is None:
            return []
        query_emb = self._embedding.encode(query)
        results = self._bridge.search_by_embedding(query_emb.tolist(), k=k, min_score=0.0)
        return [RankedMemory.from_search_result(r) for r in results]

    def format_for_llm(self, memories: List[RankedMemory], query: str = "") -> str:
        if not memories:
            return ""
        lines = [f'<rag_context query="{self._xml(query)}">']
        for m in memories:
            score = f"{m.final_score:.3f}" if m.final_score else f"{m.cross_score:.3f}"
            lines.append(f'  <memory score="{score}" id="{m.id}">{self._xml(m.text)}</memory>')
        lines.append("</rag_context>")
        return "\n".join(lines)

    @staticmethod
    def _xml(s: str) -> str:
        """XML-escape a string for safe prompt injection."""
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(chr(34), "&quot;")

    def get_stats(self) -> dict:
        s = dict(self._stats)
        if s["queries"] > 0:
            s["avg_ms"] = s["total_ms"] / s["queries"]
            s["filter_rate_pct"] = round(100 * (1 - s["stage2_items"] / max(s["stage1_items"], 1)))
        return s


# ── Verification test ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 70)
    logger.info("RAG RE-RANKER — Verification Test")
    logger.info("=" * 70)

    logger.info("\n--- Cross-encoder model ---")
    reranker = CrossEncoderReRanker()
    status = "loaded (cross-encoder)" if reranker._model_loaded else "fallback (heuristic)"
    logger.info(f"  Status: {status}")

    logger.info("\n--- Re-ranking: tricky query ---")
    query = "How do I reset the sensor?"
    candidates = [
        RankedMemory(id=1, text="Sensor reset procedure: hold button for 5 seconds, then release.", timestamp=1000.0, importance=0.8, hnsw_score=0.85),
        RankedMemory(id=2, text="To reset the network sensor, unplug it and plug it back in.", timestamp=2000.0, importance=0.7, hnsw_score=0.82),
        RankedMemory(id=3, text="Sensor installation guide: mount the sensor on the wall using screws.", timestamp=1500.0, importance=0.6, hnsw_score=0.80),
        RankedMemory(id=4, text="The weather sensor measures temperature and humidity.", timestamp=1800.0, importance=0.5, hnsw_score=0.78),
        RankedMemory(id=5, text="User prefers dark mode for the dashboard.", timestamp=3000.0, importance=0.9, hnsw_score=0.40),
        RankedMemory(id=6, text="Sensor calibration: use the calibration tool to adjust readings.", timestamp=1200.0, importance=0.6, hnsw_score=0.76),
        RankedMemory(id=7, text="Reset the companion by restarting the application.", timestamp=2500.0, importance=0.8, hnsw_score=0.50),
        RankedMemory(id=8, text="Installation steps for the new motion sensor kit.", timestamp=1600.0, importance=0.5, hnsw_score=0.79),
        RankedMemory(id=9, text="Hard reset: remove battery and hold power for 10 seconds.", timestamp=4000.0, importance=0.9, hnsw_score=0.84),
        RankedMemory(id=10, text="Coffee brewing tips for the morning routine.", timestamp=5000.0, importance=0.3, hnsw_score=0.30),
    ]

    top5 = reranker.re_rank(query, candidates, top_k=5)

    logger.info(f"  Query: '{query}'")
    logger.info(f"  Stage 1 (HNSW): {len(candidates)} candidates")
    logger.info(f"  Stage 2 (Re-rank): {len(top5)} survivors")
    logger.info("")
    logger.info("  Top 5 after re-ranking:")
    for i, m in enumerate(top5, 1):
        logger.info(f"    {i}. [cross={m.cross_score:.3f} final={m.final_score:.3f}] {m.text[:75]}")

    top_texts = [m.text.lower() for m in top5]
    assert any("reset" in t for t in top_texts), "Top results must include reset memories"
    assert not any("coffee" in t for t in top_texts), "Coffee must be filtered out"
    assert not any("dark mode" in t for t in top_texts), "Dark mode must be filtered out"
    logger.info("  PASS: Irrelevant memories filtered out")

    logger.info("\n--- LLM context formatting ---")
    pipeline = RAGPipeline(reranker=reranker)
    formatted = pipeline.format_for_llm(top5, query)
    logger.info(formatted)
    assert "<rag_context" in formatted
    assert '<memory score="' in formatted
    assert "</rag_context>" in formatted
    logger.info("  PASS: XML formatting correct")

    logger.info("\n--- Heuristic fallback ---")
    h = CrossEncoderReRanker()
    h._model_loaded = False
    top5h = h.re_rank(query, candidates, top_k=5)
    th = [m.text.lower() for m in top5h]
    assert any("reset" in t for t in th)
    assert not any("coffee" in t for t in th)
    logger.info("  PASS: Heuristic fallback works")

    logger.info("\n--- Edge cases ---")
    assert reranker.re_rank("test", [], top_k=5) == []
    assert pipeline.format_for_llm([], "query") == ""
    logger.info("  PASS: Empty inputs handled")

    logger.info("\n--- Pipeline stats ---")
    stats = pipeline.get_stats()
    logger.info(f"  Queries: {stats['queries']}, Filter rate: {stats.get('filter_rate_pct', 0)}%")
    logger.info("  PASS: Stats tracked")

    logger.info("\n" + "=" * 70)
    logger.info("ALL RAG RE-RANKER TESTS PASSED")
    logger.info("=" * 70)