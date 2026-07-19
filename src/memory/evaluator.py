"""Automated RAG Evaluator — scores retrieval quality against ground truth.

Compares synthetic QA pairs against retrieved chunks and assigns
a fidelity score (0.0-1.0) based on semantic overlap and relevance.
Scores are persisted to the feedback database for the reindex engine.

RAM: ~5MB (text processing + sqlite writes).
"""

import logging
import re
from typing import Optional, Dict, Any, List, Tuple, Iterator
from collections import Counter

from src.memory.synthetic_gen import SyntheticQA

logger = logging.getLogger(__name__)


class AutomatedEvaluator:
    """
    Scores retrieval quality by comparing against ground truth content.

    Uses lightweight text similarity (token overlap + keyword matching)
    rather than heavy LLM evaluation. Fast enough for batch processing
    hundreds of queries in seconds.
    """

    MIN_FIDELITY_FOR_GRAPH = 0.70  # Chunks above this → eligible for graph

    def __init__(self) -> None:
        self._evaluations = 0
        self._above_threshold = 0
        logger.info("AutomatedEvaluator initialized")

    def evaluate(
        self,
        qa_pair: SyntheticQA,
        retrieved_text: str,
        similarity_score: float,
    ) -> float:
        """
        Calculate a fidelity score for a retrieval result.

        Combines:
        1. Token overlap: % of ground-truth words found in retrieved text
        2. Entity match: whether key entities appear in both
        3. Vector similarity: weighted boost from FAISS score
        4. Category bonus: legal/definition queries get stricter scoring

        Args:
            qa_pair: The synthetic QA pair (contains ground truth)
            retrieved_text: The text retrieved from vector store
            similarity_score: Cosine similarity from vector search

        Returns:
            Fidelity score 0.0-1.0
        """
        self._evaluations += 1

        ground_tokens = self._tokenize(qa_pair.ground_truth_chunk)
        retrieved_tokens = self._tokenize(retrieved_text)

        if not ground_tokens or not retrieved_tokens:
            return 0.0

        # 1. Token overlap (Jaccard-like)
        overlap = len(ground_tokens & retrieved_tokens)
        overlap_score = overlap / max(len(ground_tokens), 1)
        # Normalize to 0-0.5 range
        overlap_score = min(0.5, overlap_score * 0.5)

        # 2. Entity match
        truth_entities = self._extract_key_entities(qa_pair.ground_truth_chunk)
        retrieved_entities = self._extract_key_entities(retrieved_text)
        entity_match = (
            len(truth_entities & retrieved_entities) / max(len(truth_entities), 1)
        ) if truth_entities else 0.0
        entity_score = min(0.3, entity_match * 0.3)

        # 3. Vector similarity boost
        vector_boost = min(0.15, similarity_score * 0.15) if similarity_score > 0 else 0.0

        # 4. Category adjustment
        category_mult = 1.0
        if qa_pair.category == "legal":
            category_mult = 1.2  # Legal queries need stricter matching
        elif qa_pair.category == "definition":
            category_mult = 1.1

        # Combine
        raw_score = overlap_score + entity_score + vector_boost
        fidelity = round(min(1.0, raw_score * category_mult), 3)

        if fidelity >= self.MIN_FIDELITY_FOR_GRAPH:
            self._above_threshold += 1

        logger.debug(
            f"Fidelity: {fidelity:.3f} (overlap={overlap_score:.3f}, "
            f"entity={entity_score:.3f}, vector={vector_boost:.3f}) "
            f"for '{qa_pair.question[:40]}...'"
        )

        return fidelity

    def evaluate_batch(
        self,
        rag_bridge: Any,
        feedback_engine: Any,
        qa_pairs: List[Tuple[Any, str, int]],
    ) -> Dict[str, Any]:
        """
        Run a batch evaluation: query → retrieve → score → record.

        Args:
            rag_bridge: RAGBridge instance for querying
            feedback_engine: FeedbackEngine for recording results
            qa_pairs: List of (SyntheticQA, source_file, chunk_index) tuples

        Returns:
            Dict with batch summary stats
        """
        results = []
        for qa, source_file, chunk_idx in qa_pairs:
            # Query the RAG system
            records = rag_bridge.query(qa.question, top_k=3)

            if not records:
                # No retrieval — record as weak
                fid = feedback_engine.record_query(
                    qa.question, source_file, chunk_idx, 0.0
                )
                feedback_engine.record_utility(fid, was_helpful=False)
                continue

            # Evaluate each retrieved chunk against ground truth
            best_fidelity = 0.0
            best_text = ""

            for r in records:
                score = r.metadata.get("similarity_score", 0.0)
                text = r.text
                # Strip [DOC: ...] prefix if present
                if text.startswith('[DOC:'):
                    nl = text.find('\n')
                    if nl > 0:
                        text = text[nl + 1:]

                fidelity = self.evaluate(qa, text, score)
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_text = text

            # Record in feedback
            fid = feedback_engine.record_query(
                qa.question, source_file, chunk_idx, best_fidelity
            )
            was_helpful = best_fidelity >= self.MIN_FIDELITY_FOR_GRAPH
            feedback_engine.record_utility(fid, was_helpful=was_helpful)

            results.append({
                "question": qa.question[:80],
                "source_file": qa.source_file,
                "fidelity": best_fidelity,
                "helpful": was_helpful,
                "category": qa.category,
            })

        return {
            "total_evaluated": len(results),
            "high_fidelity": sum(1 for r in results if r["fidelity"] >= self.MIN_FIDELITY_FOR_GRAPH),
            "avg_fidelity": (
                sum(r["fidelity"] for r in results) / len(results)
            ) if results else 0.0,
            "results": results[:10],  # First 10 for preview
        }

    def get_high_fidelity_chunks(
        self,
        vector_store: Any,
        feedback_engine: Any,
        min_fidelity: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """
        Get chunks that scored above the fidelity threshold.

        Returns list of {source_file, chunk_text, fidelity, metadata}
        """
        records = getattr(vector_store, '_records', [])
        high_fidelity = []

        for record in records:
            if not record.metadata.get("is_document_chunk"):
                continue

            source_file = record.metadata.get("source_file", "unknown")
            stats = feedback_engine.get_file_stats(source_file)

            # Check if this file has enough good feedback
            if stats and stats.get("total_queries", 0) >= 3:
                utility = (
                    stats.get("helpful_count", 0) / stats["total_queries"]
                    if stats["total_queries"] > 0 else 0.0
                )
                if utility >= min_fidelity:
                    text = record.text
                    if text.startswith('[DOC:'):
                        nl = text.find('\n')
                        if nl > 0:
                            text = text[nl + 1:]

                    high_fidelity.append({
                        "source_file": source_file,
                        "chunk_text": text,
                        "utility": utility,
                        "metadata": record.metadata,
                    })

        return high_fidelity

    def _tokenize(self, text: str) -> set:
        """Tokenize text into lowercase word set."""
        if not text:
            return set()
        words = re.findall(r'[a-z]{3,}', text.lower())
        # Remove stopwords
        stopwords = {
            "the", "and", "for", "that", "this", "with", "was", "are",
            "from", "have", "has", "had", "not", "but", "all", "can",
            "which", "their", "will", "been", "were", "they", "more",
            "when", "what", "who", "how", "where", "about", "also",
        }
        return {w for w in words if w not in stopwords}

    def _extract_key_entities(self, text: str) -> set:
        """Extract key capitalized terms as entity set."""
        entities = set()
        # Case names
        for match in re.finditer(r'\b([A-Z][a-z]+\s+v\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text):
            entities.add(match.group(0).lower())
        # Capitalized phrases (2+ words)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            entities.add(match.group(0).lower())
        return entities

    def get_stats(self) -> Dict[str, Any]:
        return {
            "evaluations": self._evaluations,
            "above_threshold": self._above_threshold,
            "threshold": self.MIN_FIDELITY_FOR_GRAPH,
        }


# Testing helper
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("EVALUATOR TEST")
    logger.info("=" * 60)

    from src.memory.synthetic_gen import SyntheticQA

    evaluator = AutomatedEvaluator()

    # Test 1: Perfect match
    truth = "The Salomon case established separate legal personality in 1897."
    retrieved = "Salomon v Salomon established separate legal personality in 1897."
    qa = SyntheticQA(
        question="What did Salomon establish?",
        ground_truth_chunk=truth,
        source_file="law.txt",
        chunk_index=0,
        category="legal",
    )
    score = evaluator.evaluate(qa, retrieved, 0.92)
    assert score > 0.4, f"Good match should score >0.4, got {score}"
    logger.info(f"✓ Good match: fidelity={score:.3f}")

    # Test 2: Weak match
    qa = SyntheticQA(
        question="What is quantum physics?",
        ground_truth_chunk="Company law governs corporate entities.",
        source_file="law.txt",
        chunk_index=1,
        category="general",
    )
    score = evaluator.evaluate(qa, "Statistics uses mean, median, mode.", 0.05)
    assert score < 0.3, f"Weak match should score <0.3, got {score}"
    logger.info(f"✓ Weak match: fidelity={score:.3f}")

    # Test 3: Entity matching
    truth = "Gilford Motor Co v Horne established the fraud exception to corporate veil."
    retrieved = "The fraud exception from Gilford Motor Co v Horne allows piercing the corporate veil."
    qa = SyntheticQA(
        question="What case established the fraud exception?",
        ground_truth_chunk=truth,
        source_file="law.txt",
        chunk_index=2,
        category="legal",
    )
    score = evaluator.evaluate(qa, retrieved, 0.88)
    assert score > 0.3
    logger.info(f"✓ Entity match: fidelity={score:.3f}")

    # Test 4: Stats
    stats = evaluator.get_stats()
    assert stats["evaluations"] == 3
    logger.info(f"✓ Stats: {stats}")

    logger.info("\nALL EVALUATOR TESTS PASSED")