"""
Synthetic Usage Generator — bootstraps feedback data without waiting for human interaction.

Iterates over indexed document chunks and generates synthetic QA pairs
by extracting key sentences as ground-truth answers. Feeds these through
the RAG pipeline to populate the feedback database with controlled,
high-quality evaluation data.

RAM: Streaming iteration, ~5MB overhead.
"""

import logging
import random
import re
from typing import Optional, Dict, Any, List, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SyntheticQA:
    """A synthetic question-answer pair generated from a document chunk."""
    question: str
    ground_truth_chunk: str
    source_file: str
    chunk_index: int
    category: str = "general"  # legal, technical, definition, general


class SyntheticGenerator:
    """
    Generates synthetic query-answer pairs from indexed documents.

    Strategy: For each chunk, extract key sentences as answers and
    synthesize plausible questions. No LLM needed — uses template-based
    generation with variable patterns for diversity.
    """

    QA_PATTERNS = [
        # Definition patterns
        ("What is {entity}?", "definition"),
        ("Define {entity}.", "definition"),
        ("What does {entity} mean?", "definition"),
        ("Explain the concept of {entity}.", "definition"),
        # Case/principle patterns
        ("What is the principle established in {entity}?", "legal"),
        ("What are the exceptions to {entity}?", "legal"),
        ("According to {entity}, what is the rule?", "legal"),
        ("How did {entity} affect the law?", "legal"),
        # Technical/explanatory
        ("How does {entity} work?", "technical"),
        ("What are the key points of {entity}?", "general"),
        ("Summarize {entity}.", "general"),
        ("What are the types of {entity}?", "technical"),
    ]

    def __init__(
        self,
        queries_per_document: int = 5,
        min_chunk_length: int = 100,
        random_seed: int = 42,
    ) -> None:
        self._queries_per_doc = queries_per_document
        self._min_chunk_length = min_chunk_length
        self._rng = random.Random(random_seed)
        self._total_generated = 0
        
        logger.info(
            f"SyntheticGenerator initialized "
            f"(queries/doc={queries_per_document}, min_chunk={min_chunk_length})"
        )

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text for question generation.

        Uses regex patterns for:
        - Case names: "X v Y [year]"
        - Capitalized phrases (likely named concepts)
        - Statute references: "Companies Act 2006", "s.213"
        - Defined terms in quotes
        """
        entities = []

        # Case law: "X v Y" or "X v Y [year]"
        case_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+&?\s*(?:Co|Ltd|Corporation|LLC|PLC))?(?:\s+\[\d{4}\])?)'
        cases = re.findall(case_pattern, text)
        entities.extend(cases)

        # Statute references: "Act Name [Year]" or "s.XXX Act Name"
        statute_pattern = r'((?:[A-Z][a-z]+\s+){1,4}Act\s+\d{4})'
        statutes = re.findall(statute_pattern, text)
        entities.extend(statutes)

        # Capitalized multi-word phrases (3+ words, start with capital)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,})\b'
        caps = re.findall(cap_pattern, text)
        # Filter out common words
        skip_words = {"The", "This", "That", "These", "Those", "When", "Where", "Which", "Each"}
        caps = [c for c in caps if c.split()[0] not in skip_words]
        entities.extend(caps[:3])  # Limit to 3

        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted[:2])

        # Title-case phrases (2 words)
        title_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        titles = re.findall(title_pattern, text)
        titles = [t for t in titles if t.split()[0] not in skip_words]
        entities.extend(titles[:5])

        # Deduplicate and limit
        seen = set()
        unique = []
        for e in entities:
            e_clean = e.strip().rstrip(".,;:!?'\"")
            if len(e_clean) > 3 and e_clean.lower() not in seen:
                seen.add(e_clean.lower())
                unique.append(e_clean)

        return unique[:10]

    def generate_qa_pairs(
        self,
        text: str,
        source_file: str,
        chunk_index: int,
    ) -> List[SyntheticQA]:
        """
        Generate synthetic QA pairs from a document chunk.

        Args:
            text: The chunk text (ground truth)
            source_file: Source filename
            chunk_index: Chunk index

        Returns:
            List of SyntheticQA pairs
        """
        if len(text) < self._min_chunk_length:
            return []

        entities = self.extract_entities(text)
        if not entities:
            return []

        pairs = []
        used_entities = set()

        # Select diverse patterns
        patterns = self._rng.sample(
            self.QA_PATTERNS,
            min(len(self.QA_PATTERNS), self._queries_per_doc)
        )

        for i, (template, category) in enumerate(patterns):
            if i >= self._queries_per_doc:
                break

            # Pick an entity not yet used
            available = [e for e in entities if e not in used_entities]
            if not available:
                available = entities  # Reuse if needed
            entity = self._rng.choice(available)
            used_entities.add(entity)

            question = template.format(entity=entity)

            pair = SyntheticQA(
                question=question,
                ground_truth_chunk=text[:500],  # Keep context manageable
                source_file=source_file,
                chunk_index=chunk_index,
                category=category,
            )
            pairs.append(pair)

        self._total_generated += len(pairs)
        return pairs

    def run_on_index(
        self,
        vector_store: Any,
        max_documents: int = 20,
    ) -> Iterator[Tuple[SyntheticQA, str, int]]:
        """
        Generate QA pairs from all chunks in the vector store.

        Yields:
            (SyntheticQA, source_file, chunk_index) tuples
        """
        records = getattr(vector_store, '_records', [])
        if not records:
            logger.warning("No records in vector store")
            return

        # Filter for document chunks
        doc_records = [
            r for r in records
            if hasattr(r, 'metadata') and r.metadata.get('is_document_chunk')
        ]

        if not doc_records:
            logger.warning("No document chunks found in vector store")
            return

        # Limit to prevent overload
        if len(doc_records) > max_documents:
            doc_records = self._rng.sample(doc_records, max_documents)

        logger.info(f"Generating synthetic QA from {len(doc_records)} chunks...")

        for record in doc_records:
            source_file = record.metadata.get('source_file', 'unknown')
            chunk_index = record.metadata.get('chunk_index', 0)
            text = record.text

            # Strip the [DOC: ...] prefix if present
            if text.startswith('[DOC:'):
                nl = text.find('\n')
                if nl > 0:
                    text = text[nl + 1:]

            pairs = self.generate_qa_pairs(text, source_file, chunk_index)
            for pair in pairs:
                yield (pair, source_file, chunk_index)

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "total_pairs_generated": self._total_generated,
            "queries_per_document": self._queries_per_doc,
        }


# Testing helper
if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("SYNTHETIC GENERATOR TEST")
    logger.info("=" * 60)

    generator = SyntheticGenerator(queries_per_document=3)

    # Test entity extraction
    text = (
        "The principle of Separate Legal Personality was established in "
        "Salomon v Salomon & Co Ltd [1897] AC 22. The House of Lords held "
        "that a company is a separate legal entity. The Companies Act 2006 "
        "provides statutory exceptions including s.213 and s.214. Directors "
        "must comply with fiduciary duties. The concept of 'piercing the "
        "corporate veil' emerged from Gilford Motor Co v Horne [1933]."
    )

    entities = generator.extract_entities(text)
    assert len(entities) > 0, "Should extract entities"
    assert "Salomon v Salomon & Co Ltd [1897]" in entities or any("Salomon" in e for e in entities)
    logger.info(f"✓ Entities extracted: {len(entities)} → {entities[:5]}")

    # Test QA generation
    pairs = generator.generate_qa_pairs(text, "company_law.txt", 0)
    assert len(pairs) > 0, "Should generate QA pairs"
    for p in pairs:
        assert p.question, "Question should not be empty"
        assert p.ground_truth_chunk, "Ground truth should not be empty"
    logger.info(f"✓ QA pairs generated: {len(pairs)}")
    for p in pairs[:3]:
        logger.info(f"  Q: {p.question}, category={p.category}")

    # Test stats
    stats = generator.get_stats()
    logger.info(f"✓ Stats: {stats}")

    logger.info("\nALL SYNTHETIC GENERATOR TESTS PASSED")