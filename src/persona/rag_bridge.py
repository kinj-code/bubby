"""RAG Context Bridge: Injects retrieved document chunks into LLM prompts.

Queries the FAISS vector store for document chunks relevant to the user's
question, formats them as context, and injects into the system prompt
before LLM generation.

Triggers automatically when the question contains academic/technical
keywords, or when the user explicitly says "search my notes".

RAM: Negligible (query + format existing embeddings).
"""

import logging
import re
from typing import Optional, List, Dict, Any

from src.memory.vector_db import VectorStore, MemoryRecord
from src.memory.embedding import EmbeddingEngine, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Keywords that trigger automatic RAG retrieval
RAG_TRIGGER_PATTERNS = [
    r"what (is|are|was|were) the .*(rule|principle|exception|case|doctrine|theory)",
    r"explain the .*(concept|principle|rule|law|theory|case of)",
    r"according to (the notes|my notes|the document|the textbook|the reading)",
    r"search (my )?notes",
    r"what does (the |my )?(notes|document|textbook) say",
    r"summarize (the |my )?notes",
    r"find (me )?information (about|on|regarding)",
    r"(company law|contract law|criminal law|constitutional law|tort law)",
    r"(supreme court|court of appeal|high court|magistrate)",
    r"(section|article|clause) \d+",
    r"what (are|is) the (exceptions|requirements|elements|conditions|grounds)",
    r"define .*in (legal|business|accounting|economic|statistical) terms",
]

# Maximum context tokens to inject (to stay within LLM context window)
MAX_CONTEXT_TOKENS = 800


class RAGBridge:
    """
    Bridges document retrieval with LLM synthesis.

    Flow:
    1. User asks a question (e.g., "What are exceptions to Salomon?")
    2. should_query_documents() checks if question matches trigger patterns
    3. query() searches FAISS for top-k relevant chunks
    4. format_context() builds injection string for system prompt
    5. UnifiedPersonaPrompt includes the context in synthesis
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        top_k: int = 3,
        min_similarity: float = 0.2,
    ) -> None:
        """
        Initialize RAG bridge.

        Args:
            embedding_engine: For encoding queries
            vector_store: For searching document chunks
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
        """
        self._embedding = embedding_engine
        self._vector_store = vector_store
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._queries_performed = 0
        self._chunks_retrieved = 0
        
        logger.info(
            f"RAGBridge initialized (top_k={top_k}, min_sim={min_similarity})"
        )

    def should_query_documents(self, text: str) -> bool:
        """
        Determine if the user's text should trigger document retrieval.

        Uses regex patterns that match:
        - Academic/legal keywords
        - Explicit "search my notes" commands
        - Questions asking about concepts, rules, principles
        - Case law references

        Args:
            text: User's question or observation context

        Returns:
            True if document search is warranted
        """
        if not text or len(text) < 10:
            return False

        text_lower = text.lower()

        for pattern in RAG_TRIGGER_PATTERNS:
            if re.search(pattern, text_lower):
                logger.debug(f"RAG trigger matched: {pattern[:50]}...")
                return True

        return False

    def query(self, text: str, top_k: Optional[int] = None) -> List[MemoryRecord]:
        """
        Query the vector store for relevant document chunks.

        Args:
            text: Query text (user question)
            top_k: Override number of chunks to retrieve

        Returns:
            List of MemoryRecords sorted by relevance
        """
        if not text:
            return []

        k = top_k or self._top_k
        self._queries_performed += 1

        try:
            # Encode query
            query_embedding = self._embedding.encode(text)
            if query_embedding is None or len(query_embedding) != EMBEDDING_DIM:
                logger.warning("Failed to encode RAG query")
                return []

            # Search vector store
            results = self._vector_store.search(query_embedding, k=k)

            # Filter by similarity threshold
            filtered = []
            for record, score in results:
                # Include similarity in metadata for debugging
                record.metadata["similarity_score"] = round(float(score), 3)
                if score >= self._min_similarity:
                    filtered.append(record)

            self._chunks_retrieved += len(filtered)
            logger.debug(
                f"RAG query: {len(filtered)}/{len(results)} chunks above "
                f"threshold ({self._min_similarity})"
            )

            return filtered

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return []

    def format_context(self, records: List[MemoryRecord]) -> str:
        """
        Format retrieved chunks into an LLM context block.

        Only includes document chunks (filters out conversation memories).

        Args:
            records: Retrieved MemoryRecords from vector store

        Returns:
            Formatted context string for system prompt injection
        """
        if not records:
            return ""

        doc_chunks = [
            r for r in records
            if r.metadata.get("is_document_chunk")
        ]

        if not doc_chunks:
            return ""

        # Build context block
        lines = ["## REFERENCE DOCUMENTS", ""]
        seen_files = set()

        for i, chunk in enumerate(doc_chunks[:self._top_k]):
            source = chunk.metadata.get("source_file", "unknown")
            score = chunk.metadata.get("similarity_score", 0.0)

            if source not in seen_files:
                seen_files.add(source)

            # Extract the core text (remove the [DOC: ...] prefix if present)
            text = chunk.text
            if text.startswith("[DOC:"):
                # Text already has formatting — use as-is
                lines.append(text)
            else:
                lines.append(f"[DOC: {source}]\n{text}")

            if i < len(doc_chunks[:self._top_k]) - 1:
                lines.append("")

        context = "\n".join(lines)

        # Truncate if too long (rough check)
        if len(context) > MAX_CONTEXT_TOKENS * 4:
            context = context[: MAX_CONTEXT_TOKENS * 4]
            context += "\n[... context truncated to fit context window ...]"

        return context

    def get_augmented_prompt(
        self,
        user_text: str,
        base_prompt: str,
    ) -> str:
        """
        Full pipeline: query + format + inject into prompt.

        Args:
            user_text: User's question
            base_prompt: Original user prompt for LLM

        Returns:
            Augmented prompt with document context injected
        """
        if not self.should_query_documents(user_text):
            return base_prompt

        records = self.query(user_text)
        if not records:
            return base_prompt

        context = self.format_context(records)
        if not context:
            return base_prompt

        # Inject context into prompt
        augmented = (
            f"{context}\n\n"
            f"---\n\n"
            f"Use the reference documents above to answer accurately. "
            f"If the documents don't contain relevant information, "
            f"say so honestly.\n\n"
            f"{base_prompt}"
        )

        logger.info(
            f"RAG augmented prompt: {len(records)} chunks, "
            f"context={len(context)} chars"
        )

        return augmented

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "queries_performed": self._queries_performed,
            "chunks_retrieved": self._chunks_retrieved,
            "top_k": self._top_k,
            "min_similarity": self._min_similarity,
            "vector_store_size": len(self._vector_store._records),
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
    logger.info("RAG BRIDGE TEST")
    logger.info("=" * 60)

    # Set up the stack
    from src.memory.parser import DocumentParser
    from src.memory.ingestion import DocumentIngestion

    embedding = EmbeddingEngine()
    vector_store = VectorStore(embedding_dim=384)
    ingestion = DocumentIngestion(embedding, vector_store)
    bridge = RAGBridge(embedding, vector_store, top_k=3)

    # Create and ingest a test document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(
            "COMPANY LAW 101 - Principles of Separate Legal Personality\n\n"
            "The doctrine of separate legal personality was established in "
            "the landmark case of Salomon v Salomon & Co Ltd [1897] AC 22. "
            "The House of Lords held that a duly incorporated company is a "
            "separate legal entity distinct from its members.\n\n"
            "EXCEPTIONS TO THE RULE:\n"
            "1. Fraud: Courts may pierce the corporate veil where the company "
            "is used as a device for fraud (Gilford Motor Co v Horne [1933]).\n"
            "2. Agency: Where the company acts as an agent for its members.\n"
            "3. Statutory Exceptions: Under the Companies Act, directors may "
            "be personally liable for certain breaches.\n"
            "4. Group Enterprises: In some cases, courts treat parent and "
            "subsidiary companies as a single economic entity.\n\n"
            "STATISTICS NOTES - Measures of Central Tendency\n"
            "Mean: The arithmetic average of a dataset.\n"
            "Median: The middle value when data is ordered.\n"
            "Mode: The most frequently occurring value.\n"
            + "Additional filler content. " * 20
        )
        txt_path = f.name

    parser = DocumentParser()
    doc = parser.parse_file(txt_path)
    assert doc is not None
    chunks = ingestion.ingest_document(doc)
    logger.info(f"Ingested: {chunks} chunks")

    # Test 1: Trigger detection
    assert bridge.should_query_documents("What are the exceptions to the rule in Salomon?")
    assert bridge.should_query_documents("Explain the principle of separate legal personality")
    assert bridge.should_query_documents("Search my notes for statistics")
    assert bridge.should_query_documents("What does the textbook say about fraud?")
    assert not bridge.should_query_documents("What time is it?")
    assert not bridge.should_query_documents("Hello there")
    logger.info("✓ Trigger detection: 4 positive, 2 negative — correct")

    # Test 2: Query retrieval
    records = bridge.query("What are the exceptions in Salomon?")
    assert len(records) > 0, "Should retrieve chunks"
    # Check that retrieved chunks contain document content
    has_law_content = False
    for r in records:
        if "Salomon" in r.text or "EXCEPTIONS" in r.text or "Fraud" in r.text:
            has_law_content = True
        assert "similarity_score" in r.metadata
    assert has_law_content, "Should find law content"
    logger.info(f"✓ Query retrieval: {len(records)} chunks with scores")

    # Test 3: Context formatting
    context = bridge.format_context(records)
    assert "REFERENCE DOCUMENTS" in context
    assert "DOC:" in context
    assert len(context) > 100
    logger.info(f"✓ Context formatting: {len(context)} chars")

    # Test 4: Augmented prompt
    base_prompt = "What are the exceptions to Salomon principle?"
    augmented = bridge.get_augmented_prompt(
        "What are the exceptions to the rule in Salomon v Salomon?",
        base_prompt,
    )
    assert "REFERENCE DOCUMENTS" in augmented
    assert base_prompt in augmented
    assert len(augmented) > len(base_prompt)
    logger.info(f"✓ Augmented prompt: {len(augmented)} chars (was {len(base_prompt)})")

    # Test 5: Non-trigger — no augmentation
    base = "What time is it?"
    result = bridge.get_augmented_prompt("What time is it?", base)
    assert result == base  # No augmentation
    logger.info("✓ Non-trigger: prompt unchanged")

    # Test 6: Stats
    stats = bridge.get_stats()
    assert stats["queries_performed"] >= 2
    logger.info(f"✓ Stats: {stats}")

    os.unlink(txt_path)
    logger.info("\nALL RAG BRIDGE TESTS PASSED")