"""Document ingestion pipeline: chunk, embed, store in FAISS vector index.

Takes parsed documents, splits into overlapping chunks, generates
embeddings via the existing EmbeddingEngine, and stores in the
VectorStore for RAG retrieval.

RAM: ~50MB for embedding model + chunk buffers.
"""

import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.memory.parser import ParsedDocument
from src.memory.embedding import EmbeddingEngine, EMBEDDING_DIM
from src.memory.vector_db import VectorStore, MemoryRecord

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk of document text with metadata."""
    text: str
    source_file: str
    source_path: str
    chunk_index: int
    num_chunks: int
    token_estimate: int = 0
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "num_chunks": self.num_chunks,
            "is_document_chunk": True,
        }

    def format_for_context(self) -> str:
        """Format chunk for LLM context window."""
        return f"[DOC: {self.source_file} (chunk {self.chunk_index + 1}/{self.num_chunks})]\n{self.text}"


class DocumentIngestion:
    """
    Ingests parsed documents into the RAG vector index.

    Pipeline:
    1. Receive ParsedDocument
    2. Chunk text into ~500-token segments with 50-token overlap
    3. Embed each chunk via EmbeddingEngine
    4. Store in VectorStore with source metadata
    """

    CHUNK_SIZE_TOKENS = 500      # Target tokens per chunk
    CHUNK_OVERLAP_TOKENS = 50    # Overlap between chunks
    CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate for token counting
    MIN_CHUNK_CHARS = 50          # Minimum chars for a chunk

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
    ) -> None:
        """
        Initialize ingestion pipeline.

        Args:
            embedding_engine: Existing embedding engine
            vector_store: Existing vector store
        """
        self._embedding = embedding_engine
        self._vector_store = vector_store
        self._docs_ingested = 0
        self._chunks_ingested = 0
        self._total_chars_ingested = 0
        
        logger.info(
            f"DocumentIngestion initialized "
            f"(chunk_size={self.CHUNK_SIZE_TOKENS} tokens, "
            f"overlap={self.CHUNK_OVERLAP_TOKENS} tokens)"
        )

    def ingest_document(self, doc: ParsedDocument) -> int:
        """
        Ingest a single parsed document into the vector store.

        Args:
            doc: ParsedDocument from DocumentParser

        Returns:
            Number of chunks ingested
        """
        # Chunk the document
        chunks = self._chunk_text(doc.content, doc.filename, doc.filepath)
        
        if not chunks:
            logger.warning(f"No chunks generated for {doc.filename}")
            return 0

        # Embed and store each chunk
        stored = 0
        for chunk in chunks:
            try:
                embedding = self._embedding.encode(chunk.text)
                if embedding is not None and len(embedding) == EMBEDDING_DIM:
                    # Add to vector store with document metadata
                    self._vector_store.add(
                        text=chunk.format_for_context(),
                        embedding=embedding,
                        importance=0.7,
                        metadata=chunk.metadata,
                    )
                    stored += 1
                    
            except Exception as e:
                logger.error(f"Failed to embed chunk {chunk.chunk_index}: {e}")

        self._docs_ingested += 1
        self._chunks_ingested += stored
        self._total_chars_ingested += doc.num_chars
        
        logger.info(
            f"Ingested '{doc.filename}': {stored} chunks "
            f"({doc.num_chars} chars, {doc.num_pages} pages)"
        )
        
        return stored

    def ingest_directory(self, dirpath: str, recursive: bool = True) -> Dict[str, int]:
        """
        Parse and ingest all documents in a directory.

        Args:
            dirpath: Path to directory
            recursive: If True, search subdirectories

        Returns:
            Dict mapping filename → number of chunks ingested
        """
        from src.memory.parser import DocumentParser
        
        parser = DocumentParser()
        docs = parser.parse_directory(dirpath, recursive=recursive)
        
        results = {}
        for doc in docs:
            chunks = self.ingest_document(doc)
            results[doc.filename] = chunks
        
        total_chunks = sum(results.values())
        logger.info(
            f"Directory ingestion complete: {len(docs)} docs, "
            f"{total_chunks} chunks total"
        )
        
        return results

    def _chunk_text(
        self,
        text: str,
        source_file: str,
        source_path: str,
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks using sentence-aware boundaries.

        Strategy: Slide a window of ~CHUNK_SIZE_TOKENS chars through the text
        with ~CHUNK_OVERLAP_TOKENS chars overlap. Tries to break at sentence
        boundaries (., !, ?, newline) to keep chunks coherent.

        Args:
            text: Full document text
            source_file: Source filename
            source_path: Source file path

        Returns:
            List of TextChunks
        """
        if not text or len(text) < self.MIN_CHUNK_CHARS:
            return []

        # Calculate char boundaries from token estimates
        chunk_chars = self.CHUNK_SIZE_TOKENS * self.CHARS_PER_TOKEN_ESTIMATE  # ~2000 chars
        overlap_chars = self.CHUNK_OVERLAP_TOKENS * self.CHARS_PER_TOKEN_ESTIMATE  # ~200 chars

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))

            # Try to find a sentence boundary to break at
            if end < len(text):
                # Search for sentence-ending punctuation within 200 chars back
                search_start = max(start, end - 200)
                boundary = self._find_sentence_boundary(text, search_start, end)
                if boundary > 0:
                    end = boundary

            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.MIN_CHUNK_CHARS:
                token_est = len(chunk_text) // self.CHARS_PER_TOKEN_ESTIMATE
                chunks.append(TextChunk(
                    text=chunk_text,
                    source_file=source_file,
                    source_path=source_path,
                    chunk_index=chunk_idx,
                    num_chunks=0,  # Will be set after
                    token_estimate=token_est,
                ))
                chunk_idx += 1

            # Move window forward (with overlap)
            start = start + chunk_chars - overlap_chars
            
            # If we're near the end, just capture the rest
            if start >= len(text):
                break
            if len(text) - start < self.MIN_CHUNK_CHARS:
                # Capture remaining bit as final chunk
                remaining = text[start:].strip()
                if len(remaining) >= self.MIN_CHUNK_CHARS:
                    chunks.append(TextChunk(
                        text=remaining,
                        source_file=source_file,
                        source_path=source_path,
                        chunk_index=chunk_idx,
                        num_chunks=0,
                        token_estimate=len(remaining) // self.CHARS_PER_TOKEN_ESTIMATE,
                    ))
                break

        # Set num_chunks on all
        for c in chunks:
            c.num_chunks = len(chunks)

        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find a natural sentence boundary between start and end.
        
        Returns the position after the boundary, or 0 if none found.
        """
        # Search backward from end for sentence end
        search_text = text[start:end]
        
        # Look for: period + space + capital letter (strong sentence boundary)
        matches = list(re.finditer(r'[.!?]\s+[A-Z]', search_text))
        if matches:
            last_match = matches[-1]
            return start + last_match.end() - 1  # Include the punctuation

        # Look for: double newline (paragraph boundary)
        match = re.search(r'\n\s*\n', search_text[::-1])  # Search from end
        if match:
            target = len(search_text) - match.start()
            return start + target

        # Look for: single newline
        idx = search_text.rfind('\n')
        if idx > len(search_text) // 2:  # Only if it's in the latter half
            return start + idx + 1

        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            "docs_ingested": self._docs_ingested,
            "chunks_ingested": self._chunks_ingested,
            "total_chars_ingested": self._total_chars_ingested,
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
    logger.info("INGESTION PIPELINE TEST")
    logger.info("=" * 60)

    # Set up the full stack
    embedding = EmbeddingEngine()
    vector_store = VectorStore(embedding_dim=EMBEDDING_DIM)
    ingestion = DocumentIngestion(embedding, vector_store)

    # Create test documents
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # ~2000 words so we get multiple chunks
        f.write(
            "Introduction to Company Law.\n\n"
            "The principle of separate legal personality was established in "
            "Salomon v Salomon & Co Ltd [1897] AC 22. This landmark case "
            "established that a company is a separate legal entity from its "
            "shareholders. Mr. Salomon incorporated his boot manufacturing "
            "business as a limited company. When the company failed, the "
            "creditors argued that Mr. Salomon should be personally liable. "
            "The House of Lords held that the company was a separate legal "
            "person and Mr. Salomon was not personally liable for its debts.\n\n"
            "This principle has several exceptions including: fraud, agency, "
            "and statutory exceptions under the Companies Act. The courts "
            "may pierce the corporate veil in cases of fraud or where "
            "the company is a mere facade.\n\n"
            + "Additional content for chunking test. " * 30
        )
        txt_path = f.name

    # Parse and ingest
    from src.memory.parser import DocumentParser
    parser = DocumentParser()
    doc = parser.parse_file(txt_path)
    assert doc is not None
    logger.info(f"Document: {doc.num_chars} chars")

    chunks = ingestion.ingest_document(doc)
    assert chunks > 0, f"Expected chunks > 0, got {chunks}"
    logger.info(f"✓ Ingested: {chunks} chunks from {doc.filename}")

    # Verify chunks are in vector store
    assert len(vector_store._records) >= chunks
    logger.info(f"✓ Vector store: {len(vector_store._records)} records")

    # Verify metadata
    for record in vector_store._records:
        assert record.metadata.get("is_document_chunk"), "Missing is_document_chunk flag"
        assert "source_file" in record.metadata
        break
    logger.info(f"✓ Chunk metadata tagged correctly")

    # Test chunk formatting
    from src.memory.ingestion import TextChunk
    chunk = TextChunk(
        text="Test chunk content",
        source_file="test.txt",
        source_path="/tmp/test.txt",
        chunk_index=0,
        num_chunks=1,
    )
    formatted = chunk.format_for_context()
    assert "[DOC:" in formatted
    assert "test.txt" in formatted
    assert "Test chunk content" in formatted
    logger.info(f"✓ Chunk formatting: {formatted[:60]}...")

    # Stats
    stats = ingestion.get_stats()
    assert stats["docs_ingested"] == 1
    logger.info(f"✓ Stats: {stats}")

    # Cleanup
    os.unlink(txt_path)

    logger.info("\nALL INGESTION TESTS PASSED")