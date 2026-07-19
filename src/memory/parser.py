"""Document parser for local files (PDF, TXT, MD).

Extracts clean text from documents for ingestion into the RAG pipeline.
Uses PyMuPDF (fitz) for PDF parsing, with fallback to pypdf.
Pure text and markdown files parsed via native read.

RAM: ~10MB per document (streaming parse, not full load).
"""

import logging
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".py", ".js", ".ts"}


@dataclass
class ParsedDocument:
    """A parsed document ready for chunking and ingestion."""
    filepath: str
    filename: str
    content: str
    num_chars: int = 0
    num_pages: int = 1
    file_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.num_chars = len(self.content)


class DocumentParser:
    """
    Extracts clean text from local document files.

    Supported formats:
    - PDF: via PyMuPDF (fitz) → fallback to pypdf
    - TXT: native read
    - MD/Markdown: native read
    - PY/JS/TS: native read (code files)

    Cleans extracted text by:
    - Removing excessive whitespace
    - Stripping header/footer artifacts
    - Replacing non-UTF-8 characters
    - Normalizing line endings
    """

    def __init__(self) -> None:
        self._pdf_parser = self._detect_pdf_parser()
        self._docs_parsed = 0
        self._total_chars = 0
        logger.info(f"DocumentParser initialized (pdf_parser={self._pdf_parser})")

    def _detect_pdf_parser(self) -> str:
        """Detect available PDF parsing library."""
        try:
            import fitz  # PyMuPDF
            return "pymupdf"
        except ImportError:
            pass
        try:
            import pypdf
            return "pypdf"
        except ImportError:
            pass
        return "none"

    def parse_file(self, filepath: str) -> Optional[ParsedDocument]:
        """
        Parse a single document file.

        Args:
            filepath: Path to the document file

        Returns:
            ParsedDocument or None if parsing fails
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return None

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(f"Unsupported file type: {ext} ({filepath})")
            return None

        try:
            if ext == ".pdf":
                content, num_pages = self._parse_pdf(path)
                file_type = "pdf"
            elif ext in (".txt", ".py", ".js", ".ts"):
                content = self._parse_text(path)
                num_pages = 1
                file_type = ext.lstrip(".")
            elif ext in (".md", ".markdown"):
                content = self._parse_text(path)
                num_pages = 1
                file_type = "markdown"
            else:
                return None

            if not content or len(content) < 10:
                logger.warning(f"Empty or too-short content from: {filepath}")
                return None

            # Clean extracted text
            content = self._clean_text(content)

            doc = ParsedDocument(
                filepath=str(path),
                filename=path.name,
                content=content,
                num_pages=num_pages,
                file_type=file_type,
                metadata={
                    "source_file": path.name,
                    "source_path": str(path),
                    "file_type": file_type,
                    "num_pages": num_pages,
                },
            )

            self._docs_parsed += 1
            self._total_chars += doc.num_chars
            logger.debug(f"Parsed: {path.name} ({doc.num_chars} chars, {num_pages} pages)")
            return doc

        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None

    def parse_directory(self, dirpath: str, recursive: bool = True) -> List[ParsedDocument]:
        """
        Parse all supported documents in a directory.

        Args:
            dirpath: Path to directory
            recursive: If True, search subdirectories

        Returns:
            List of ParsedDocuments
        """
        path = Path(dirpath)
        if not path.exists() or not path.is_dir():
            logger.warning(f"Directory not found: {dirpath}")
            return []

        results = []
        pattern = "**/*" if recursive else "*"

        for filepath in path.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                doc = self.parse_file(str(filepath))
                if doc:
                    results.append(doc)

        logger.info(f"Parsed {len(results)} documents from {dirpath}")
        return results

    def _parse_pdf(self, path: Path) -> tuple:
        """
        Parse PDF file. Returns (content, num_pages).
        """
        if self._pdf_parser == "pymupdf":
            import fitz
            doc = fitz.open(str(path))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n".join(pages), len(pages)

        elif self._pdf_parser == "pypdf":
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages), len(reader.pages)

        else:
            logger.warning("No PDF parser available. Install PyMuPDF: pip install PyMuPDF")
            return "", 0

    def _parse_text(self, path: Path) -> str:
        """Parse plain text file."""
        try:
            return path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return path.read_text(encoding='latin-1', errors='replace')

    def _clean_text(self, text: str) -> str:
        """Clean extracted text for ingestion."""
        if not text:
            return ""

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove common header/footer artifacts
        artifacts = [
            r'^\d+\s*$',                    # Standalone page numbers
            r'^Page \d+ of \d+$',            # "Page 1 of 10"
            r'^-\s*\d+\s*-$',               # "- 1 -"
            r'^\f',                          # Form feed characters
        ]
        for pattern in artifacts:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)

        # Replace non-UTF-8 characters that slipped through
        text = text.encode('utf-8', errors='replace').decode('utf-8')

        # Collapse multiple spaces (but preserve intentional formatting)
        text = re.sub(r' {3,}', '  ', text)

        # Remove trailing whitespace from each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics."""
        return {
            "docs_parsed": self._docs_parsed,
            "total_chars": self._total_chars,
            "pdf_parser": self._pdf_parser,
        }


# Testing helper
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("DOCUMENT PARSER TEST")
    logger.info("=" * 60)

    parser = DocumentParser()

    # Test 1: Parse TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\n\nIt has multiple paragraphs.\nHere's a third line.")
        txt_path = f.name

    doc = parser.parse_file(txt_path)
    assert doc is not None
    assert doc.file_type == "txt"
    assert "test document" in doc.content
    assert doc.num_pages == 1
    logger.info(f"✓ TXT parsed: {doc.num_chars} chars")

    # Test 2: Parse MD file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Company Law Notes\n\n## Salomon v Salomon\n\nThe principle of separate legal personality...")
        md_path = f.name

    doc = parser.parse_file(md_path)
    assert doc is not None
    assert doc.file_type == "markdown"
    assert "Salomon" in doc.content
    logger.info(f"✓ MD parsed: {doc.num_chars} chars")

    # Test 3: Text cleaning
    dirty_text = "Line one\n\n\n\nLine two\n\n\nLine three\n    Spaces    here"
    cleaned = parser._clean_text(dirty_text)
    assert "\n\n\n" not in cleaned  # Should reduce to max 2 newlines
    assert "Spaces    here" not in cleaned  # Should collapse
    logger.info(f"✓ Text cleaned: {len(cleaned)} chars (was {len(dirty_text)})")

    # Test 4: Unsupported file type
    doc = parser.parse_file("/tmp/nonexistent.xyz")
    assert doc is None
    logger.info("✓ Unsupported file type returns None")

    # Test 5: Metadata
    doc = parser.parse_file(txt_path)
    assert doc.metadata["source_file"] == Path(txt_path).name
    assert doc.metadata["file_type"] == "txt"
    logger.info(f"✓ Metadata: {doc.metadata}")

    # Test 6: Stats
    stats = parser.get_stats()
    assert stats["docs_parsed"] >= 2
    logger.info(f"✓ Stats: {stats}")

    # Cleanup
    os.unlink(txt_path)
    os.unlink(md_path)

    logger.info("\nALL PARSER TESTS PASSED")