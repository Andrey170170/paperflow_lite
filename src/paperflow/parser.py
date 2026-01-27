"""PDF parsing with docling."""

import hashlib
import json
import re
import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter

from paperflow.config import ParserConfig
from paperflow.models import ParsedPaper


class PDFParseError(Exception):
    """Error raised when PDF parsing fails."""

    pass


class PDFParser:
    """Parser for extracting text from PDF files using docling."""

    def __init__(self, config: ParserConfig) -> None:
        """Initialize the parser.

        Args:
            config: Parser configuration.
        """
        self.config = config
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_bytes: bytes, cache_key: str | None) -> ParsedPaper:
        """Parse a PDF file and extract text content.

        Args:
            pdf_bytes: Raw PDF file bytes.
            cache_key: Optional key for caching. If provided, cached results
                will be returned if available.

        Returns:
            ParsedPaper with extracted content.

        Raises:
            PDFParseError: If parsing fails.
        """
        # Check cache first
        if cache_key is not None:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        # Parse with docling
        try:
            result = self._parse_with_docling(pdf_bytes)
        except Exception as e:
            raise PDFParseError(f"Failed to parse PDF: {e}") from e

        # Cache the result
        if cache_key is not None:
            self._save_cache(cache_key, result)

        return result

    def _parse_with_docling(self, pdf_bytes: bytes) -> ParsedPaper:
        """Use docling to convert PDF to text.

        Args:
            pdf_bytes: Raw PDF content.

        Returns:
            ParsedPaper with extracted content.
        """
        # Write bytes to temp file for docling
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        try:
            converter = DocumentConverter()
            result = converter.convert(tmp_path)

            # Export to markdown format
            markdown_text = result.document.export_to_markdown()

            # Get page count from input document
            page_count = result.input.page_count

            # Check if truncated (page count exceeds max)
            truncated = page_count > self.config.max_pages

            # Extract title and abstract from markdown
            title = self._extract_title(markdown_text)
            abstract = self._extract_abstract(markdown_text)

            return ParsedPaper(
                title=title,
                abstract=abstract,
                full_text=markdown_text,
                page_count=page_count,
                truncated=truncated,
            )
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    def _extract_title(self, markdown: str) -> str | None:
        """Extract paper title from markdown.

        Looks for first level-1 heading.

        Args:
            markdown: Markdown content.

        Returns:
            Title string or None if not found.
        """
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_abstract(self, markdown: str) -> str | None:
        """Extract abstract from markdown.

        Looks for content under "Abstract" heading.

        Args:
            markdown: Markdown content.

        Returns:
            Abstract text or None if not found.
        """
        # Look for ## Abstract or # Abstract followed by content
        pattern = r"#+\s*Abstract\s*\n\n(.+?)(?=\n#|\Z)"
        match = re.search(pattern, markdown, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _get_cached(self, cache_key: str) -> ParsedPaper | None:
        """Retrieve cached parsing result.

        Args:
            cache_key: Cache key identifier.

        Returns:
            Cached ParsedPaper or None if not found.
        """
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            return ParsedPaper.model_validate(data)
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, cache_key: str, paper: ParsedPaper) -> None:
        """Save parsing result to cache.

        Args:
            cache_key: Cache key identifier.
            paper: Parsed paper to cache.
        """
        cache_path = self._cache_path(cache_key)
        with open(cache_path, "w") as f:
            json.dump(paper.model_dump(), f)

    def _cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a key.

        Args:
            cache_key: Cache key identifier.

        Returns:
            Path to the cache file.
        """
        # Hash the key for safe filename
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        return Path(self.config.cache_dir) / f"{safe_key}.json"
