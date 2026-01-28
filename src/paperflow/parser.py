"""PDF parsing with pymupdf."""

import hashlib
import json
import re
from pathlib import Path

import pymupdf

from paperflow.config import ParserConfig
from paperflow.models import ParsedPaper


class PDFParseError(Exception):
    """Error raised when PDF parsing fails."""

    pass


class PDFParser:
    """Parser for extracting text from PDF files using pymupdf."""

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

        # Parse with pymupdf
        try:
            result = self._parse_pdf(pdf_bytes)
        except Exception as e:
            raise PDFParseError(f"Failed to parse PDF: {e}") from e

        # Cache the result
        if cache_key is not None:
            self._save_cache(cache_key, result)

        return result

    def _parse_pdf(self, pdf_bytes: bytes) -> ParsedPaper:
        """Use pymupdf to extract text from PDF.

        Args:
            pdf_bytes: Raw PDF content.

        Returns:
            ParsedPaper with extracted content.
        """
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        try:
            total_pages = len(doc)
            pages_to_parse = min(total_pages, self.config.max_pages)
            truncated = total_pages > self.config.max_pages

            # Extract text from each page
            text_parts = []
            for page_num in range(pages_to_parse):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts)

            # Try to extract title and abstract
            title = self._extract_title(full_text)
            abstract = self._extract_abstract(full_text)

            return ParsedPaper(
                title=title,
                abstract=abstract,
                full_text=full_text,
                page_count=total_pages,
                truncated=truncated,
            )
        finally:
            doc.close()

    def _extract_title(self, text: str) -> str | None:
        """Extract paper title from text.

        Assumes title is in the first few lines, typically the longest
        line or a line before "Abstract".

        Args:
            text: Full text content.

        Returns:
            Title string or None if not found.
        """
        lines = text.strip().split("\n")[:20]  # Check first 20 lines

        # Look for a substantial line before "Abstract"
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # If we hit abstract, the previous non-empty line might be title
            if re.match(r"^abstract\b", line, re.IGNORECASE):
                # Look back for title
                for j in range(i - 1, -1, -1):
                    prev_line = lines[j].strip()
                    if prev_line and len(prev_line) > 10:
                        return prev_line
                break

        # Fallback: find longest line in first 10 lines (likely title)
        candidates = [ln.strip() for ln in lines[:10] if ln.strip() and len(ln.strip()) > 15]
        if candidates:
            return max(candidates, key=len)

        return None

    def _extract_abstract(self, text: str) -> str | None:
        """Extract abstract from text.

        Looks for content after "Abstract" heading.

        Args:
            text: Full text content.

        Returns:
            Abstract text or None if not found.
        """
        # Pattern to match "Abstract" followed by content
        patterns = [
            r"(?:^|\n)\s*Abstract[:\s]*\n+(.*?)(?=\n\s*(?:1\.?\s*Introduction|Keywords|I\.\s|1\s+Introduction)|\Z)",
            r"(?:^|\n)\s*ABSTRACT[:\s]*\n+(.*?)(?=\n\s*(?:1\.?\s*Introduction|Keywords|I\.\s|1\s+INTRODUCTION)|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up: remove excessive whitespace
                abstract = re.sub(r"\s+", " ", abstract)
                # Limit length
                if len(abstract) > 100:  # Reasonable abstract length
                    return abstract[:2000] if len(abstract) > 2000 else abstract

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
