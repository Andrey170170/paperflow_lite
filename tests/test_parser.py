"""Tests for PDF parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paperflow.config import ParserConfig
from paperflow.models import ParsedPaper
from paperflow.parser import PDFParseError, PDFParser


class TestPDFParser:
    """Tests for PDFParser class."""

    @pytest.fixture
    def parser_config(self, tmp_path: Path) -> ParserConfig:
        """Create a parser config with temp cache directory."""
        return ParserConfig(max_pages=10, cache_dir=str(tmp_path / "cache"))

    @pytest.fixture
    def parser(self, parser_config: ParserConfig) -> PDFParser:
        """Create a PDFParser instance."""
        return PDFParser(parser_config)

    def test_parse_pdf_success(self, parser: PDFParser) -> None:
        """Test successful PDF parsing with mocked pymupdf."""
        # Create mock page
        mock_page = MagicMock()
        mock_page.get_text.return_value = (
            "Test Paper Title\n\n"
            "Abstract\n\n"
            "This is the abstract of the paper.\n\n"
            "1. Introduction\n\n"
            "This is the introduction section."
        )

        # Create mock document
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("paperflow.parser.pymupdf.open", return_value=mock_doc):
            result = parser.parse(b"fake pdf bytes", cache_key=None)

        assert isinstance(result, ParsedPaper)
        assert result.page_count == 5
        assert not result.truncated
        mock_doc.close.assert_called_once()

    def test_parse_pdf_with_cache(self, parser: PDFParser) -> None:
        """Test that cached results are returned without re-parsing."""
        cache_key = "test_paper_123"

        # Create a cached result
        cached_paper = ParsedPaper(
            title="Cached Paper",
            abstract="Cached abstract",
            full_text="Cached content",
            page_count=3,
            truncated=False,
        )
        parser._save_cache(cache_key, cached_paper)

        # Parse should return cached result without calling pymupdf
        with patch("paperflow.parser.pymupdf.open") as mock_open:
            result = parser.parse(b"pdf bytes", cache_key=cache_key)
            mock_open.assert_not_called()

        assert result.title == "Cached Paper"
        assert result.full_text == "Cached content"

    def test_parse_pdf_truncation(self, parser_config: ParserConfig) -> None:
        """Test that papers exceeding max_pages are truncated."""
        config = ParserConfig(max_pages=5, cache_dir=parser_config.cache_dir)
        parser = PDFParser(config)

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page content"

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=20)  # Exceeds max_pages
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("paperflow.parser.pymupdf.open", return_value=mock_doc):
            result = parser.parse(b"pdf bytes", cache_key=None)

        assert result.truncated
        assert result.page_count == 20

    def test_parse_pdf_extract_title(self, parser: PDFParser) -> None:
        """Test title extraction from text."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = (
            "Attention Is All You Need\n\n"
            "Abstract\n\n"
            "We propose a new architecture..."
        )

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("paperflow.parser.pymupdf.open", return_value=mock_doc):
            result = parser.parse(b"pdf bytes", cache_key=None)

        assert result.title == "Attention Is All You Need"

    def test_parse_pdf_extract_abstract(self, parser: PDFParser) -> None:
        """Test abstract extraction from text."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = (
            "Title Here\n\n"
            "Abstract\n\n"
            "This is a detailed abstract that summarizes the paper's contributions "
            "and main findings in a comprehensive manner.\n\n"
            "1. Introduction\n\n"
            "Intro content."
        )

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("paperflow.parser.pymupdf.open", return_value=mock_doc):
            result = parser.parse(b"pdf bytes", cache_key=None)

        assert result.abstract is not None
        assert "summarizes the paper" in result.abstract

    def test_parse_pdf_error(self, parser: PDFParser) -> None:
        """Test handling of parsing errors."""
        with patch("paperflow.parser.pymupdf.open") as mock_open:
            mock_open.side_effect = Exception("PDF is corrupted")

            with pytest.raises(PDFParseError, match="Failed to parse PDF"):
                parser.parse(b"corrupt pdf", cache_key=None)

    def test_cache_persistence(self, parser: PDFParser) -> None:
        """Test that cache persists to disk."""
        cache_key = "persistent_test"
        paper = ParsedPaper(
            title="Test",
            abstract=None,
            full_text="Content",
            page_count=1,
            truncated=False,
        )

        parser._save_cache(cache_key, paper)
        loaded = parser._get_cached(cache_key)

        assert loaded is not None
        assert loaded.title == "Test"

    def test_cache_miss(self, parser: PDFParser) -> None:
        """Test cache miss returns None."""
        result = parser._get_cached("nonexistent_key")
        assert result is None

    def test_no_cache_when_key_none(self, parser: PDFParser) -> None:
        """Test that None cache_key skips caching."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content"

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("paperflow.parser.pymupdf.open", return_value=mock_doc):
            parser.parse(b"pdf bytes", cache_key=None)

        # Cache directory should be empty
        cache_files = list(Path(parser.config.cache_dir).glob("*.json"))
        assert len(cache_files) == 0
