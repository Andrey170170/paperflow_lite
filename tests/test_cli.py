"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from paperflow.cli import app

runner = CliRunner()


@pytest.fixture
def mock_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a mock config file and set env vars."""
    monkeypatch.setenv("ZOTERO_API_KEY", "test_zotero_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")

    config_content = """
zotero:
  library_id: "12345"
  library_type: "user"
  api_key: "${ZOTERO_API_KEY}"
  inbox_collection: "Inbox"

llm:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "openai/gpt-4.1-mini"

parser:
  max_pages: 10
  cache_dir: ".cache/parsed"

processing:
  batch_size: 5
  dry_run: false
  add_summary_note: true

collections:
  - name: "ML / Deep Learning"
    description: "Machine learning papers"
    keywords: ["neural", "deep learning"]
  - name: "Review Later"
    description: "Unclear"
    keywords: []

tags:
  - name: "foundational"
    description: "Classic paper"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_help_output(self) -> None:
        """Test main help shows available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "process" in result.stdout
        assert "status" in result.stdout
        assert "config" in result.stdout

    def test_process_help(self) -> None:
        """Test process command help."""
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "dry-run" in result.stdout

    def test_config_help(self) -> None:
        """Test config command help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout


class TestConfigValidate:
    """Tests for config validate command."""

    def test_validate_valid_config(
        self, mock_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test validating a valid config file."""
        result = runner.invoke(app, ["config", "validate", str(mock_config)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_validate_invalid_config(self, tmp_path: Path) -> None:
        """Test validating an invalid config file."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("zotero:\n  library_type: user\n")

        result = runner.invoke(app, ["config", "validate", str(invalid_config)])
        assert result.exit_code == 1
        assert "error" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_validate_nonexistent_config(self) -> None:
        """Test validating a nonexistent config file."""
        result = runner.invoke(app, ["config", "validate", "/nonexistent/config.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()


class TestProcessCommand:
    """Tests for process command."""

    def test_process_dry_run(
        self, mock_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test process command with dry-run flag."""
        with patch("paperflow.cli.ZoteroClient") as mock_zotero_cls:
            mock_zotero = MagicMock()
            mock_zotero.get_inbox_items.return_value = []
            mock_zotero_cls.return_value = mock_zotero

            result = runner.invoke(
                app, ["process", "--config", str(mock_config), "--dry-run"]
            )

        assert result.exit_code == 0
        assert "dry run" in result.stdout.lower() or "no items" in result.stdout.lower()

    def test_process_no_items(
        self, mock_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test process command when inbox is empty."""
        with patch("paperflow.cli.ZoteroClient") as mock_zotero_cls:
            mock_zotero = MagicMock()
            mock_zotero.get_inbox_items.return_value = []
            mock_zotero_cls.return_value = mock_zotero

            result = runner.invoke(app, ["process", "--config", str(mock_config)])

        assert result.exit_code == 0
        assert "no items" in result.stdout.lower() or "0" in result.stdout

    def test_process_with_items(
        self, mock_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test process command with items to process."""
        from paperflow.models import (
            Classification,
            PaperSummary,
            PaperType,
            ParsedPaper,
            ZoteroItem,
        )

        mock_item = ZoteroItem(
            key="ITEM001",
            title="Test Paper",
            creators=["Smith, J."],
            item_type="journalArticle",
            collections=["INBOX"],
            tags=[],
            has_pdf=True,
            pdf_attachment_key="PDF001",
        )

        mock_paper = ParsedPaper(
            title="Test Paper",
            abstract="Abstract",
            full_text="Content",
            page_count=5,
            truncated=False,
        )

        mock_summary = PaperSummary(
            summary="A test paper",
            key_points=["Point 1"],
            methods="Method",
            paper_type=PaperType.EMPIRICAL,
        )

        mock_classification = Classification(
            collections=["ML / Deep Learning"],
            tags=["foundational"],
            confidence=0.9,
            reasoning="Test",
        )

        with (
            patch("paperflow.cli.ZoteroClient") as mock_zotero_cls,
            patch("paperflow.cli.PDFParser") as mock_parser_cls,
            patch("paperflow.cli.Classifier") as mock_classifier_cls,
            patch("paperflow.cli.asyncio.run") as mock_async_run,
        ):
            mock_zotero = MagicMock()
            mock_zotero.get_inbox_items.return_value = [mock_item]
            mock_zotero.is_processed.return_value = False
            mock_zotero.get_item_pdf.return_value = b"pdf bytes"
            mock_zotero.get_collection_key.return_value = "COLL123"
            mock_zotero_cls.return_value = mock_zotero

            mock_parser = MagicMock()
            mock_parser.parse.return_value = mock_paper
            mock_parser_cls.return_value = mock_parser

            mock_classifier = MagicMock()
            mock_async_run.return_value = (mock_summary, mock_classification)
            mock_classifier_cls.return_value = mock_classifier

            result = runner.invoke(app, ["process", "--config", str(mock_config)])

        assert result.exit_code == 0


class TestStatusCommand:
    """Tests for status command."""

    def test_status_no_daemon(self, mock_config: Path) -> None:
        """Test status when no daemon is running."""
        result = runner.invoke(app, ["status", "--config", str(mock_config)])
        assert result.exit_code == 0
        # Should indicate daemon is not running
        assert "not running" in result.stdout.lower() or "status" in result.stdout.lower()
