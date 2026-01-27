"""Tests for daemon service."""

import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paperflow.config import (
    AppConfig,
    CollectionDef,
    LLMConfig,
    ParserConfig,
    ProcessingConfig,
    TagDef,
    ZoteroConfig,
)
from paperflow.daemon import Daemon


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Create a test app configuration."""
    return AppConfig(
        zotero=ZoteroConfig(
            library_id="12345",
            library_type="user",
            api_key="test_key",
            inbox_collection="Inbox",
        ),
        llm=LLMConfig(
            provider="openrouter",
            api_key="test_key",
            model="test-model",
        ),
        parser=ParserConfig(
            max_pages=10,
            cache_dir=str(tmp_path / "cache"),
        ),
        processing=ProcessingConfig(
            batch_size=5,
            dry_run=False,
            add_summary_note=True,
        ),
        collections=[
            CollectionDef(name="Test", description="Test collection", keywords=[])
        ],
        tags=[TagDef(name="test-tag", description="A test tag")],
    )


@pytest.fixture
def pid_file(tmp_path: Path) -> Path:
    """Create a temp PID file path."""
    return tmp_path / "paperflow.pid"


class TestDaemon:
    """Tests for Daemon class."""

    def test_init(self, app_config: AppConfig, pid_file: Path) -> None:
        """Test daemon initialization."""
        daemon = Daemon(app_config, interval=60, pid_file=pid_file)
        assert daemon.interval == 60
        assert daemon.pid_file == pid_file
        assert not daemon.running

    def test_pid_file_creation(self, app_config: AppConfig, pid_file: Path) -> None:
        """Test PID file is created on start."""
        daemon = Daemon(app_config, interval=60, pid_file=pid_file)
        daemon._write_pid_file()

        assert pid_file.exists()
        pid = int(pid_file.read_text().strip())
        assert pid > 0

    def test_pid_file_removal(self, app_config: AppConfig, pid_file: Path) -> None:
        """Test PID file is removed on stop."""
        daemon = Daemon(app_config, interval=60, pid_file=pid_file)
        daemon._write_pid_file()
        assert pid_file.exists()

        daemon._remove_pid_file()
        assert not pid_file.exists()

    def test_already_running_check(self, app_config: AppConfig, pid_file: Path) -> None:
        """Test detection of already running daemon."""
        # Write a fake PID file with our own PID (simulating running)
        import os

        pid_file.write_text(str(os.getpid()))

        daemon = Daemon(app_config, interval=60, pid_file=pid_file)
        assert daemon.is_already_running()

    def test_stale_pid_file(self, app_config: AppConfig, pid_file: Path) -> None:
        """Test handling of stale PID file (process not running)."""
        # Write a PID that definitely doesn't exist
        pid_file.write_text("999999999")

        daemon = Daemon(app_config, interval=60, pid_file=pid_file)
        assert not daemon.is_already_running()

    @pytest.mark.asyncio
    async def test_run_once(self, app_config: AppConfig, pid_file: Path) -> None:
        """Test single processing run."""
        daemon = Daemon(app_config, interval=60, pid_file=pid_file)

        with (
            patch("paperflow.daemon.ZoteroClient") as mock_zotero_cls,
            patch("paperflow.daemon.PDFParser") as mock_parser_cls,
            patch("paperflow.daemon.Classifier") as mock_classifier_cls,
        ):
            mock_zotero = MagicMock()
            mock_zotero.get_inbox_items.return_value = []
            mock_zotero_cls.return_value = mock_zotero

            mock_parser_cls.return_value = MagicMock()
            mock_classifier_cls.return_value = MagicMock()

            results = await daemon.run_once()

        assert results == []
        mock_zotero.get_inbox_items.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(
        self, app_config: AppConfig, pid_file: Path
    ) -> None:
        """Test that stop() sets running to False."""
        daemon = Daemon(app_config, interval=60, pid_file=pid_file)
        daemon.running = True

        daemon.stop()

        assert not daemon.running


class TestDaemonSignals:
    """Tests for signal handling."""

    def test_signal_handler_setup(
        self, app_config: AppConfig, pid_file: Path
    ) -> None:
        """Test signal handlers are set up correctly."""
        daemon = Daemon(app_config, interval=60, pid_file=pid_file)

        with patch("signal.signal") as mock_signal:
            daemon._setup_signal_handlers()

            # Should set up handlers for SIGTERM and SIGINT
            calls = mock_signal.call_args_list
            signal_types = [call[0][0] for call in calls]
            assert signal.SIGTERM in signal_types
            assert signal.SIGINT in signal_types
