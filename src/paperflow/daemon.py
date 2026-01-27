"""Background daemon for paperflow."""

import asyncio
import logging
import os
import signal
from pathlib import Path

from paperflow.classifier import Classifier
from paperflow.config import AppConfig
from paperflow.models import ProcessingResult, ProcessingStatus
from paperflow.parser import PDFParseError, PDFParser
from paperflow.zotero import ZoteroClient

logger = logging.getLogger(__name__)

DEFAULT_PID_FILE = Path(".paperflow.pid")


class DaemonError(Exception):
    """Error raised for daemon issues."""

    pass


class Daemon:
    """Background service that polls Zotero and processes papers."""

    def __init__(
        self,
        config: AppConfig,
        interval: int = 300,
        pid_file: Path = DEFAULT_PID_FILE,
    ) -> None:
        """Initialize the daemon.

        Args:
            config: Application configuration.
            interval: Polling interval in seconds (default 5 minutes).
            pid_file: Path to PID file.
        """
        self.config = config
        self.interval = interval
        self.pid_file = pid_file
        self.running = False
        self._zotero: ZoteroClient | None = None
        self._parser: PDFParser | None = None
        self._classifier: Classifier | None = None

    def is_already_running(self) -> bool:
        """Check if another daemon instance is already running.

        Returns:
            True if another daemon is running.
        """
        if not self.pid_file.exists():
            return False

        try:
            pid = int(self.pid_file.read_text().strip())
            # Check if process is running
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            # Invalid PID, process not running, or no permission
            return False

    def _write_pid_file(self) -> None:
        """Write current PID to file."""
        self.pid_file.write_text(str(os.getpid()))

    def _remove_pid_file(self) -> None:
        """Remove PID file."""
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _setup_signal_handlers(self) -> None:
        """Set up handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _init_components(self) -> None:
        """Initialize Zotero client, parser, and classifier."""
        self._zotero = ZoteroClient(self.config.zotero)
        self._parser = PDFParser(self.config.parser)
        self._classifier = Classifier(
            self.config.llm,
            self.config.collections,
            self.config.tags,
        )

    async def run_once(self) -> list[ProcessingResult]:
        """Run a single processing cycle.

        Returns:
            List of processing results.
        """
        if self._zotero is None:
            self._init_components()

        assert self._zotero is not None
        assert self._parser is not None
        assert self._classifier is not None

        results: list[ProcessingResult] = []

        try:
            items = self._zotero.get_inbox_items()
        except Exception as e:
            logger.error(f"Failed to fetch items: {e}")
            return results

        # Filter unprocessed items
        items_to_process = [
            item for item in items if not self._zotero.is_processed(item)
        ]

        if not items_to_process:
            logger.info("No items to process")
            return results

        logger.info(f"Found {len(items_to_process)} items to process")

        # Limit to batch size
        batch = items_to_process[: self.config.processing.batch_size]

        for item in batch:
            logger.info(f"Processing: {item.title}")

            # Skip items without PDF
            if not item.has_pdf or not item.pdf_attachment_key:
                result = ProcessingResult(
                    item_key=item.key,
                    status=ProcessingStatus.SKIPPED,
                    error="No PDF attachment",
                )
                results.append(result)
                continue

            # Download and parse PDF
            try:
                pdf_bytes = self._zotero.get_item_pdf(item.pdf_attachment_key)
                if pdf_bytes is None:
                    raise PDFParseError("Could not download PDF")

                parsed = self._parser.parse(pdf_bytes, cache_key=item.key)
            except PDFParseError as e:
                result = ProcessingResult(
                    item_key=item.key,
                    status=ProcessingStatus.FAILED,
                    error=f"PDF parsing failed: {e}",
                )
                results.append(result)
                continue

            # Classify with LLM
            try:
                summary, classification = await self._classifier.process(parsed)
            except Exception as e:
                result = ProcessingResult(
                    item_key=item.key,
                    status=ProcessingStatus.FAILED,
                    error=f"Classification failed: {e}",
                )
                results.append(result)
                continue

            # Apply changes
            if not self.config.processing.dry_run:
                try:
                    # Add to collections
                    for coll_name in classification.collections:
                        coll_key = self._zotero.get_collection_key(coll_name)
                        if coll_key:
                            self._zotero.add_to_collection(item.key, coll_key)

                    # Add tags
                    self._zotero.add_tags(item.key, classification.tags)

                    # Add summary note
                    if self.config.processing.add_summary_note:
                        note_html = self._format_note(summary, classification)
                        self._zotero.add_note(item.key, note_html)

                    # Mark as processed
                    self._zotero.mark_as_processed(item.key)
                except Exception as e:
                    result = ProcessingResult(
                        item_key=item.key,
                        status=ProcessingStatus.FAILED,
                        error=f"Update failed: {e}",
                    )
                    results.append(result)
                    continue

            result = ProcessingResult(
                item_key=item.key,
                status=ProcessingStatus.COMPLETED,
                summary=summary,
                classification=classification,
            )
            results.append(result)
            logger.info(f"Completed: {item.title}")

        return results

    def _format_note(self, summary, classification) -> str:  # type: ignore[no-untyped-def]
        """Format summary as HTML note."""
        key_points_html = "\n".join(f"<li>{p}</li>" for p in summary.key_points)
        tags_html = ", ".join(classification.tags) if classification.tags else "None"

        return f"""<h2>Summary</h2>
<p>{summary.summary}</p>
<h3>Key Points</h3>
<ul>{key_points_html}</ul>
<h3>Methods</h3>
<p>{summary.methods}</p>
<h3>Classification</h3>
<p><strong>Collections:</strong> {', '.join(classification.collections)}</p>
<p><strong>Tags:</strong> {tags_html}</p>
<p><strong>Confidence:</strong> {classification.confidence:.0%}</p>
<hr><p><small>Generated by paperflow</small></p>
"""

    async def run(self) -> None:
        """Start the daemon loop.

        Raises:
            DaemonError: If already running.
        """
        if self.is_already_running():
            raise DaemonError("Daemon is already running")

        self._write_pid_file()
        self._setup_signal_handlers()
        self._init_components()
        self.running = True

        logger.info(f"Daemon started, polling every {self.interval}s")

        try:
            while self.running:
                try:
                    await self.run_once()
                except Exception as e:
                    logger.error(f"Error in processing cycle: {e}")

                # Wait for next cycle
                for _ in range(self.interval):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
        finally:
            self._remove_pid_file()
            logger.info("Daemon stopped")

    def stop(self) -> None:
        """Stop the daemon gracefully."""
        self.running = False
