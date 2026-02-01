"""CLI for paperflow."""

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from paperflow.classifier import Classifier
from paperflow.config import load_config
from paperflow.logging_config import get_logger, setup_logging
from paperflow.models import Classification, PaperSummary, ProcessingResult, ProcessingStatus
from paperflow.parser import PDFParseError, PDFParser
from paperflow.webdav import WebDAVClient
from paperflow.zotero import ZoteroClient, ZoteroError

# Logger will be configured when commands run
logger = get_logger("cli")

app = typer.Typer(
    name="paperflow",
    help="Smart paper sorting and summarization for Zotero",
)
config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")

console = Console()

DEFAULT_CONFIG_PATH = Path("config.yaml")
PID_FILE = Path(".paperflow.pid")


def get_config_path(config: Path | None) -> Path:
    """Get config path, using default if not specified."""
    return config if config else DEFAULT_CONFIG_PATH


@config_app.command("validate")
def config_validate(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
) -> None:
    """Validate a configuration file."""
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    try:
        cfg = load_config(config_path)
        console.print("[green]Config is valid![/green]")
        console.print(f"  Zotero library: {cfg.zotero.library_id}")
        console.print(f"  LLM model: {cfg.llm.model}")
        console.print(f"  Collections: {len(cfg.collections)}")
        console.print(f"  Tags: {len(cfg.tags)}")
    except Exception as e:
        console.print(f"[red]Error: Invalid config - {e}[/red]")
        raise typer.Exit(1) from None


@app.command("process")
def process(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview without changes")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging to console")
    ] = False,
) -> None:
    """Process papers in the inbox."""
    # Initialize logging
    setup_logging(verbose=verbose)

    config_path = get_config_path(config)

    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        raise typer.Exit(1) from None

    # Override dry_run from CLI flag
    if dry_run:
        cfg.processing.dry_run = True

    if cfg.processing.dry_run:
        console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")

    # Initialize components
    try:
        webdav = WebDAVClient(cfg.webdav) if cfg.webdav else None
        zotero = ZoteroClient(cfg.zotero, webdav=webdav)
        parser = PDFParser(cfg.parser)
        classifier = Classifier(cfg.llm, cfg.collections, cfg.tags)
    except Exception as e:
        console.print(f"[red]Error initializing components: {e}[/red]")
        raise typer.Exit(1) from None

    # Fetch inbox items
    try:
        items = zotero.get_inbox_items()
    except ZoteroError as e:
        console.print(f"[red]Error fetching items: {e}[/red]")
        raise typer.Exit(1) from None

    # Filter out already processed items
    items_to_process = [item for item in items if not zotero.is_processed(item)]

    if not items_to_process:
        logger.info("No items to process")
        console.print("[green]No items to process.[/green]")
        return

    logger.info(f"Found {len(items_to_process)} items to process (batch_size={cfg.processing.batch_size})")
    console.print(f"Found {len(items_to_process)} items to process")

    # Limit to batch size
    batch = items_to_process[: cfg.processing.batch_size]
    results: list[ProcessingResult] = []

    for item in batch:
        console.print(f"\nProcessing: [bold]{item.title}[/bold]")
        logger.info(f"Processing item: {item.key} - {item.title}")

        # Skip items without PDF
        if not item.has_pdf or not item.pdf_attachment_key:
            result = ProcessingResult(
                item_key=item.key,
                status=ProcessingStatus.SKIPPED,
                error="No PDF attachment",
            )
            results.append(result)
            logger.info(f"Skipping item {item.key}: No PDF attachment")
            console.print("  [yellow]Skipped: No PDF attachment[/yellow]")
            # Mark as skipped so it won't be processed again
            if not cfg.processing.dry_run:
                zotero.mark_as_skipped(item.key, "No PDF attachment")
            continue

        # Download and parse PDF
        try:
            logger.debug(f"Downloading PDF for item {item.key}")
            pdf_bytes = zotero.get_item_pdf(item.pdf_attachment_key)
            if pdf_bytes is None:
                raise PDFParseError("Could not download PDF")

            logger.debug(f"Parsing PDF for item {item.key}")
            parsed = parser.parse(pdf_bytes, cache_key=item.key)
            logger.info(f"Parsed {parsed.page_count} pages for item {item.key}")
            console.print(f"  Parsed {parsed.page_count} pages")
        except PDFParseError as e:
            result = ProcessingResult(
                item_key=item.key,
                status=ProcessingStatus.FAILED,
                error=f"PDF parsing failed: {e}",
            )
            results.append(result)
            logger.error(f"PDF parsing failed for item {item.key}: {e}")
            console.print(f"  [red]Failed: {e}[/red]")
            continue

        # Classify with LLM
        try:
            logger.info(f"Classifying item {item.key}")
            summary, classification = asyncio.run(classifier.process(parsed))
            logger.info(
                f"Classification complete for {item.key}: "
                f"collections={classification.collections}, "
                f"tags={classification.tags}, "
                f"confidence={classification.confidence:.0%}"
            )
            console.print(f"  Summary: {summary.summary[:100]}...")
            console.print(f"  Collections: {', '.join(classification.collections)}")
            console.print(f"  Tags: {', '.join(classification.tags)}")
            console.print(f"  Confidence: {classification.confidence:.0%}")
        except Exception as e:
            result = ProcessingResult(
                item_key=item.key,
                status=ProcessingStatus.FAILED,
                error=f"Classification failed: {e}",
            )
            results.append(result)
            logger.error(f"Classification failed for item {item.key}: {e}")
            console.print(f"  [red]Classification failed: {e}[/red]")
            continue

        # Apply changes (unless dry run)
        if not cfg.processing.dry_run:
            try:
                logger.info(f"Applying changes to item {item.key}")

                # Add to collections (create if they don't exist)
                for coll_name in classification.collections:
                    logger.debug(f"Adding item {item.key} to collection '{coll_name}'")
                    coll_key = zotero.get_or_create_collection(coll_name)
                    zotero.add_to_collection(item.key, coll_key)

                # Add tags
                logger.debug(f"Adding tags to item {item.key}: {classification.tags}")
                zotero.add_tags(item.key, classification.tags)

                # Add summary note
                if cfg.processing.add_summary_note:
                    logger.debug(f"Adding summary note to item {item.key}")
                    note_html = _format_summary_note(summary, classification)
                    zotero.add_note(item.key, note_html)

                # Mark as processed
                zotero.mark_as_processed(item.key)

                logger.info(f"Successfully updated item {item.key}")
                console.print("  [green]Updated successfully[/green]")
            except Exception as e:
                result = ProcessingResult(
                    item_key=item.key,
                    status=ProcessingStatus.FAILED,
                    error=f"Update failed: {e}",
                )
                results.append(result)
                logger.error(f"Update failed for item {item.key}: {e}")
                console.print(f"  [red]Update failed: {e}[/red]")
                continue

        result = ProcessingResult(
            item_key=item.key,
            status=ProcessingStatus.COMPLETED,
            summary=summary,
            classification=classification,
        )
        results.append(result)

    # Print summary
    _print_summary(results)


@app.command("status")
def status(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging to console")
    ] = False,
) -> None:
    """Show processing status."""
    # Initialize logging
    setup_logging(verbose=verbose)

    config_path = get_config_path(config)

    # Check daemon status
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            console.print(f"[green]Daemon running (PID: {pid})[/green]")
        except (ValueError, OSError):
            console.print("[yellow]Daemon status unknown (corrupt PID file)[/yellow]")
    else:
        console.print("[yellow]Daemon not running[/yellow]")

    # Try to load config and show queue
    try:
        cfg = load_config(config_path)
        zotero = ZoteroClient(cfg.zotero)
        items = zotero.get_inbox_items()
        unprocessed = [i for i in items if not zotero.is_processed(i)]
        console.print("\nInbox status:")
        console.print(f"  Total items: {len(items)}")
        console.print(f"  Unprocessed: {len(unprocessed)}")
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch inbox status: {e}[/yellow]")


@app.command("start")
def start(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
    interval: Annotated[
        int, typer.Option("--interval", "-i", help="Polling interval in seconds")
    ] = 300,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging to console")
    ] = False,
) -> None:
    """Start the daemon (blocking).

    Runs continuously, polling Zotero for new papers every INTERVAL seconds.
    Default interval is 300 seconds (5 minutes).
    """
    from paperflow.daemon import Daemon, DaemonError

    # Initialize logging
    setup_logging(verbose=verbose)

    config_path = get_config_path(config)

    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"[green]Starting daemon with {interval}s polling interval...[/green]")
    logger.info(f"Starting daemon with interval={interval}s")

    daemon = Daemon(cfg, interval=interval)

    try:
        asyncio.run(daemon.run())
    except DaemonError as e:
        console.print(f"[red]Daemon error: {e}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Daemon stopped by user[/yellow]")


def _format_summary_note(summary: PaperSummary, classification: Classification) -> str:
    """Format summary and classification as HTML note."""
    key_points_html = "\n".join(f"<li>{p}</li>" for p in summary.key_points)
    tags_html = ", ".join(classification.tags) if classification.tags else "None"

    return f"""<h2>Summary</h2>
<p>{summary.summary}</p>

<h3>Key Points</h3>
<ul>
{key_points_html}
</ul>

<h3>Methods</h3>
<p>{summary.methods}</p>

<h3>Classification</h3>
<p><strong>Collections:</strong> {', '.join(classification.collections)}</p>
<p><strong>Tags:</strong> {tags_html}</p>
<p><strong>Confidence:</strong> {classification.confidence:.0%}</p>
<p><em>{classification.reasoning}</em></p>

<hr>
<p><small>Generated by paperflow</small></p>
"""


def _print_summary(results: list[ProcessingResult]) -> None:
    """Print processing summary table."""
    table = Table(title="Processing Results")
    table.add_column("Item Key")
    table.add_column("Status")
    table.add_column("Details")

    for result in results:
        status_str = result.status.value
        if result.status == ProcessingStatus.COMPLETED:
            status_str = f"[green]{status_str}[/green]"
        elif result.status == ProcessingStatus.FAILED:
            status_str = f"[red]{status_str}[/red]"
        elif result.status == ProcessingStatus.SKIPPED:
            status_str = f"[yellow]{status_str}[/yellow]"

        details = ""
        if result.error:
            details = result.error
        elif result.classification:
            details = ", ".join(result.classification.collections)

        table.add_row(result.item_key, status_str, details)

    console.print(table)

    completed = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
    failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
    skipped = sum(1 for r in results if r.status == ProcessingStatus.SKIPPED)

    console.print(
        f"\nTotal: {len(results)} | Completed: {completed} | "
        f"Failed: {failed} | Skipped: {skipped}"
    )


if __name__ == "__main__":
    app()
