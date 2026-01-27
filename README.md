# paperflow_lite

Smart paper sorting and summarization for Zotero.

> 🚧 **Status**: Planning phase — see [SPEC.md](./SPEC.md) for design

## What it does

1. Fetches unsorted papers from your Zotero library
2. Parses PDFs and extracts content
3. Uses an LLM to summarize and classify each paper
4. Automatically organizes into collections and applies tags

## Stack

- **Python 3.11+** (managed with `uv`)
- **pyzotero** — Zotero API client
- **docling** — PDF parsing
- **OpenRouter** — LLM API (configurable model)
- **Typer + Rich** — CLI

## Setup

```bash
# Clone and enter
cd paperflow_lite

# Create venv and install
uv venv
uv pip install -e ".[dev]"

# Copy and edit config
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys and collections

# Run
paperflow process --dry-run
```

## Development

```bash
# Lint
ruff check src/ tests/

# Type check
ty check src/

# Test
pytest
```

## License

MIT
