# paperflow_lite

**Smart paper sorting and summarization for Zotero**

## Problem

Research papers accumulate fast — you save interesting stuff from Google News, X, Reddit, but never organize it. The Zotero library becomes an unsorted dumping ground that's hard to navigate and easy to ignore.

## Solution

A lightweight CLI tool that:
1. Fetches unsorted papers from Zotero
2. Extracts/parses PDF content
3. Uses an LLM to generate summaries and classify papers
4. Automatically sorts into Zotero collections (folders) and applies tags
5. Stores structured metadata back in Zotero

## User Stories

```
As a researcher, I want to run a single command that processes my unsorted papers
So that my Zotero library stays organized without manual effort

As a researcher, I want summaries and key points extracted from each paper
So that I can quickly decide if a paper is worth deep reading

As a researcher, I want to define my own categories and tags
So that the sorting matches my research interests and workflow
```

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Zotero    │────▶│  PDF Parse  │────▶│  LLM API    │────▶│   Zotero    │
│  (fetch)    │     │  (docling)  │     │ (OpenRouter)│     │  (update)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      │                                        ▼
      │                              ┌─────────────────┐
      │                              │ prompts/*.md    │
      │                              │ config.yaml     │
      │                              └─────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Zotero Library                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Inbox/   │  │ ML/      │  │ Neuro/   │  │ Methods/ │  ...           │
│  │ Unsorted │  │ Deep     │  │ Cognitive│  │ Stats    │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Zotero Client (`src/zotero.py`)
- Uses `pyzotero` library
- Fetches items from a configurable "inbox" collection (or all unsorted)
- Updates items with: tags, collection membership, notes (for summaries)
- Handles attachments (PDF retrieval)

### 2. PDF Parser (`src/parser.py`)
- Uses `docling` for PDF → text extraction
- Handles edge cases: scanned PDFs, multi-column layouts
- Extracts: title, abstract, full text (or first N pages for cost control)
- Caches parsed content to avoid re-processing

### 3. LLM Classifier (`src/classifier.py`)
- Calls OpenRouter API (model configurable, default: `gpt-oss-120B`)
- Loads prompts from `prompts/*.md` files
- Two-stage processing:
  1. **Summarize**: extract summary, key points, methods
  2. **Classify**: assign collection(s) and tags based on config
- Returns structured output (JSON)

### 4. Config Loader (`src/config.py`)
- Reads `config.yaml` for:
  - Zotero API credentials
  - OpenRouter API key + model selection
  - Collection definitions (name, description, keywords)
  - Tag definitions (name, description, when to apply)
  - Processing options (max pages, batch size, dry-run mode)

### 5. CLI (`src/cli.py`)
- Commands:
  - `paperflow process` — run the full pipeline
  - `paperflow process --dry-run` — show what would happen
  - `paperflow status` — show inbox count, last run
  - `paperflow config validate` — check config file

---

## File Structure

```
paperflow_lite/
├── pyproject.toml          # uv/pip project config
├── config.yaml             # user configuration (gitignored template provided)
├── config.example.yaml     # example config (committed)
├── SPEC.md                 # this file
├── README.md               # usage docs
│
├── prompts/
│   ├── summarize.md        # prompt for summary generation
│   └── classify.md         # prompt for classification
│
├── src/
│   └── paperflow/
│       ├── __init__.py
│       ├── cli.py          # CLI entry point
│       ├── zotero.py       # Zotero API client
│       ├── parser.py       # PDF parsing with docling
│       ├── classifier.py   # LLM classification logic
│       ├── config.py       # Config loading/validation
│       └── models.py       # Pydantic models for data structures
│
└── tests/
    ├── conftest.py         # pytest fixtures
    ├── test_zotero.py      # Zotero client tests (mocked)
    ├── test_parser.py      # Parser tests with sample PDFs
    ├── test_classifier.py  # LLM tests (mocked responses)
    └── fixtures/
        └── sample.pdf      # test PDF
```

---

## Config Schema (`config.yaml`)

```yaml
# Zotero settings
zotero:
  library_id: "12345"           # Your Zotero user/group ID
  library_type: "user"          # "user" or "group"
  api_key: "${ZOTERO_API_KEY}"  # env var reference
  inbox_collection: "Inbox"     # collection to pull from (null = all unsorted)

# LLM settings
llm:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "openai/gpt-4.1-mini"  # configurable
  max_tokens: 2000
  temperature: 0.3

# PDF parsing
parser:
  max_pages: 10                 # limit pages to reduce cost
  cache_dir: ".cache/parsed"    # cache parsed text

# Processing
processing:
  batch_size: 5                 # papers per run
  dry_run: false                # preview without changes
  add_summary_note: true        # add summary as Zotero note

# Collections (folders) — LLM uses descriptions to classify
collections:
  - name: "ML / Deep Learning"
    description: "Machine learning, neural networks, deep learning architectures, training methods"
    keywords: ["neural network", "deep learning", "transformer", "CNN", "RNN"]
  
  - name: "Neuroscience / Cognitive"
    description: "Brain science, cognitive psychology, neurobiology, perception, memory"
    keywords: ["fMRI", "neuron", "cognitive", "brain", "hippocampus"]
  
  - name: "Methods / Statistics"
    description: "Research methods, statistical analysis, experimental design"
    keywords: ["regression", "bayesian", "p-value", "sample size"]
  
  - name: "Review Later"
    description: "Doesn't fit other categories or unclear from abstract"

# Tags — applied based on paper characteristics
tags:
  - name: "⭐ foundational"
    description: "Seminal/classic paper, highly cited, establishes key concepts"
  
  - name: "🔧 methods-focused"
    description: "Primarily about a technique, tool, or methodology"
  
  - name: "📊 empirical"
    description: "Reports original experimental/observational data"
  
  - name: "📝 review/meta"
    description: "Literature review, meta-analysis, or survey paper"
  
  - name: "🚀 cutting-edge"
    description: "Very recent, potentially not yet peer-reviewed, bleeding edge"
  
  - name: "⚠️ needs-pdf"
    description: "Could not access or parse the PDF"
```

---

## Prompts

### `prompts/summarize.md`

```markdown
You are a research assistant helping organize an academic paper library.

Given the following paper content, extract:

1. **Summary** (2-3 sentences): What is this paper about? What problem does it address?
2. **Key Points** (3-5 bullets): Main findings, contributions, or arguments
3. **Methods** (1-2 sentences): What methodology or approach was used?
4. **Paper Type**: One of [empirical, theoretical, review, methods, commentary]

## Paper Content

{content}

## Output Format

Respond in JSON:
```json
{
  "summary": "...",
  "key_points": ["...", "..."],
  "methods": "...",
  "paper_type": "..."
}
```
```

### `prompts/classify.md`

```markdown
You are a research librarian classifying papers into a personal collection.

Given the paper summary and the available collections/tags, determine:
1. Which collection(s) this paper belongs to (1-2 max)
2. Which tags apply

## Paper Summary

{summary}

## Available Collections

{collections}

## Available Tags

{tags}

## Output Format

Respond in JSON:
```json
{
  "collections": ["Collection Name"],
  "tags": ["tag1", "tag2"],
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}
```
```

---

## Data Flow

```
1. FETCH
   └─▶ Zotero API → get items from "Inbox" collection
   └─▶ Filter: has PDF attachment, not already processed

2. PARSE
   └─▶ Download PDF attachment
   └─▶ docling: PDF → structured text
   └─▶ Cache result locally

3. SUMMARIZE
   └─▶ Load prompts/summarize.md
   └─▶ Call LLM with paper text
   └─▶ Parse JSON response → Summary object

4. CLASSIFY
   └─▶ Load prompts/classify.md
   └─▶ Load config.yaml collections/tags
   └─▶ Call LLM with summary + options
   └─▶ Parse JSON response → Classification object

5. UPDATE
   └─▶ Zotero API: add to collection(s)
   └─▶ Zotero API: add tags
   └─▶ Zotero API: create note with summary (optional)
   └─▶ Zotero API: remove from "Inbox" (or add "processed" tag)

6. REPORT
   └─▶ Print summary of actions taken
   └─▶ Log any errors or skipped items
```

---

## Testing Strategy

### Unit Tests
- **Zotero client**: Mock `pyzotero` responses, test fetch/update logic
- **Parser**: Use fixture PDFs, test extraction quality
- **Classifier**: Mock LLM responses, test prompt formatting and response parsing
- **Config**: Test validation, env var substitution, schema errors

### Integration Tests
- End-to-end with test Zotero library (or sandbox)
- Mock LLM to avoid costs, but test real Zotero API

### Test Fixtures
- `fixtures/sample.pdf` — simple 2-page paper
- `fixtures/scanned.pdf` — OCR-needed paper
- `fixtures/config_valid.yaml` — valid config
- `fixtures/config_invalid.yaml` — schema violations

---

## Dependencies

```toml
[project]
name = "paperflow-lite"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pyzotero>=1.5",      # Zotero API
    "docling>=0.1",       # PDF parsing
    "httpx>=0.27",        # HTTP client for OpenRouter
    "pydantic>=2.0",      # Data validation
    "pyyaml>=6.0",        # Config parsing
    "typer>=0.12",        # CLI framework
    "rich>=13.0",         # Pretty terminal output
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "ty>=0.1",            # type checker
    "respx>=0.21",        # HTTP mocking
]

[project.scripts]
paperflow = "paperflow.cli:app"
```

---

## Open Questions

1. **Inbox detection**: Should we use a dedicated "Inbox" collection, or detect "unfiled" items automatically?
2. **Conflict handling**: What if a paper could fit multiple collections equally?
3. **Re-processing**: How to handle papers that were processed before but config changed?
4. **Rate limiting**: Zotero API has limits — batch size and delays needed?
5. **Cost control**: Estimate tokens per paper, add budget warnings?

---

## Future Enhancements (out of scope for v1)

- [ ] Web UI for reviewing classifications before applying
- [ ] Semantic search across summaries
- [ ] Citation network analysis
- [ ] Automatic "reading queue" prioritization
- [ ] Sync summaries to Notion/Obsidian

---

## Success Criteria

- [ ] Can process 10 papers in under 5 minutes
- [ ] Classification accuracy >80% (subjective, user validates)
- [ ] Zero data loss (never deletes or corrupts Zotero data)
- [ ] Config changes don't require code changes
- [ ] Clear error messages when things fail
