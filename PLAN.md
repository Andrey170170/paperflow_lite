# Implementation Plan for paperflow_lite

## Overview

Implementing a CLI tool that automatically organizes Zotero papers using LLM classification. Following TDD approach with strict linting.

## Phase 1: Project Setup

### 1.1 Dependencies
- `pyzotero>=1.5` - Zotero API client
- `docling>=2.0` - PDF parsing
- `httpx>=0.27` - HTTP client for OpenRouter
- `pydantic>=2.0` - Data validation
- `pyyaml>=6.0` - Config parsing
- `typer>=0.12` - CLI framework
- `rich>=13.0` - Pretty terminal output
- `python-dotenv>=1.0` - Environment variable loading

Dev: pytest, pytest-asyncio, ruff, respx

## Phase 2: Core Models (`models.py`)
- `PaperSummary` - Summary, key_points, methods, paper_type
- `Classification` - collections, tags, confidence, reasoning
- `ZoteroItem` - Representation of a Zotero library item
- `ProcessingResult` - Result of processing a single paper

## Phase 3: Configuration (`config.py`)
- Environment variable substitution
- Validation with helpful error messages
- Default values
- Load from YAML file

## Phase 4: Zotero Client (`zotero.py`)
- Fetch items from inbox collection
- Download PDF attachments
- Add to collections, tags, notes

## Phase 5: PDF Parser (`parser.py`)
- PDF to text extraction with docling
- Caching parsed content

## Phase 6: LLM Classifier (`classifier.py`)
- Two-stage: summarize then classify
- OpenRouter API integration

## Phase 7: CLI (`cli.py`)
- process, status, serve, stop commands

## Phase 8: Daemon (`daemon.py`)
- Background polling service

## Implementation Order
1. Project setup + models
2. Config loading
3. PDF parser
4. Zotero client
5. LLM classifier
6. CLI commands
7. Daemon service
