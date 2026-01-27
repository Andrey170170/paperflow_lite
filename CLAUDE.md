# CLAUDE.md - Coding Rules

## Project Setup

- **Package manager**: `uv` — use `uv add <package>` to add deps, `uv run` to execute
- **Python version**: 3.13+
- **API keys**: Available in `.env` (ZOTERO_API_KEY, OPENROUTER_API_KEY)

## Workflow

1. **Read SPEC.md first** — understand what you're building
2. **Create a detailed plan** before coding anything
3. **Write tests FIRST** — then implementation (TDD)
4. **Iterate**: test → implement → verify → next component

## Code Quality Rules

### Linting (STRICT)

- **NEVER silence linter errors** at file level (no `# noqa` for entire files, no `# type: ignore` blanket ignores)
- Fix every lint error properly — understand why it's complaining
- If a specific line genuinely needs an exception, use a targeted inline comment with explanation
- All code must pass `ruff check` and `ty check` (or pyright/mypy) with zero errors

### Testing

- **Aim for full test coverage** — every function, every branch
- Use pytest, pytest-asyncio for async code
- Mock external services (Zotero API, OpenRouter) — don't make real API calls in tests
- Test fixtures go in `tests/fixtures/`

### API Calls (COST CONTROL)

- **Test API calls lightly** — we have limited budget on both Zotero and OpenRouter
- Use mocks for tests, only hit real APIs for manual verification
- When you DO test real APIs, use minimal data (1 item, shortest prompts)

## Structure

Follow the file structure in SPEC.md:
```
src/paperflow/
├── __init__.py
├── cli.py
├── zotero.py
├── parser.py
├── classifier.py
├── config.py
└── models.py
```

## Git

- Commit frequently with clear messages
- Don't commit `.env` or secrets

## When Stuck

- Re-read the relevant section of SPEC.md
- Check if there's a simpler approach
- If genuinely blocked, say so clearly
