# Insult — Discord Bot with Longitudinal Memory + Claude API

## Quick Start
```bash
python -m insult run          # Start the bot
python -m insult db-stats     # Show memory stats
pytest -v --cov               # Run tests
ruff check . && ruff format . # Lint + format
```

## Project Structure
```
insult/
├── __main__.py          # CLI (typer): run, db-stats, db-clean
├── app.py               # DI container (Container dataclass)
├── bot.py               # Discord lifecycle, events, health check
├── config.py            # Pydantic Settings (.env)
├── cogs/
│   ├── chat.py          # !chat — conversation with memory + attachments
│   └── utility.py       # !ping, !memoria, !buscar, !perfil
└── core/
    ├── attachments.py   # Classify + process Discord attachments (images, text, PDFs)
    ├── character.py     # Break detection (20 regex), sanitization, adaptive prompt
    ├── errors.py        # In-character error responses (never expose "Claude")
    ├── llm.py           # Claude API client + character break retry
    ├── memory.py        # SQLite longitudinal memory + user style profiles
    └── style.py         # User style profiling (EMA: language, formality, tech level)
```

## Rules
Detailed rules live in `.claude/rules/`:
- **architecture.md** — Project structure, patterns, dependencies, attachments, security
- **robustness.md** — Error handling, rate limiting, LLM resilience, logging, lifecycle
- **testing.md** — Unit tests, E2E testing with Discord MCP, test mode, checklist
- **workflow.md** — Git workflow, CI pipeline (4 layers), local dev flow
- **persona.md** — Character identity, guard system, style adaptation, prompt architecture

## CI Pipeline (GitHub Actions)
4 layers, all must pass:
1. **Ruff Lint & Format** — code quality gate (~8s)
2. **Tests + Coverage** — 98 tests, 80% minimum (~27s)
3. **Dependency Audit** — pip-audit for CVEs (~23s)
4. **Code Security** — bandit SAST (~7s)

## E2E Testing
Use the Discord MCP server (`barryyip0625/mcp-discord`) to test the bot end-to-end.
See `.claude/rules/testing.md` for full instructions.
Key: start bot locally → send commands via MCP → verify responses.

## Key Patterns
- **Persona**: `persona.md` defines the Insult character — never expose "Claude" or break character
- **Style Adaptation**: Bot profiles each user (EMA) and adjusts tone without losing identity
- **Memory**: Append-only SQLite, 50 recent messages + keyword search, user style profiles
- **Attachments**: Images (vision), text/code (25+ extensions), PDFs — 5MB limit
- **Character Guard**: 20 regex patterns detect breaks → auto-retry → sanitize as fallback
