# Architecture Rules

## Project Structure
- `insult/config.py` — Pydantic Settings singleton, all config via .env
- `insult/app.py` — DI container (Container dataclass), wires all deps
- `insult/bot.py` — Discord lifecycle, events, signal handling, health check
- `insult/cogs/chat.py` — !chat command with style adaptation + attachments
- `insult/cogs/utility.py` — !ping, !memoria, !buscar, !perfil commands
- `insult/core/llm.py` — Claude API client (async) + character break retry
- `insult/core/memory.py` — Longitudinal memory (SQLite, append-only) + user profiles
- `insult/core/character.py` — Break detection, sanitization, adaptive prompt building
- `insult/core/errors.py` — In-character error responses, error classification
- `insult/core/style.py` — User style profiling (EMA, language, formality, tech level)
- `insult/core/attachments.py` — Discord attachment processing (images, text, PDFs)
- `persona.md` — System prompt for the Insult persona (root of project)
- `tests/` — pytest suite (unit + cog tests with mocked DI container)
- `pyproject.toml` — ruff config, pytest config, coverage config, bandit config

## Patterns
- DI container via `Container` dataclass in `app.py` — all deps injected into cogs
- Cogs pattern from discord.py — commands grouped by concern (chat, utility)
- Settings singleton with Pydantic BaseSettings (env_file=".env")
- Structured logging via structlog (JSON-ready, never use print())
- Memory is append-only: never delete, only grow ("infinite conversation")
- Context is hierarchical: recent messages (50) + keyword-relevant messages (5)
- All DB operations go through _ensure_connection() for auto-reconnect
- Commands must have @commands.cooldown to prevent token burn
- Character break detection → auto-retry → sanitize as fallback
- User style adaptation via EMA (exponential moving average) with confidence gate (5 msgs)

## Attachments
- Images (png/jpg/gif/webp): sent to Claude as base64 vision blocks
- Text/code (25+ extensions): read as UTF-8, injected as text blocks
- PDFs: sent as base64 document blocks
- Unsupported types: rejected with in-character error message
- Max size: 5MB per attachment
- Multiple attachments per message supported
- Attachment content is NOT stored in longitudinal memory (only the text message)

## Dependencies
- discord.py >= 2.3.0
- anthropic >= 0.42.0
- aiosqlite >= 0.19.0
- pydantic-settings >= 2.1.0
- structlog >= 24.1.0
- typer >= 0.9.0
- ruff >= 0.11.0 (dev)
- pytest >= 8.0.0, pytest-asyncio, pytest-cov (dev)
- bandit >= 1.8.0, pip-audit >= 2.7.0 (dev)

## Security
- .env is gitignored — NEVER commit tokens
- Bot validates required tokens at startup (validate_required)
- All user input is parameterized in SQL (no injection)
- Max message length enforced before sending to LLM (4000 chars)
- Max attachment size enforced (5MB)
- Character breaks auto-detected and sanitized (never expose model identity)
- In-character errors never expose "Claude", "Anthropic", or API internals
