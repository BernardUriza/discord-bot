# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
```bash
# Run
python -m insult run              # Start the bot
python -m insult db-stats         # Show memory stats
python -m insult db-clean         # Clean old data

# Test
pytest -v --cov                   # All tests + coverage report
pytest tests/test_chat_cog.py -v  # Single test file
pytest -k "test_detect_break" -v  # Single test by name

# Lint & Format
ruff check . && ruff format .     # Lint + format (run before every commit)
ruff check --fix .                # Auto-fix lint issues

# Full CI locally
ruff check . && ruff format --check . && pytest -v --cov --cov-fail-under=75 && bandit -r insult/ -c pyproject.toml && pip-audit
```

## Architecture

**Request flow**: User message → `ChatCog.on_message` (cogs/chat.py) → memory store + profile update → context build (recent 50 + 5 keyword-relevant) → `build_adaptive_prompt` (core/character.py) layers system prompt → `LLMClient.chat` (core/llm.py) with break detection retry → response chunked to Discord (1990 char limit).

**DI container**: `app.py` creates a `Container` dataclass holding Settings, MemoryStore, LLMClient, and Bot. Cogs receive the container via constructor. All tests mock this container (see `tests/conftest.py` for fixtures).

**Config**: `config.py` uses Pydantic BaseSettings with `.env` file taking priority over shell env vars (custom source ordering). Settings singleton is created at module import time — tests that import from `insult.core.*` modules work fine, but importing `insult.config` directly requires `.env` to exist.

**Chat flow (no prefix)**: The bot responds to ALL messages in channels (via `on_message` listener), not just `!chat`. Messages starting with `!` are ignored by the listener (handled as commands). Per-user cooldown is 15s.

**System prompt composition** (core/character.py `build_adaptive_prompt`):
1. Base persona from `persona.md` (loaded at startup into settings.system_prompt)
2. Style adaptation hints appended per-user (if profile has 5+ messages)
3. Identity reinforcement suffix for conversations >10 messages

**Memory** (core/memory.py): Append-only SQLite via aiosqlite. Context is built per-channel (all users see same conversation), but style profiles are per-user. `_ensure_connection()` auto-reconnects before every DB operation.

**Azure backup**: Optional. If `AZURE_STORAGE_CONNECTION_STRING` is set, DB uploads every 10 min and downloads on first startup.

## Testing Patterns

Tests use a fully mocked DI container (`conftest.py`). To test a cog:
1. Create the cog with `mock_container` fixture
2. Call the method directly (e.g., `cog._respond(mock_message, "text")`)
3. Assert on `mock_llm.chat`, `mock_memory.store`, `message.channel.send`

Coverage threshold is 75% (`pyproject.toml`). Pure I/O modules (bot.py, app.py, config.py, llm.py, memory.py) are excluded from coverage.

## Rules
Detailed rules in `.claude/rules/`: architecture, robustness, testing, workflow, persona. Key non-obvious rules:
- **Never expose "Claude"/"Anthropic"/"AI"** in bot responses — character guard auto-retries and sanitizes
- Error messages to users must be in-character (via `core/errors.py`)
- All logging via structlog, never print()
- DB write failures are logged but don't crash commands
