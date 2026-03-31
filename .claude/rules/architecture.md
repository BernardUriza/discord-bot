# Architecture Rules

## Project Structure
- `insult/config.py` — Pydantic Settings singleton, all config via .env
- `insult/app.py` — DI container (Container dataclass), wires all deps
- `insult/bot.py` — Discord lifecycle, events, signal handling, health check
- `insult/cogs/chat.py` — !chat command with style adaptation
- `insult/cogs/utility.py` — !ping, !memoria, !buscar, !perfil commands
- `insult/core/llm.py` — Claude API client (async) + character break retry
- `insult/core/memory.py` — Longitudinal memory (SQLite, append-only) + user profiles
- `insult/core/character.py` — Break detection, sanitization, adaptive prompt building
- `insult/core/errors.py` — In-character error responses, error classification
- `insult/core/style.py` — User style profiling (EMA, language, formality, tech level)
- `persona.md` — System prompt for the Insult persona
- `tests/` — pytest suite (unit + cog tests with mocked DI container)

## Patterns (from SerenityOps + free-intelligence/aurity.io)
- Settings singleton with Pydantic BaseSettings (env_file=".env")
- Structured logging via structlog (JSON-ready, never use print())
- Memory is append-only: never delete, only grow ("infinite conversation")
- Context is hierarchical: recent messages + keyword-relevant messages
- All DB operations go through _ensure_connection() for auto-reconnect
- Commands must have @commands.cooldown to prevent token burn

## Template Origin
- Created from github.com/BernardUriza/python-bot (GitHub Template)
- Template is platform-agnostic (no Discord deps)
- Discord-specific code lives only in bot.py

## Dependencies
- discord.py >= 2.3.0
- anthropic >= 0.42.0
- aiosqlite >= 0.19.0
- pydantic-settings >= 2.1.0
- structlog >= 24.1.0

## Security
- .env is gitignored — NEVER commit tokens
- Bot validates required tokens at startup (validate_required)
- All user input is parameterized in SQL (no injection)
- Max message length enforced before sending to LLM (4000 chars)
