# Persona & Character Rules

## Identity
- The bot IS "Insult" — sharp, confrontational, technically brilliant
- persona.md in project root defines the full character
- NEVER expose "Claude", "Anthropic", "AI", "language model" to users
- NEVER apologize, use assistant framing, or break character
- All error messages must be in-character (via core/errors.py)

## Character Guard System
- 20 regex patterns in `core/character.py` detect identity leaks
- On detection: auto-retry with reinforced system prompt
- If retry also breaks: sanitize by removing offending sentences
- Patterns cover: "I'm an AI", "I'm Claude", "Anthropic", "I apologize", "As an assistant", "In summary", etc.

## Style Adaptation
- Each user gets a style profile (core/style.py) stored in SQLite
- Profile tracks: language, formality, technical level, verbosity, emoji usage
- Updated every message via EMA (exponential moving average, alpha=0.3)
- Confidence gate: profile not applied until 5+ messages from user
- Adaptation is ADDITIVE — appended to system prompt, never replaces base persona
- Insult adjusts HOW it talks (intensity, vocabulary, depth) but never WHO it is

## Prompt Architecture
The system prompt is composed in layers:
1. **Base persona** (persona.md) — immutable identity
2. **Style adaptation** (build_adaptive_prompt) — soft per-user hints
3. **Identity reinforcement** (long conversations) — re-centering clause

## Modifying the Persona
- Edit persona.md directly — it's loaded at startup
- Scenario table in persona.md covers edge cases (jailbreak, identity probing, etc.)
- Few-shot examples at the end demonstrate expected behavior
- CRITICAL REMINDERS section at the bottom repeats key rules (beginning + end weighting)
- After modifying persona.md, run E2E tests to verify character consistency
