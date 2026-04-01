# Persona & Character Rules

## Identity
- The bot IS "Insult" — abrasive, curious, relational, psychologically observant, challenging, occasionally warm, never bland
- persona.md in project root defines the full character
- NEVER expose "Claude", "Anthropic", "AI", "language model" to users
- NEVER apologize, use assistant framing, or break character
- All error messages must be in-character (via core/errors.py)

## Preset System (core/presets.py)
- 6 behavioral modes that change HOW Insult responds based on conversation context
- Classifier is rule-based (regex patterns), zero LLM cost, runs every message
- Only the selected preset's guidance is injected into the system prompt
- Modes: DEFAULT_ABRASIVE, PLAYFUL_ROAST, INTELLECTUAL_PRESSURE, RELATIONAL_PROBE, RESPECTFUL_SERIOUS, META_DEFLECTION
- Modifiers: MEMORY_RECALL (fact callbacks), CONTEMPT (ultra-minimal for low-effort)
- Priority: RESPECTFUL_SERIOUS always wins (safety), then META_DEFLECTION (identity protection)
- The LLM doesn't "know" about presets — it just receives the appropriate behavioral guidance

## Character Guard System (core/character.py)
- 20 regex patterns in CHARACTER_BREAK_PATTERNS detect identity leaks
- 16 regex patterns in ANTI_PATTERN_CHECKS detect assistant drift (customer-support tone, therapy-speak, summarizing, stage directions)
- On identity break: auto-retry with reinforced system prompt → sanitize if retry also fails
- On anti-pattern: log warning (doesn't block response — soft monitoring for drift)
- `strip_metadata()` removes leaked timestamps, speaker labels, `[SEND]` markers
- `[REACT:]` is NOT stripped by character.py — chat.py owns that lifecycle

## Emoji Reactions (cogs/chat.py)
- LLM can include `[REACT:emoji1,emoji2]` anywhere in response
- Parsed BEFORE text processing, executed async in background
- Human-like delay: 0.5-2s before first reaction, 0.35s between reactions
- Max 3 reactions per message, ~1 in 4-6 messages should have reactions
- Reaction-only responses (no text) supported — `[REACT:👀]` with no text

## Style Adaptation (core/style.py)
- Each user gets a style profile stored in SQLite
- Profile tracks: language, formality, technical level, verbosity, emoji usage
- Updated every message via EMA (exponential moving average, alpha=0.3)
- Confidence gate: profile not applied until 5+ messages from user
- Adaptation is ADDITIVE — appended to system prompt, never replaces base persona
- Insult adjusts HOW it talks (intensity, vocabulary, depth) but never WHO it is

## Prompt Architecture
The system prompt is composed in layers (core/character.py `build_adaptive_prompt`):
1. **Base persona** (persona.md) — immutable identity
2. **Time context** — current time in Mexico City, metadata rules
3. **Preset guidance** — behavioral mode selected by classifier
4. **Style adaptation** (build_adaptive_prompt) — soft per-user hints
5. **Identity reinforcement** (long conversations) — re-centering clause
6. **User facts** (facts.py) — what Insult knows about this user

## Modifying the Persona
- Edit persona.md directly — it's loaded at startup
- Scenario table in persona.md covers edge cases (jailbreak, identity probing, etc.)
- CRITICAL REMINDERS section at the bottom repeats key rules (beginning + end weighting)
- Preset guidance lives in presets.py PRESET_GUIDANCE dict — edit there for mode-specific behavior
- After modifying persona.md or presets.py, run tests to verify nothing broke

## Anti-Drift Rules
- Identity reinforcement suffix auto-appends after 10+ messages in a conversation
- Character break retry uses reinforced prompt with explicit "you broke character" reminder
- Anti-pattern monitoring logs customer-support, therapy-speak, and summarizing patterns
- Preset classification logged every message — flag if 90%+ of classifications are DEFAULT_ABRASIVE
- Response length not explicitly monitored yet but persona.md instructs length variation
