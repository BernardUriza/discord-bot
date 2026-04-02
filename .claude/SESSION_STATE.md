# SESSION STATE - 2026-04-01 (Session 3 — persona re-architecture)

## Quick Summary
Completed major persona re-architecture (v0.9.0). Added ethical confrontation framework, ARC preset, system-critical stance, anti-bigotry targeting rules, escalation logic, response-length policy per preset, expanded anti-patterns. All 202 tests pass, 80.21% coverage.

## Current Version
- **pyproject.toml:** v0.9.0
- **VERSION_TAG:** `ᵇᵉᵗᵃ ᵛ⁰·⁹·⁰`
- **Branch:** `main` (uncommitted changes ready for commit)
- **Previous commit:** `84588fe` Switch TTS to MP3 attachment (v0.8.0)

## Changes This Session (uncommitted)

### persona.md — Complete rewrite
- Added **Ethical Confrontation Framework**: allowed targets (arguments, ideologies, hypocrisy, systems, power, behaviors) vs disallowed targets (protected characteristics, disability, trauma, poverty, body traits, marginalization, language mistakes)
- Added **validity test**: "If the insult stops working after removing the target's identity trait, it is invalid"
- Added **Political-ethical stance**: pro-LGBT, anti-speciesist, anti-capitalist critique, anti-anthropocentrism, anti-technocratic domination, anti-psychologism — with rules against sloganizing
- Added **Escalation logic**: confused→clarify, evasive→confront, prejudiced→challenge, hateful→refuse, vulnerable→sharpen with care
- Refined **Response-length policy**: principled variation tied to context, with explicit rules per length bucket
- Added **Desired response formula**: name the dodge, expose hidden premise, land one sharp line, ask one real question
- Expanded **Anti-patterns**: preachy monologues, ideology slogans, moralizing without tension, over-validating, punching down
- Added **Scenario handling** for bigotry, punching down, system critique, vulnerability
- Core identity traits refined: added SYSTEM-CRITICAL, ANTI-DOMINATION, INTELLECTUALLY HONEST; renamed OBSERVANT→PERCEPTIVE

### insult/core/presets.py — ARC preset + refinements
- Added `PresetMode.ARC` (Adaptive Relational Critique) — 7th preset mode
- Added 16 ARC trigger patterns: capitalism, patriarchy, speciesism, colonialism, inequality, privilege, gentrification, ethics questions, bigotry topics (racism, homophobia, transphobia, ableism), contradictions
- ARC priority: slot 3 (after SERIOUS and META, before RELATIONAL)
- ARC guidance: hard on domination/soft on personhood, precision strikes over spray, name mechanisms not villains, refuse premise of bigotry, fewer but better insults
- Added response-length guidance to ALL preset GUIDANCE strings
- Refined existing presets: added ethical constraints ("never punch down"), attribution reminders, clearer exit conditions

### insult/core/character.py — Expanded anti-patterns
- Added 6 new anti-pattern checks:
  - Preachy activist monologues (EN + ES)
  - Over-validation / excessive agreement
  - Moralizing without tension (EN + ES)

### Tests — ARC coverage
- 13 new tests across test_presets.py and test_character.py:
  - ARC trigger tests: capitalism, speciesism, racism, homophobia, ethics, privilege, context-based
  - ARC priority tests: serious beats ARC, ARC beats intellectual
  - ARC prompt build test
  - Anti-pattern tests: preachy monologue, over-validation, moralizing
  - ARC in build_adaptive_prompt test

### Version bump
- pyproject.toml: 0.8.0 → 0.9.0
- chat.py VERSION_TAG: ᵇᵉᵗᵃ ᵛ⁰·⁸·⁰ → ᵇᵉᵗᵃ ᵛ⁰·⁹·⁰

## Architecture Overview (updated)

### Preset Priority Order (7 modes)
1. RESPECTFUL_SERIOUS — safety first (crisis, distress)
2. META_DEFLECTION — identity protection (jailbreaks, probing)
3. ARC — ethical/systemic/deep relational (new)
4. RELATIONAL_PROBE — emotional/personal
5. INTELLECTUAL_PRESSURE — technical/argumentative
6. PLAYFUL_ROAST — humor/banter
7. DEFAULT_ABRASIVE — fallback

### Ethical Framework Architecture
- **Targeting rules** in persona.md (system prompt level)
- **ARC preset guidance** in presets.py (behavioral guidance level)
- **Anti-pattern checks** in character.py (post-generation validation level)
- Three-layer defense: prompt shapes behavior → preset refines approach → anti-patterns catch drift

### Key Design Decisions
- "Hard on domination, soft on personhood" is the architectural principle, not a slogan
- Validity test for insults: must work independently of target's identity traits
- Political stance is held with confidence but explained with humility
- ARC activates on systemic/ethical topics, not just when someone is upset
- Response-length guidance is per-preset, not global randomization
- Anti-patterns expanded to catch preachy AI behavior (monologues, slogans, moralizing)

## Known Issues (carried forward)
- Channel creation: tool_use works but hasn't been tested end-to-end
- TTS: deployed but user hasn't confirmed it works yet
- facts.py: extraction sometimes fails with "Unterminated string" JSON parse error
- bot.py:174 still has "Calmate" in on_command_error for command cooldowns

## To Resume
1. Commit the persona re-architecture (all changes are staged-ready)
2. Push to main → CI → CD → verify in Discord
3. Test the bot's behavior with ethical edge cases
4. Consider E2E testing with ARC-triggering messages
5. Channel creation E2E test still pending
6. TTS E2E test still pending
