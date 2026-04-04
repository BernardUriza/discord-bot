"""Extract and manage persistent user facts via LLM.

After each conversation exchange, we ask the LLM to extract/update
interesting facts about the user. Facts are stored in SQLite and
injected into the system prompt so Insult always knows who it's
talking to — even across sessions.
"""

import json

import anthropic
import structlog

log = structlog.get_logger()

EXTRACTION_PROMPT = """\
You are a fact extractor. Given a conversation in a group chat involving "Insult" (a chatbot) \
and multiple users, extract interesting, persistent facts about the TARGET USER.

Focus on facts a friend would remember:
- Name, nickname, or how they want to be called
- Profession, job, what they study
- Location, country, city
- Interests, hobbies, passions
- Technical skills, programming languages, tools they use
- Personal traits, health conditions, important life events
- Relationships mentioned (friends, family, pets)
- Preferences, opinions, recurring topics
- Languages they speak

Rules:
- Only extract facts about the TARGET USER, never about Insult or other users
- CRITICAL: Preserve WHO said/did WHAT. If Alex told Insult something, that's Alex→Insult, NOT Alex→Bernard
- Each fact must be a short, standalone sentence (max 15 words)
- Use the language the user predominantly speaks (Spanish or English)
- If the user corrects a previous fact, use the corrected version
- Ignore greetings, small talk, and filler with no factual content
- If no new facts are found, return the existing facts unchanged
- Messages are prefixed with speaker names (e.g. "bernard2389: text"). Use these to track who said what

You will receive:
1. The target user's display name
2. Their existing facts (may be empty)
3. Recent conversation messages (with speaker labels)

Return a JSON array of objects with "fact" and "category" keys.
Categories: identity, profession, location, interests, technical, personal, preferences

Example output:
[
  {"fact": "Se llama Bernard, le dicen Bern", "category": "identity"},
  {"fact": "Es programador, trabaja con Python y Rust", "category": "profession"},
  {"fact": "Vive en México", "category": "location"},
  {"fact": "Le interesa la inteligencia artificial", "category": "interests"}
]

Return ONLY the JSON array, no other text."""


async def extract_facts(
    client: anthropic.AsyncAnthropic,
    model: str,
    user_name: str,
    existing_facts: list[dict],
    recent_messages: list[dict],
) -> list[dict]:
    """Extract/update user facts from recent conversation using LLM.

    Returns a list of fact dicts with 'fact' and 'category' keys.
    """
    existing_str = "\n".join(f"- [{f['category']}] {f['fact']}" for f in existing_facts) if existing_facts else "(none)"

    conversation_str = "\n".join(f"{m.get('user_name', 'Insult')}: {m['content']}" for m in recent_messages[-10:])

    user_prompt = (
        f"User display name: {user_name}\n\nExisting facts:\n{existing_str}\n\nRecent conversation:\n{conversation_str}"
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        # Parse JSON — handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        facts = json.loads(raw)
        if not isinstance(facts, list):
            log.warning("facts_extraction_bad_format", raw=raw[:200])
            return existing_facts

        valid = [
            {"fact": f["fact"], "category": f.get("category", "general")}
            for f in facts
            if isinstance(f, dict) and "fact" in f
        ]
        log.info("facts_extracted", user_name=user_name, count=len(valid))
        return valid

    except (json.JSONDecodeError, anthropic.APIError, KeyError, IndexError) as e:
        log.warning("facts_extraction_failed", error=str(e), user_name=user_name)
        return existing_facts


def build_facts_prompt(user_name: str, facts: list[dict]) -> str:
    """Build a system prompt section with the user's known facts.

    When called with semantically-searched facts (via search_facts_semantic),
    the facts are already filtered to the most relevant ones for the current
    message. When called with all facts (fallback), all facts are included.

    Callers should use semantic search when len(facts) > 5 to avoid bloating
    the system prompt with irrelevant facts.
    """
    if not facts:
        return ""

    lines = [f"- [{f['category']}] {f['fact']}" for f in facts]
    return f"\n\n## What you know about {user_name} (use naturally, don't list these)\n" + "\n".join(lines)
