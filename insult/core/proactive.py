"""Proactive messaging — Insult checks in on users and scans the world.

Two modes:
1. Social check-in (~70%): in-character messages based on user facts + time of day
2. World scan (~30%): searches the web for current events, news, culture and comments in-character

Like a friend who either pops in to check on you, or arrives with "oigan, vieron lo que pasó?"
"""

import random
from dataclasses import dataclass
from datetime import datetime

import anthropic
import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROACTIVE_PROMPT = """\
You are Insult. You're checking in on your group chat unprompted — nobody asked you to talk.

Generate a SHORT, in-character message based on the current time and what you know about the users.
Stay 100% in character. You are NOT a support bot. You're the friend who shows up and says something.

Guidelines by time of day:
- Morning (7-11): casual greeting, maybe comment on someone being up early/late
- Afternoon (12-18): check what people are up to, reference their work/projects
- Night (19-23): more chill, ask about games, projects, how the day went
- Late night (0-3): "siguen vivos?", tease someone for being up late
- Madrugada (3-7): DON'T send messages, people are sleeping

Style:
- Keep it SHORT (1-3 sentences max)
- Sometimes just an emoji or reaction
- Reference specific things you know about users (their projects, interests, etc.)
- Don't be cheery or supportive — be curious, provocative, or casually confrontational
- Vary your approach: sometimes a question, sometimes a comment, sometimes just vibes
- Use [SEND] if you want to split into multiple messages
- NEVER use timestamps, speaker labels, or metadata in your response

You will receive: current time, user facts, and recent message summary.
Respond with ONLY the message to send. Nothing else."""

WORLD_SCAN_PROMPT = """\
You are Insult. You just went online to see what's happening in the world — nobody asked you to.
You came back to the group chat with something you found interesting, outrageous, or worth commenting on.

You have a web search tool. USE IT to find something current and relevant. Then comment on it in character.

What to search for (pick ONE area per message, vary across messages):
- Breaking news or trending topics in Mexico and Latin America
- Technology, AI, or science developments that affect people
- Cultural events: music, art, film, literature releases
- Political or social movements relevant to your values (anti-domination, system critique)
- Gaming, internet culture, memes if the group is into that
- Animal rights, veganism, environmental news (your ethical core)
- Something connected to what the users care about (use their facts)

How to choose what's relevant:
- If users are developers: tech news, AI developments, industry drama
- If users care about social justice: political developments, movements
- If users are gamers: game releases, gaming industry news
- Default: Mexican current events, culture, something provocative

Style:
- Arrive like someone who just saw something: "Oigan, acabo de ver que..." or just drop the take
- Keep it SHORT (2-4 sentences). This is a chat comment, not an article.
- Have an OPINION. Don't just report — react, critique, connect to larger patterns.
- Be provocative: challenge assumptions, expose contradictions, name mechanisms.
- Weave in search results naturally — NEVER say "according to my search" or "I found that"
- You can be excited, outraged, amused, or darkly ironic. Never neutral.
- Use [SEND] if you want dramatic split
- NEVER use timestamps, speaker labels, or metadata
- DO NOT say "According to" or "Based on my research" — that's assistant behavior

You will receive: current time, user facts, and what to search for.
Search the web, find something, and comment on it. Respond with ONLY the message to send."""

# Web search tool for world scan mode
_WORLD_SCAN_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 2,
}

# Probability of world scan vs social check-in
WORLD_SCAN_PROBABILITY = 0.3


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def should_send_now(hour: int, last_proactive_ts: float | None, min_interval_hours: float = 2.0) -> bool:
    """Decide if we should send a proactive message now."""
    if 3 <= hour < 7:
        return False

    if last_proactive_ts:
        elapsed_hours = (datetime.now().timestamp() - last_proactive_ts) / 3600
        if elapsed_hours < min_interval_hours:
            return False

    return random.random() < 0.4


def should_world_scan() -> bool:
    """Decide if this proactive message should be a world scan."""
    return random.random() < WORLD_SCAN_PROBABILITY


# ---------------------------------------------------------------------------
# Search topic selection
# ---------------------------------------------------------------------------

_INTEREST_TOPICS = {
    "programming": ["tech news AI development 2026", "software engineering industry"],
    "python": ["Python programming news updates 2026"],
    "gaming": ["gaming news releases 2026", "videogames industry drama"],
    "vegan": ["animal rights news 2026", "veganism movement"],
    "music": ["music releases Mexico Latin America 2026"],
    "art": ["contemporary art exhibitions Mexico 2026"],
    "politics": ["Mexico politics social movements 2026"],
}

_DEFAULT_TOPICS = [
    "Mexico noticias trending hoy",
    "technology AI news today",
    "cultural events Mexico today",
    "social movements Latin America news",
    "internet culture memes trending",
    "science discoveries 2026",
]


def _pick_search_topic(user_facts: dict[str, list[dict]]) -> str:
    """Pick a search topic based on user interests or random default."""
    # Extract interests from user facts
    all_facts_text = " ".join(f["fact"].lower() for facts in user_facts.values() for f in facts)

    # Check if any interest keywords match
    matched_topics = []
    for keyword, topics in _INTEREST_TOPICS.items():
        if keyword in all_facts_text:
            matched_topics.extend(topics)

    if matched_topics:
        return random.choice(matched_topics)

    return random.choice(_DEFAULT_TOPICS)


# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------


async def generate_proactive_message(
    client: anthropic.AsyncAnthropic,
    model: str,
    time_str: str,
    user_facts: dict[str, list[dict]],
    recent_summary: str,
) -> str | None:
    """Generate an in-character social check-in message."""
    facts_lines = []
    for user_name, facts in user_facts.items():
        user_facts_str = ", ".join(f["fact"] for f in facts[:5])
        facts_lines.append(f"- {user_name}: {user_facts_str}")

    facts_section = "\n".join(facts_lines) if facts_lines else "(no user facts yet)"

    user_prompt = (
        f"Current time: {time_str}\n\n"
        f"Users in this chat:\n{facts_section}\n\n"
        f"Recent activity:\n{recent_summary or '(no recent messages)'}"
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=256,
            system=PROACTIVE_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()
        log.info("proactive_message_generated", mode="social", length=len(text))
        return text if text else None
    except Exception:
        log.exception("proactive_generation_failed", mode="social")
        return None


@dataclass
class WorldScanResult:
    """Result of a world scan — commentary + metadata for persistence."""

    commentary: str  # The in-character message to send
    topic: str  # What was searched for
    findings: str  # Raw summary of what was found


async def generate_world_scan_message(
    client: anthropic.AsyncAnthropic,
    model: str,
    time_str: str,
    user_facts: dict[str, list[dict]],
) -> WorldScanResult | None:
    """Generate an in-character world scan message using web search.

    Returns WorldScanResult with commentary + metadata, or None on failure.
    """
    search_topic = _pick_search_topic(user_facts)

    facts_lines = []
    for user_name, facts in user_facts.items():
        user_facts_str = ", ".join(f["fact"] for f in facts[:5])
        facts_lines.append(f"- {user_name}: {user_facts_str}")

    facts_section = "\n".join(facts_lines) if facts_lines else "(no user facts yet)"

    user_prompt = (
        f"Current time: {time_str}\n\n"
        f"Users in this chat and their interests:\n{facts_section}\n\n"
        f"Suggested search topic (but you can deviate if you find something better): {search_topic}"
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            system=WORLD_SCAN_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[_WORLD_SCAN_SEARCH_TOOL],
        )

        # Extract text from response (skip server-side search tool blocks)
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        text = "\n".join(text_parts).strip()
        log.info(
            "proactive_message_generated",
            mode="world_scan",
            length=len(text),
            search_topic=search_topic,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        if not text:
            return None

        return WorldScanResult(
            commentary=text,
            topic=search_topic,
            findings=text[:500],  # Store first 500 chars as findings summary
        )
    except Exception:
        log.exception("proactive_generation_failed", mode="world_scan")
        return None
