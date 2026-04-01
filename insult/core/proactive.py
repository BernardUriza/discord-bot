"""Proactive messaging — Insult checks in on users periodically.

Generates in-character messages to active channels based on time of day,
recent activity, and user facts. Like a friend who pops in to check on you.
"""

import random
from datetime import datetime

import anthropic
import structlog

log = structlog.get_logger()

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


def should_send_now(hour: int, last_proactive_ts: float | None, min_interval_hours: float = 2.0) -> bool:
    """Decide if we should send a proactive message now."""
    # Don't message during deep sleep hours (3-7 AM)
    if 3 <= hour < 7:
        return False

    # Check minimum interval since last proactive message
    if last_proactive_ts:
        elapsed_hours = (datetime.now().timestamp() - last_proactive_ts) / 3600
        if elapsed_hours < min_interval_hours:
            return False

    # Random chance to add unpredictability (~40% chance when eligible)
    return random.random() < 0.4


async def generate_proactive_message(
    client: anthropic.AsyncAnthropic,
    model: str,
    time_str: str,
    user_facts: dict[str, list[dict]],
    recent_summary: str,
) -> str | None:
    """Generate an in-character proactive message using LLM."""
    # Build user facts section
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
        log.info("proactive_message_generated", length=len(text))
        return text if text else None
    except Exception:
        log.exception("proactive_generation_failed")
        return None
