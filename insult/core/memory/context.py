"""Pure functions for rendering conversation context into LLM-ready blocks.

These belong in their own module because they have zero DB dependency — the
repositories hand them `list[dict]` rows and they produce `list[dict]`
message blocks. Extracted from the monolithic MemoryStore so that prompt-
building code can import just the formatter without dragging in SQLite.
"""

from __future__ import annotations

import time


def format_relative_time(timestamp: float) -> str:
    """Convert a Unix timestamp into a human-readable relative-time phrase.

    Spanish-first because the persona speaks Spanish. The boundaries
    ("justo ahora", "hace 2h", "ayer", "hace 3 días") are deliberately
    chosen so the LLM can feel time passing without needing exact clocks —
    which are deprioritized by the persona rules ("don't announce timestamps").
    """
    now = time.time()
    diff = now - timestamp

    if diff < 60:
        return "justo ahora"
    if diff < 3600:
        mins = int(diff / 60)
        return f"hace {mins}min"
    if diff < 86400:
        hours = int(diff / 3600)
        return f"hace {hours}h"
    days = int(diff / 86400)
    if days == 1:
        return "ayer"
    if days < 7:
        return f"hace {days} días"
    if days < 30:
        weeks = int(days / 7)
        return f"hace {weeks} sem"
    return f"hace {int(days / 30)} meses"


def build_context(recent: list[dict], relevant: list[dict] | None = None) -> list[dict]:
    """Assemble the LLM context: recent turns + relevant retrievals.

    Output is the list the Claude messages API expects (`role` + `content`).
    The "relevant" block is prepended as ONE synthetic user message with a
    header marker so the model knows these are older excerpts, not the live
    thread. Both recent and relevant lines are prefixed with a relative
    timestamp + speaker label so the model keeps track of authorship and
    recency without needing a separate metadata field.
    """
    context: list[dict] = []

    if relevant:
        seen_contents = {m["content"] for m in recent}
        unique_relevant = [m for m in relevant if m["content"] not in seen_contents]
        if unique_relevant:
            context.append(
                {
                    "role": "user",
                    "content": "[Contexto relevante de conversaciones anteriores]\n"
                    + "\n".join(
                        f"[{format_relative_time(m['timestamp'])}] {m['user_name']}: {m['content']}"
                        for m in unique_relevant
                    ),
                }
            )

    for msg in recent:
        ts = f"[{format_relative_time(msg['timestamp'])}] "
        # Both user and assistant messages get name prefix for clear speaker
        # attribution — downstream character.strip_metadata() removes these
        # before the text reaches Discord.
        content = f"{ts}{msg['user_name']}: {msg['content']}"
        context.append({"role": msg["role"], "content": content})

    return context
