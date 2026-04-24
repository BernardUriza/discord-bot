"""Conversation-context and user-facts loading for a single turn.

Pure async helpers. No state of their own — everything comes from the
`memory` store and `settings` passed in. `cog.py` owns the MemoryStore and
hands it to `turn.run_turn` which forwards it here.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()


async def build_context(
    memory,
    settings,
    channel_id: str,
    text: str,
    attachment_blocks: list,
) -> tuple[list[dict] | None, list[dict]]:
    """Return (context, recent) or (None, []) on memory failure.

    `recent` is the raw last-N messages used for flow analysis + shape
    selection. `context` is the merged recent+relevant list formatted as
    Anthropic MessageParam dicts, with the user's current attachments
    spliced into the last user message when present.
    """
    try:
        recent = await memory.get_recent(channel_id, settings.memory_recent_limit)
        relevant = await memory.search(channel_id, text, settings.memory_relevant_limit)
        context = memory.build_context(recent, relevant)
    except Exception:
        log.exception("chat_context_failed", channel_id=channel_id)
        return None, []

    if attachment_blocks and context:
        last_msg = context[-1]
        if last_msg["role"] == "user":
            text_block = {"type": "text", "text": last_msg["content"]}
            context[-1] = {"role": "user", "content": [text_block, *attachment_blocks]}

    return context, recent


async def load_facts(memory, user_id: str) -> list[dict]:
    """Return all user facts, or [] on failure."""
    try:
        return await memory.get_facts(user_id)
    except Exception:
        log.exception("chat_facts_load_failed", user_id=user_id)
        return []


async def load_facts_smart(memory, user_id: str, query: str) -> list[dict]:
    """Like `load_facts` but uses semantic search when the user has 6+ facts.

    Below 6 facts the full list is cheap to inject; above that we filter
    by relevance to the current query to keep the system prompt lean and
    improve preset/flow routing signal.
    """
    try:
        all_facts = await memory.get_facts(user_id)
    except Exception:
        log.exception("chat_facts_load_failed", user_id=user_id)
        return []

    if len(all_facts) > 5:
        try:
            return await memory.search_facts_semantic(user_id, query, limit=10)
        except Exception:
            log.exception("chat_facts_semantic_failed", user_id=user_id)
            return all_facts

    return all_facts
