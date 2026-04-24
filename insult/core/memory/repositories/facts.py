"""User facts — structured long-term memory about each user.

Two provenance tiers coexist in the same table, distinguished by `source`:
- `'auto'`: produced by the LLM fact-extraction background task. Wiped on
  every re-extraction (the extractor sees the last N messages and outputs
  a full refreshed list).
- `'manual'`: curated by a human operator or cross-user injection. NEVER
  wiped. This distinction exists because before the `source` column was
  introduced, every re-extraction nuked manual injections — the bot would
  forget anything a teammate contributed within one turn.
"""

from __future__ import annotations

import time

import aiosqlite
import structlog

from insult.core.memory.base import BaseRepository

log = structlog.get_logger()


class FactsRepository(BaseRepository):
    """Owns the `user_facts` table. Semantic search integrates `core/vectors`."""

    async def get_facts(self, user_id: str) -> list[dict]:
        """All facts for a user, newest-updated first."""
        db = await self._conn()
        cursor = await db.execute(
            "SELECT id, fact, category, updated_at FROM user_facts WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [{"id": r[0], "fact": r[1], "category": r[2], "updated_at": r[3]} for r in rows]

    async def get_all_facts(self) -> list[dict]:
        """Every fact for every user — used by cross-user prompt injection."""
        db = await self._conn()
        cursor = await db.execute(
            "SELECT user_id, id, fact, category, updated_at FROM user_facts ORDER BY user_id, updated_at DESC",
        )
        rows = await cursor.fetchall()
        return [
            {
                "user_id": r[0],
                "id": r[1],
                "fact": r[2],
                "category": r[3],
                "updated_at": r[4],
            }
            for r in rows
        ]

    async def save_facts(self, user_id: str, facts: list[dict]) -> None:
        """Replace AUTO-extracted facts for a user with a new snapshot.

        Rows with source='manual' are PRESERVED. This is the critical
        invariant: before it existed, any curated fact died on the next
        conversation turn because the LLM extractor only sees ~10 messages
        and routinely omits older facts from its output. The DELETE below
        is scoped to source='auto' precisely to protect manual entries."""
        db = await self._conn()
        now = time.time()
        try:
            await db.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND source = 'auto'",
                (user_id,),
            )
            for f in facts:
                await db.execute(
                    "INSERT INTO user_facts (user_id, fact, category, updated_at, source) VALUES (?, ?, ?, ?, 'auto')",
                    (user_id, f["fact"], f.get("category", "general"), now),
                )
            await db.commit()
            log.info("facts_saved", user_id=user_id, count=len(facts))

            # Update vector embeddings if available. Swallow per-call errors —
            # a failed vector upsert must not break the primary SQL commit.
            if self.vectors_available:
                try:
                    from insult.core.vectors import upsert_fact_vectors

                    await upsert_fact_vectors(db, user_id, facts)
                except Exception as ve:
                    log.warning("facts_vector_upsert_failed", user_id=user_id, error=str(ve))
        except aiosqlite.Error as e:
            log.error("facts_save_failed", user_id=user_id, error=str(e))

    async def add_manual_fact(
        self,
        user_id: str,
        fact: str,
        category: str = "general",
    ) -> int:
        """Insert a curated fact marked source='manual' so extract_facts can't wipe it.

        Returns the inserted row id. This is an append, not an upsert —
        callers that care about dedup must do their own check."""
        db = await self._conn()
        now = time.time()
        cursor = await db.execute(
            "INSERT INTO user_facts (user_id, fact, category, updated_at, source) VALUES (?, ?, ?, ?, 'manual')",
            (user_id, fact, category, now),
        )
        await db.commit()
        row_id = cursor.lastrowid
        log.info("manual_fact_added", user_id=user_id, fact_id=row_id, category=category)
        return row_id or 0

    async def search_facts_semantic(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        """Hybrid vector + FTS search for relevance-ranked facts.

        Falls back to `get_facts()` (unranked) when vectors are unavailable
        so callers can treat this as a single entry point regardless of
        whether sqlite-vec loaded at boot."""
        if not self.vectors_available:
            return await self.get_facts(user_id)

        db = await self._conn()
        try:
            from insult.core.vectors import search_facts_hybrid

            results = await search_facts_hybrid(db, user_id, query, limit=limit)
            if results:
                log.info(
                    "facts_semantic_search",
                    user_id=user_id,
                    query=query[:50],
                    results=len(results),
                )
                return results
            return await self.get_facts(user_id)
        except Exception as e:
            log.warning("facts_semantic_search_failed", user_id=user_id, error=str(e))
            return await self.get_facts(user_id)
