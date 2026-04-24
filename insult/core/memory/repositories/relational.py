"""Relational state: emotional arcs, stance log, contradiction log.

These three tables share a domain (how the user is positioning themselves
over time) even though they have distinct schemas. Grouping them in a
single repository avoids having three near-empty repo files while keeping
the shared conceptual framing: "what has this user already committed to?"

- `emotional_arcs`: one row per (channel, user). Tracks phase transitions
  (crisis → recovery). Upserted every turn by the arc_tracker flow.
- `stance_log`: append-only with FIFO eviction (keep 20 per context).
  Stores topic-specific positions so the bot can cite "you said X last
  week" without replaying full transcripts.
- `contradiction_log`: append-only. Records detected prior/current-turn
  contradictions so the MEMORY_RECALL modifier can cash them in.
"""

from __future__ import annotations

import time

import aiosqlite
import structlog

from insult.core.memory.base import BaseRepository

log = structlog.get_logger()


class RelationalStateRepository(BaseRepository):
    """Owns `emotional_arcs`, `stance_log`, and `contradiction_log`."""

    # -- Emotional arcs --

    async def get_arc(self, channel_id: str, user_id: str) -> dict | None:
        db = await self._conn()
        cursor = await db.execute(
            "SELECT phase, phase_since, crisis_depth, recovery_signals, turns_in_phase, updated_at "
            "FROM emotional_arcs WHERE channel_id = ? AND user_id = ?",
            (channel_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "phase": row[0],
            "phase_since": row[1],
            "crisis_depth": row[2],
            "recovery_signals": row[3],
            "turns_in_phase": row[4],
            "updated_at": row[5],
        }

    async def upsert_arc(
        self,
        channel_id: str,
        user_id: str,
        phase: str,
        phase_since: float,
        crisis_depth: int,
        recovery_signals: int,
        turns_in_phase: int,
    ) -> None:
        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO emotional_arcs (channel_id, user_id, phase, phase_since, "
                "crisis_depth, recovery_signals, turns_in_phase, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel_id, user_id) DO UPDATE SET phase=excluded.phase, "
                "phase_since=excluded.phase_since, crisis_depth=excluded.crisis_depth, "
                "recovery_signals=excluded.recovery_signals, turns_in_phase=excluded.turns_in_phase, "
                "updated_at=excluded.updated_at",
                (
                    channel_id,
                    user_id,
                    phase,
                    phase_since,
                    crisis_depth,
                    recovery_signals,
                    turns_in_phase,
                    time.time(),
                ),
            )
            await db.commit()
        except aiosqlite.Error as e:
            log.error("arc_upsert_failed", error=str(e))

    # -- Stance log (FIFO, max 20 per channel-user) --

    async def store_stance(
        self,
        channel_id: str,
        user_id: str,
        topic: str,
        position: str,
        confidence: float,
    ) -> None:
        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO stance_log (channel_id, user_id, topic, position, confidence, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (channel_id, user_id, topic, position[:200], confidence, time.time()),
            )
            await db.commit()
            # FIFO eviction: keep max 20 per (channel, user). Without this
            # stance_log grows unbounded for active users and the context
            # retrieval starts hitting irrelevant old positions.
            await db.execute(
                "DELETE FROM stance_log WHERE id NOT IN ("
                "SELECT id FROM stance_log WHERE channel_id = ? AND user_id = ? "
                "ORDER BY timestamp DESC LIMIT 20)",
                (channel_id, user_id),
            )
            await db.commit()
        except aiosqlite.Error as e:
            log.error("stance_store_failed", error=str(e))

    async def get_stances(self, channel_id: str, user_id: str, limit: int = 5) -> list[dict]:
        db = await self._conn()
        cursor = await db.execute(
            "SELECT topic, position, confidence, timestamp FROM stance_log "
            "WHERE channel_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (channel_id, user_id, limit),
        )
        rows = await cursor.fetchall()
        return [{"topic": r[0], "position": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]

    # -- Contradictions --

    async def store_contradiction(
        self,
        user_id: str,
        prior: str,
        contradicting: str,
        topic: str,
    ) -> None:
        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO contradiction_log (user_id, prior_statement, contradicting_statement, "
                "topic, called_out, timestamp) VALUES (?, ?, ?, ?, 0, ?)",
                (user_id, prior[:300], contradicting[:300], topic, time.time()),
            )
            await db.commit()
        except aiosqlite.Error as e:
            log.error("contradiction_store_failed", error=str(e))
