"""Channel summaries — cross-channel awareness via pre-aggregated digests.

Populated by a background summarization task. Read during prompt building
to give Insult a "server pulse" (what's happening in other channels)
without shipping every channel's transcript into context."""

from __future__ import annotations

import time

import aiosqlite
import structlog

from insult.core.memory.base import BaseRepository

log = structlog.get_logger()


class ChannelSummariesRepository(BaseRepository):
    """Owns the `channel_summaries` table."""

    async def upsert_channel_summary(
        self,
        guild_id: str,
        channel_id: str,
        channel_name: str,
        summary: str,
        message_count: int,
        last_message_ts: float,
        is_private: bool = False,
    ) -> None:
        """Insert or overwrite the summary for a (guild, channel) pair."""
        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO channel_summaries (guild_id, channel_id, channel_name, summary, "
                "message_count, last_message_ts, is_private, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(guild_id, channel_id) DO UPDATE SET "
                "channel_name=excluded.channel_name, summary=excluded.summary, "
                "message_count=excluded.message_count, last_message_ts=excluded.last_message_ts, "
                "is_private=excluded.is_private, updated_at=excluded.updated_at",
                (
                    guild_id,
                    channel_id,
                    channel_name,
                    summary,
                    message_count,
                    last_message_ts,
                    int(is_private),
                    time.time(),
                ),
            )
            await db.commit()
            log.info(
                "channel_summary_upserted",
                guild_id=guild_id,
                channel_id=channel_id,
                channel_name=channel_name,
            )
        except aiosqlite.Error as e:
            log.error("channel_summary_upsert_failed", channel_id=channel_id, error=str(e))

    async def get_channel_summaries(
        self,
        guild_id: str,
        exclude_channel_id: str | None = None,
        limit: int = 8,
    ) -> list[dict]:
        """Summaries for the server-pulse prompt block.

        `exclude_channel_id` is typically the channel Insult is currently
        replying in — we don't want to echo that channel's own summary back
        at it as "cross-channel awareness"."""
        db = await self._conn()
        if exclude_channel_id:
            cursor = await db.execute(
                "SELECT channel_id, channel_name, summary, message_count, last_message_ts, "
                "is_private, updated_at FROM channel_summaries "
                "WHERE guild_id = ? AND channel_id != ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (guild_id, exclude_channel_id, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT channel_id, channel_name, summary, message_count, last_message_ts, "
                "is_private, updated_at FROM channel_summaries "
                "WHERE guild_id = ? ORDER BY updated_at DESC LIMIT ?",
                (guild_id, limit),
            )
        rows = await cursor.fetchall()
        return [
            {
                "channel_id": r[0],
                "channel_name": r[1],
                "summary": r[2],
                "message_count": r[3],
                "last_message_ts": r[4],
                "is_private": bool(r[5]),
                "updated_at": r[6],
            }
            for r in rows
        ]
