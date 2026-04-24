"""World scans — Insult's internal notes on what's happening out there.

Populated by a background task that summarizes news/topics into a findings
+ commentary pair. Read at turn-build time to give Insult cross-topic
awareness without requiring a web_search call on every message."""

from __future__ import annotations

import time

import aiosqlite
import structlog

from insult.core.memory.base import BaseRepository

log = structlog.get_logger()


class WorldScansRepository(BaseRepository):
    """Owns the `world_scans` table."""

    async def store_world_scan(self, topic: str, findings: str, commentary: str) -> None:
        """Append a scan. Failures logged-not-raised — a dropped scan is recoverable."""
        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO world_scans (topic, findings, commentary, timestamp) VALUES (?, ?, ?, ?)",
                (topic, findings, commentary, time.time()),
            )
            await db.commit()
            log.info("world_scan_stored", topic=topic[:80])
        except aiosqlite.Error as e:
            log.error("world_scan_store_failed", error=str(e))

    async def get_recent_world_scans(self, limit: int = 5) -> list[dict]:
        """Most recent scans for prompt injection."""
        db = await self._conn()
        cursor = await db.execute(
            "SELECT topic, findings, commentary, timestamp FROM world_scans ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [{"topic": r[0], "findings": r[1], "commentary": r[2], "timestamp": r[3]} for r in rows]
