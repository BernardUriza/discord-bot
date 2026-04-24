"""Disclosure log — severity-tagged record of user self-disclosures.

Populated by the `core/disclosure` scanner whenever a user reveals
personal or sensitive information (mental health, relationships, etc.).
Used for telemetry and to trigger downstream escalations; the persona
does NOT cite these entries directly.
"""

from __future__ import annotations

import time

import aiosqlite
import structlog

from insult.core.memory.base import BaseRepository

log = structlog.get_logger()


class DisclosureRepository(BaseRepository):
    """Owns the `disclosure_log` table."""

    async def store_disclosure(
        self,
        channel_id: str,
        user_id: str,
        category: str,
        severity: int,
        signals: str,
        excerpt: str,
    ) -> None:
        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO disclosure_log (channel_id, user_id, category, severity, signals, "
                "message_excerpt, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (channel_id, user_id, category, severity, signals, excerpt[:200], time.time()),
            )
            await db.commit()
        except aiosqlite.Error as e:
            log.error("disclosure_store_failed", error=str(e))
