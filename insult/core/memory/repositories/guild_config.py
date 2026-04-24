"""Guild configuration — per-guild system channel IDs.

Populated during the guild-setup flow that creates a dedicated category
with facts + reminders channels. The bot reads this on every message
needing to post an auto-notification (fact extraction, reminder delivery)
to avoid spamming the conversation channel."""

from __future__ import annotations

import aiosqlite
import structlog

from insult.core.memory.base import BaseRepository

log = structlog.get_logger()


class GuildConfigRepository(BaseRepository):
    """Owns the `guild_config` table."""

    async def get_guild_config(self, guild_id: str) -> dict | None:
        """Return config dict or None if setup hasn't run for this guild."""
        db = await self._conn()
        cursor = await db.execute(
            "SELECT guild_id, category_id, facts_channel_id, reminders_channel_id, setup_complete "
            "FROM guild_config WHERE guild_id = ?",
            (guild_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "guild_id": row[0],
            "category_id": row[1],
            "facts_channel_id": row[2],
            "reminders_channel_id": row[3],
            "setup_complete": bool(row[4]),
        }

    async def save_guild_config(
        self,
        guild_id: str,
        category_id: str,
        facts_channel_id: str,
        reminders_channel_id: str,
    ) -> None:
        """Insert or overwrite config for a guild. `setup_complete` is forced to 1."""
        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO guild_config (guild_id, category_id, facts_channel_id, "
                "reminders_channel_id, setup_complete) VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(guild_id) DO UPDATE SET category_id=excluded.category_id, "
                "facts_channel_id=excluded.facts_channel_id, "
                "reminders_channel_id=excluded.reminders_channel_id, setup_complete=1",
                (guild_id, category_id, facts_channel_id, reminders_channel_id),
            )
            await db.commit()
            log.info("guild_config_saved", guild_id=guild_id)
        except aiosqlite.Error as e:
            log.error("guild_config_save_failed", guild_id=guild_id, error=str(e))
