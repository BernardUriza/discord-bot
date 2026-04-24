"""User style profiles — EMA-tracked speech characteristics per user."""

from __future__ import annotations

import time

import aiosqlite
import structlog

from insult.core.memory.base import BaseRepository
from insult.core.style import UserStyleProfile

log = structlog.get_logger()


class ProfilesRepository(BaseRepository):
    """Owns the `user_profiles` table.

    The profile is serialized to JSON because the shape evolves (new
    metrics get added) and a schema migration per change would be painful
    for a field nobody queries on — we always load-by-user_id and update
    holistically. Confidence gates + adaptation logic live in `core/style`."""

    async def get_profile(self, user_id: str) -> UserStyleProfile:
        """Load the stored profile, or return a fresh default if none exists."""
        db = await self._conn()
        cursor = await db.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            return UserStyleProfile.from_json(row[0])
        return UserStyleProfile()

    async def update_profile(self, user_id: str, message: str) -> UserStyleProfile:
        """Incorporate a new message into the user's style profile via EMA.

        Logs and swallows aiosqlite errors so a DB hiccup doesn't crash the
        turn — the user still gets a response, the profile just stalls."""
        profile = await self.get_profile(user_id)
        profile.update(message)

        db = await self._conn()
        try:
            await db.execute(
                "INSERT INTO user_profiles (user_id, profile_json, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "profile_json=excluded.profile_json, updated_at=excluded.updated_at",
                (user_id, profile.to_json(), time.time()),
            )
            await db.commit()
        except aiosqlite.Error as e:
            log.error("profile_update_failed", user_id=user_id, error=str(e))

        return profile
