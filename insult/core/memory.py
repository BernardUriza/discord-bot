"""Memoria longitudinal con SQLite async.

Per-user context: each user has their own conversation with Insult.
Contexto jerarquico: reciente + relevante (keyword match).
Append-only: nunca se borra, solo crece.
"""

import time
from pathlib import Path

import aiosqlite
import structlog

from insult.core.style import UserStyleProfile

log = structlog.get_logger()


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self):
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                for_user_id TEXT
            )
        """)
        # Migration: add for_user_id if upgrading from old schema
        try:
            await self._db.execute("ALTER TABLE messages ADD COLUMN for_user_id TEXT")
        except aiosqlite.OperationalError:
            pass  # column already exists

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_channel_ts
            ON messages(channel_id, timestamp DESC)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_context
            ON messages(channel_id, user_id, timestamp DESC)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_for_user
            ON messages(channel_id, for_user_id, timestamp DESC)
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await self._db.commit()
        log.info("memory_connected", db_path=str(self.db_path))

    async def _ensure_connection(self):
        """Reconecta si la conexion se perdio."""
        if self._db is None:
            log.warning("memory_reconnecting")
            await self.connect()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None
            log.info("memory_closed")

    async def store(
        self,
        channel_id: str,
        user_id: str,
        user_name: str,
        role: str,
        content: str,
        for_user_id: str | None = None,
    ):
        """Guarda un mensaje en la memoria longitudinal.

        Args:
            for_user_id: For assistant messages, the user_id this reply is for.
                         Enables per-user context isolation.
        """
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO messages (channel_id, user_id, user_name, role, content, timestamp, for_user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (channel_id, user_id, user_name, role, content, time.time(), for_user_id),
            )
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("memory_store_failed", channel_id=channel_id, user_id=user_id, error=str(e))
            raise

    async def get_recent(self, channel_id: str, limit: int = 20, user_id: str | None = None) -> list[dict]:
        """Ultimos N mensajes del canal, opcionalmente filtrados por usuario.

        When user_id is provided, returns only:
        - Messages sent BY that user
        - Assistant replies TO that user (for_user_id match)
        """
        await self._ensure_connection()
        if user_id:
            cursor = await self._db.execute(
                "SELECT user_name, role, content, timestamp FROM messages "
                "WHERE channel_id = ? AND (user_id = ? OR for_user_id = ?) "
                "ORDER BY timestamp DESC LIMIT ?",
                (channel_id, user_id, user_id, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT user_name, role, content, timestamp FROM messages "
                "WHERE channel_id = ? ORDER BY timestamp DESC LIMIT ?",
                (channel_id, limit),
            )
        rows = await cursor.fetchall()
        return [{"user_name": r[0], "role": r[1], "content": r[2], "timestamp": r[3]} for r in reversed(rows)]

    async def search(self, channel_id: str, query: str, limit: int = 5, user_id: str | None = None) -> list[dict]:
        """Busca mensajes relevantes por keywords, opcionalmente filtrados por usuario."""
        await self._ensure_connection()
        words = [f"%{w}%" for w in query.split() if len(w) > 2]
        if not words:
            return []

        conditions = " OR ".join(["content LIKE ?"] * len(words))
        if user_id:
            cursor = await self._db.execute(
                f"SELECT user_name, role, content, timestamp FROM messages "  # noqa: S608
                f"WHERE channel_id = ? AND (user_id = ? OR for_user_id = ?) AND ({conditions}) "
                f"ORDER BY timestamp DESC LIMIT ?",
                (channel_id, user_id, user_id, *words, limit),
            )
        else:
            cursor = await self._db.execute(
                f"SELECT user_name, role, content, timestamp FROM messages "  # noqa: S608
                f"WHERE channel_id = ? AND ({conditions}) "
                f"ORDER BY timestamp DESC LIMIT ?",
                (channel_id, *words, limit),
            )
        rows = await cursor.fetchall()
        return [{"user_name": r[0], "role": r[1], "content": r[2], "timestamp": r[3]} for r in reversed(rows)]

    async def get_stats(self, channel_id: str | None = None) -> dict:
        """Estadisticas de la memoria."""
        await self._ensure_connection()
        if channel_id:
            cursor = await self._db.execute(
                "SELECT COUNT(*), COUNT(DISTINCT user_id) FROM messages WHERE channel_id = ?",
                (channel_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT channel_id) FROM messages"
            )
        row = await cursor.fetchone()
        return {
            "total_messages": row[0],
            "unique_users": row[1],
            "unique_channels": row[2] if len(row) > 2 else None,
        }

    async def get_profile(self, user_id: str) -> UserStyleProfile:
        """Recupera el perfil de estilo de un usuario, o crea uno nuevo."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            return UserStyleProfile.from_json(row[0])
        return UserStyleProfile()

    async def update_profile(self, user_id: str, message: str) -> UserStyleProfile:
        """Actualiza el perfil de estilo con un nuevo mensaje (EMA)."""
        profile = await self.get_profile(user_id)
        profile.update(message)

        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO user_profiles (user_id, profile_json, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=excluded.updated_at",
                (user_id, profile.to_json(), time.time()),
            )
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("profile_update_failed", user_id=user_id, error=str(e))

        return profile

    async def delete_before(self, cutoff: float) -> int:
        """Delete messages older than cutoff timestamp. Returns count deleted."""
        await self._ensure_connection()
        cursor = await self._db.execute("SELECT COUNT(*) FROM messages WHERE timestamp < ?", (cutoff,))
        row = await cursor.fetchone()
        count = row[0]
        if count > 0:
            await self._db.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
            await self._db.commit()
            log.info("memory_cleaned", deleted=count, cutoff=cutoff)
        return count

    def build_context(self, recent: list[dict], relevant: list[dict] | None = None) -> list[dict]:
        """Construye el contexto para el LLM: reciente + relevante."""
        context = []

        if relevant:
            seen_contents = {m["content"] for m in recent}
            unique_relevant = [m for m in relevant if m["content"] not in seen_contents]
            if unique_relevant:
                context.append(
                    {
                        "role": "user",
                        "content": "[Contexto relevante de conversaciones anteriores]\n"
                        + "\n".join(f"{m['user_name']}: {m['content']}" for m in unique_relevant),
                    }
                )

        for msg in recent:
            content = f"{msg['user_name']}: {msg['content']}" if msg["role"] == "user" else msg["content"]
            context.append({"role": msg["role"], "content": content})

        return context
