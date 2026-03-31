"""Memoria longitudinal con SQLite async.

Inspirado en free-intelligence/aurity.io:
- Sin sesiones: una conversación infinita por canal
- Contexto jerárquico: reciente + relevante (keyword match)
- Append-only: nunca se borra, solo crece
"""

import aiosqlite
import time
from pathlib import Path


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
                timestamp REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_channel_ts
            ON messages(channel_id, timestamp DESC)
        """)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def store(
        self,
        channel_id: str,
        user_id: str,
        user_name: str,
        role: str,
        content: str,
    ):
        """Guarda un mensaje en la memoria longitudinal."""
        await self._db.execute(
            "INSERT INTO messages (channel_id, user_id, user_name, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (channel_id, user_id, user_name, role, content, time.time()),
        )
        await self._db.commit()

    async def get_recent(self, channel_id: str, limit: int = 20) -> list[dict]:
        """Últimos N mensajes del canal (siempre incluidos en contexto)."""
        cursor = await self._db.execute(
            "SELECT user_name, role, content, timestamp FROM messages "
            "WHERE channel_id = ? ORDER BY timestamp DESC LIMIT ?",
            (channel_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"user_name": r[0], "role": r[1], "content": r[2], "timestamp": r[3]}
            for r in reversed(rows)
        ]

    async def search(
        self, channel_id: str, query: str, limit: int = 5
    ) -> list[dict]:
        """Busca mensajes relevantes por keywords (cross-session)."""
        words = [f"%{w}%" for w in query.split() if len(w) > 2]
        if not words:
            return []

        conditions = " OR ".join(["content LIKE ?"] * len(words))
        cursor = await self._db.execute(
            f"SELECT user_name, role, content, timestamp FROM messages "
            f"WHERE channel_id = ? AND ({conditions}) "
            f"ORDER BY timestamp DESC LIMIT ?",
            (channel_id, *words, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"user_name": r[0], "role": r[1], "content": r[2], "timestamp": r[3]}
            for r in reversed(rows)
        ]

    async def get_stats(self, channel_id: str | None = None) -> dict:
        """Estadísticas de la memoria."""
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

    def build_context(
        self, recent: list[dict], relevant: list[dict] | None = None
    ) -> list[dict]:
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
                        + "\n".join(
                            f'{m["user_name"]}: {m["content"]}' for m in unique_relevant
                        ),
                    }
                )

        for msg in recent:
            context.append({"role": msg["role"], "content": msg["content"]})

        return context
