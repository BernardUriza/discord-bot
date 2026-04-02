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
        self._vectors_available: bool = False

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
        import contextlib

        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN for_user_id TEXT")

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
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                updated_at REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_facts
            ON user_facts(user_id)
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS world_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                findings TEXT NOT NULL,
                commentary TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_world_scans_ts
            ON world_scans(timestamp DESC)
        """)
        await self._db.commit()

        # Initialize vector search (sqlite-vec + FTS5)
        try:
            import sqlite_vec

            self._db._conn.enable_load_extension(True)
            sqlite_vec.load(self._db._conn)

            from insult.core.vectors import init_vector_tables

            await init_vector_tables(self._db)
            self._vectors_available = True
            log.info("vectors_initialized")
        except Exception as e:
            log.warning("vectors_not_available", reason=str(e))
            self._vectors_available = False

        log.info("memory_connected", db_path=str(self.db_path))

    async def _ensure_connection(self):
        """Reconecta si la conexion se perdio."""
        if self._db is None:
            log.warning("memory_reconnecting")
            await self.connect()

    async def close(self):
        if self._db:
            # Checkpoint WAL to merge journal into main DB file before closing.
            # Critical for Azure backup: upload_db reads the main file only.
            try:
                await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                log.warning("wal_checkpoint_failed_retrying_passive")
                try:
                    await self._db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    log.warning("wal_checkpoint_passive_also_failed")
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

    @staticmethod
    def _format_relative_time(timestamp: float) -> str:
        """Convert Unix timestamp to human-readable relative time."""
        now = time.time()
        diff = now - timestamp

        if diff < 60:
            return "justo ahora"
        if diff < 3600:
            mins = int(diff / 60)
            return f"hace {mins}min"
        if diff < 86400:
            hours = int(diff / 3600)
            return f"hace {hours}h"
        days = int(diff / 86400)
        if days == 1:
            return "ayer"
        if days < 7:
            return f"hace {days} días"
        if days < 30:
            weeks = int(days / 7)
            return f"hace {weeks} sem"
        return f"hace {int(days / 30)} meses"

    def build_context(self, recent: list[dict], relevant: list[dict] | None = None) -> list[dict]:
        """Construye el contexto para el LLM: reciente + relevante (con timestamps)."""
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
                            f"[{self._format_relative_time(m['timestamp'])}] {m['user_name']}: {m['content']}"
                            for m in unique_relevant
                        ),
                    }
                )

        for msg in recent:
            ts = f"[{self._format_relative_time(msg['timestamp'])}] "
            # Both user and assistant messages get name prefix for clear speaker attribution
            content = f"{ts}{msg['user_name']}: {msg['content']}"
            context.append({"role": msg["role"], "content": content})

        return context

    async def get_all_user_messages(self, limit_per_user: int = 30) -> dict[str, list[dict]]:
        """Get recent messages grouped by user_id (cross-channel). For bulk fact extraction."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT DISTINCT user_id, user_name FROM messages WHERE role = 'user' ORDER BY timestamp DESC"
        )
        users = await cursor.fetchall()

        result = {}
        for user_id, user_name in users:
            cursor = await self._db.execute(
                "SELECT user_name, role, content, timestamp FROM messages "
                "WHERE user_id = ? OR for_user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, user_id, limit_per_user),
            )
            rows = await cursor.fetchall()
            result[user_id] = {
                "user_name": user_name,
                "messages": [
                    {"user_name": r[0], "role": r[1], "content": r[2], "timestamp": r[3]} for r in reversed(rows)
                ],
            }
        return result

    # --- User Facts ---

    async def get_facts(self, user_id: str) -> list[dict]:
        """Get all facts for a user, ordered by most recently updated."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT id, fact, category, updated_at FROM user_facts WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [{"id": r[0], "fact": r[1], "category": r[2], "updated_at": r[3]} for r in rows]

    async def save_facts(self, user_id: str, facts: list[dict]):
        """Replace all facts for a user with new ones (full rewrite per extraction).

        Each fact dict should have 'fact' and 'category' keys.
        """
        await self._ensure_connection()
        now = time.time()
        try:
            await self._db.execute("DELETE FROM user_facts WHERE user_id = ?", (user_id,))
            for f in facts:
                await self._db.execute(
                    "INSERT INTO user_facts (user_id, fact, category, updated_at) VALUES (?, ?, ?, ?)",
                    (user_id, f["fact"], f.get("category", "general"), now),
                )
            await self._db.commit()
            log.info("facts_saved", user_id=user_id, count=len(facts))

            # Update vector embeddings if available
            if self._vectors_available:
                try:
                    from insult.core.vectors import upsert_fact_vectors

                    await upsert_fact_vectors(self._db, user_id, facts)
                except Exception as ve:
                    log.warning("facts_vector_upsert_failed", user_id=user_id, error=str(ve))
        except aiosqlite.Error as e:
            log.error("facts_save_failed", user_id=user_id, error=str(e))

    async def search_facts_semantic(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        """Search user facts by semantic similarity using hybrid vector + FTS search.

        Falls back to returning all facts if vectors are not available.
        """
        if not self._vectors_available:
            return await self.get_facts(user_id)

        await self._ensure_connection()
        try:
            from insult.core.vectors import search_facts_hybrid

            results = await search_facts_hybrid(self._db, user_id, query, limit=limit)
            if results:
                log.info("facts_semantic_search", user_id=user_id, query=query[:50], results=len(results))
                return results
            # If search returned nothing, fall back to all facts
            return await self.get_facts(user_id)
        except Exception as e:
            log.warning("facts_semantic_search_failed", user_id=user_id, error=str(e))
            return await self.get_facts(user_id)

    # --- World Scans (Insult's internal knowledge of the world) ---

    async def store_world_scan(self, topic: str, findings: str, commentary: str):
        """Store a world scan result — Insult's internal memory of what's happening out there."""
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO world_scans (topic, findings, commentary, timestamp) VALUES (?, ?, ?, ?)",
                (topic, findings, commentary, time.time()),
            )
            await self._db.commit()
            log.info("world_scan_stored", topic=topic[:80])
        except aiosqlite.Error as e:
            log.error("world_scan_store_failed", error=str(e))

    async def get_recent_world_scans(self, limit: int = 5) -> list[dict]:
        """Get most recent world scans for context in conversations."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT topic, findings, commentary, timestamp FROM world_scans ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [{"topic": r[0], "findings": r[1], "commentary": r[2], "timestamp": r[3]} for r in rows]
