"""Memoria longitudinal con SQLite async.

Per-user context: each user has their own conversation with Insult.
Contexto jerarquico: reciente + relevante (keyword match).
Append-only: nunca se borra, solo crece.
"""

import contextlib
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
        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN for_user_id TEXT")

        # Migration: add guild_id and channel_name for cross-channel awareness
        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN guild_id TEXT")
        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN channel_name TEXT")

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
        # Idempotent migration: add 'source' column so we can distinguish
        # facts written by extract_facts (auto) from curated injections
        # (manual). save_facts only wipes the auto ones; manual survives
        # forever. Prior to this column, any manual fact died on the next
        # conversation turn because save_facts DELETE'd all user_facts rows.
        with contextlib.suppress(aiosqlite.OperationalError):
            # Fails silently when the column already exists after first run.
            await self._db.execute("ALTER TABLE user_facts ADD COLUMN source TEXT NOT NULL DEFAULT 'auto'")
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
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS channel_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                last_message_ts REAL NOT NULL,
                is_private INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_channel_summaries_guild_channel
            ON channel_summaries(guild_id, channel_id)
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                guild_id TEXT,
                created_by TEXT NOT NULL,
                description TEXT NOT NULL,
                remind_at REAL NOT NULL,
                mention_user_ids TEXT NOT NULL DEFAULT '',
                recurring TEXT NOT NULL DEFAULT 'none',
                delivered INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_pending
            ON reminders(delivered, remind_at)
        """)

        # --- Phase 1 tables (v3.0.0): disclosure, emotional arcs, stance, contradictions ---
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS disclosure_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                severity INTEGER NOT NULL,
                signals TEXT NOT NULL,
                message_excerpt TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_disclosure_user
            ON disclosure_log(user_id, timestamp DESC)
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS emotional_arcs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                phase_since REAL NOT NULL,
                crisis_depth INTEGER NOT NULL DEFAULT 0,
                recovery_signals INTEGER NOT NULL DEFAULT 0,
                turns_in_phase INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_arc_user_channel
            ON emotional_arcs(channel_id, user_id)
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS stance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                position TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_stance_channel_user
            ON stance_log(channel_id, user_id, timestamp DESC)
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS contradiction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                prior_statement TEXT NOT NULL,
                contradicting_statement TEXT NOT NULL,
                topic TEXT NOT NULL,
                called_out INTEGER NOT NULL DEFAULT 0,
                timestamp REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_contradiction_user
            ON contradiction_log(user_id, timestamp DESC)
        """)

        # --- Guild config (v3.3.0): system channels for facts/reminders ---
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS guild_config (
                guild_id TEXT PRIMARY KEY,
                category_id TEXT,
                facts_channel_id TEXT,
                reminders_channel_id TEXT,
                setup_complete INTEGER NOT NULL DEFAULT 0
            )
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
        guild_id: str | None = None,
        channel_name: str | None = None,
    ):
        """Guarda un mensaje en la memoria longitudinal.

        Args:
            for_user_id: For assistant messages, the user_id this reply is for.
                         Enables per-user context isolation.
            guild_id: Discord guild (server) ID for cross-channel awareness.
            channel_name: Human-readable channel name for summaries.
        """
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO messages (channel_id, user_id, user_name, role, content, timestamp, for_user_id, guild_id, channel_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (channel_id, user_id, user_name, role, content, time.time(), for_user_id, guild_id, channel_name),
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

    async def get_all_facts(self) -> list[dict]:
        """Get all facts for all users, grouped by user_id."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT user_id, id, fact, category, updated_at FROM user_facts ORDER BY user_id, updated_at DESC",
        )
        rows = await cursor.fetchall()
        return [{"user_id": r[0], "id": r[1], "fact": r[2], "category": r[3], "updated_at": r[4]} for r in rows]

    async def save_facts(self, user_id: str, facts: list[dict]):
        """Replace AUTO-extracted facts for a user with new ones.

        Facts inserted via `add_manual_fact` (source='manual') are preserved —
        only rows with source='auto' get wiped and rewritten. This prevents
        the previous bug where any curated fact died on the next conversation
        turn, because the LLM only sees the last ~10 messages and routinely
        omits older facts from its output.

        Each fact dict should have 'fact' and 'category' keys. All inserts
        here are marked source='auto'; use `add_manual_fact` for curated ones.
        """
        await self._ensure_connection()
        now = time.time()
        try:
            await self._db.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND source = 'auto'",
                (user_id,),
            )
            for f in facts:
                await self._db.execute(
                    "INSERT INTO user_facts (user_id, fact, category, updated_at, source) VALUES (?, ?, ?, ?, 'auto')",
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

    async def add_manual_fact(
        self,
        user_id: str,
        fact: str,
        category: str = "general",
    ) -> int:
        """Insert a curated fact marked source='manual' so extract_facts can't wipe it.

        Returns the inserted row id. Manual facts are intended for:
        - Human-curated injections from chat operators
        - Important incidents the LLM omitted on its own
        - Signals other users gave about the target (cross-user attribution)

        Caller is responsible for deduping if they care — this is an append,
        not an upsert.
        """
        await self._ensure_connection()
        now = time.time()
        cursor = await self._db.execute(
            "INSERT INTO user_facts (user_id, fact, category, updated_at, source) VALUES (?, ?, ?, ?, 'manual')",
            (user_id, fact, category, now),
        )
        await self._db.commit()
        row_id = cursor.lastrowid
        log.info("manual_fact_added", user_id=user_id, fact_id=row_id, category=category)
        return row_id or 0

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

    # --- Channel Summaries (cross-channel awareness) ---

    async def get_channel_participants(self, channel_id: str, limit: int = 10) -> list[dict]:
        """Get distinct users who posted in a channel, most recent first."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT user_id, user_name, MAX(timestamp) as last_ts FROM messages "
            "WHERE channel_id = ? AND role = 'user' "
            "GROUP BY user_id ORDER BY last_ts DESC LIMIT ?",
            (channel_id, limit),
        )
        rows = await cursor.fetchall()
        return [{"user_id": r[0], "user_name": r[1], "last_ts": r[2]} for r in rows]

    async def get_channel_activity_since(self, guild_id: str, since_ts: float) -> list[dict]:
        """Returns (channel_id, count) pairs for channels with activity since given timestamp."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT channel_id, COUNT(*) as cnt FROM messages "
            "WHERE guild_id = ? AND timestamp > ? "
            "GROUP BY channel_id ORDER BY cnt DESC",
            (guild_id, since_ts),
        )
        rows = await cursor.fetchall()
        return [{"channel_id": r[0], "count": r[1]} for r in rows]

    async def get_channels_overview(self, limit: int = 50) -> list[dict]:
        """Returns all channels with message counts and latest timestamp. Debug helper."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT channel_id, MAX(channel_name), MAX(guild_id), COUNT(*), MAX(timestamp) "
            "FROM messages GROUP BY channel_id ORDER BY MAX(timestamp) DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "channel_id": r[0],
                "channel_name": r[1],
                "guild_id": r[2],
                "count": r[3],
                "last_ts": r[4],
            }
            for r in rows
        ]

    async def get_recent_for_summary(self, channel_id: str, limit: int = 50) -> list[dict]:
        """Get recent messages for summarization (used by background task)."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT user_name, role, content, timestamp FROM messages "
            "WHERE channel_id = ? ORDER BY timestamp DESC LIMIT ?",
            (channel_id, limit),
        )
        rows = await cursor.fetchall()
        return [{"user_name": r[0], "role": r[1], "content": r[2], "timestamp": r[3]} for r in reversed(rows)]

    async def upsert_channel_summary(
        self,
        guild_id: str,
        channel_id: str,
        channel_name: str,
        summary: str,
        message_count: int,
        last_message_ts: float,
        is_private: bool = False,
    ):
        """Insert or update a channel summary (ON CONFLICT upsert)."""
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO channel_summaries (guild_id, channel_id, channel_name, summary, message_count, last_message_ts, is_private, updated_at) "
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
            await self._db.commit()
            log.info("channel_summary_upserted", guild_id=guild_id, channel_id=channel_id, channel_name=channel_name)
        except aiosqlite.Error as e:
            log.error("channel_summary_upsert_failed", channel_id=channel_id, error=str(e))

    async def get_channel_summaries(
        self, guild_id: str, exclude_channel_id: str | None = None, limit: int = 8
    ) -> list[dict]:
        """Get channel summaries for the server pulse, excluding the current channel."""
        await self._ensure_connection()
        if exclude_channel_id:
            cursor = await self._db.execute(
                "SELECT channel_id, channel_name, summary, message_count, last_message_ts, is_private, updated_at "
                "FROM channel_summaries WHERE guild_id = ? AND channel_id != ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (guild_id, exclude_channel_id, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT channel_id, channel_name, summary, message_count, last_message_ts, is_private, updated_at "
                "FROM channel_summaries WHERE guild_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
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

    # --- Reminders ---

    async def save_reminder(
        self,
        channel_id: str,
        guild_id: str | None,
        created_by: str,
        description: str,
        remind_at: float,
        mention_user_ids: str = "",
        recurring: str = "none",
    ) -> int:
        """Save a new reminder. Returns the reminder ID."""
        await self._ensure_connection()
        try:
            cursor = await self._db.execute(
                "INSERT INTO reminders (channel_id, guild_id, created_by, description, remind_at, "
                "mention_user_ids, recurring, delivered, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (channel_id, guild_id, created_by, description, remind_at, mention_user_ids, recurring, time.time()),
            )
            await self._db.commit()
            reminder_id = cursor.lastrowid
            log.info(
                "reminder_saved",
                reminder_id=reminder_id,
                channel_id=channel_id,
                description=description[:80],
                remind_at=remind_at,
            )
            return reminder_id
        except aiosqlite.Error as e:
            log.error("reminder_save_failed", channel_id=channel_id, error=str(e))
            raise

    async def get_pending_reminders(self, now: float) -> list[dict]:
        """Get all reminders that are due (remind_at <= now and not delivered)."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT id, channel_id, guild_id, created_by, description, remind_at, "
            "mention_user_ids, recurring FROM reminders "
            "WHERE delivered = 0 AND remind_at <= ? ORDER BY remind_at ASC",
            (now,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "channel_id": r[1],
                "guild_id": r[2],
                "created_by": r[3],
                "description": r[4],
                "remind_at": r[5],
                "mention_user_ids": r[6],
                "recurring": r[7],
            }
            for r in rows
        ]

    async def mark_reminder_delivered(self, reminder_id: int) -> None:
        """Mark a reminder as delivered."""
        await self._ensure_connection()
        try:
            await self._db.execute("UPDATE reminders SET delivered = 1 WHERE id = ?", (reminder_id,))
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("reminder_mark_delivered_failed", reminder_id=reminder_id, error=str(e))

    async def update_reminder_time(self, reminder_id: int, new_remind_at: float) -> None:
        """Update the remind_at time for a recurring reminder."""
        await self._ensure_connection()
        try:
            await self._db.execute("UPDATE reminders SET remind_at = ? WHERE id = ?", (new_remind_at, reminder_id))
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("reminder_update_time_failed", reminder_id=reminder_id, error=str(e))

    async def get_channel_reminders(self, channel_id: str) -> list[dict]:
        """Get all pending (not delivered) reminders for a channel."""
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT id, channel_id, guild_id, created_by, description, remind_at, "
            "mention_user_ids, recurring FROM reminders "
            "WHERE channel_id = ? AND delivered = 0 ORDER BY remind_at ASC",
            (channel_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "channel_id": r[1],
                "guild_id": r[2],
                "created_by": r[3],
                "description": r[4],
                "remind_at": r[5],
                "mention_user_ids": r[6],
                "recurring": r[7],
            }
            for r in rows
        ]

    async def delete_reminder(self, reminder_id: int) -> bool:
        """Delete a reminder by ID. Returns True if a row was deleted."""
        await self._ensure_connection()
        try:
            cursor = await self._db.execute("DELETE FROM reminders WHERE id = ? AND delivered = 0", (reminder_id,))
            await self._db.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                log.info("reminder_deleted", reminder_id=reminder_id)
            else:
                log.warning("reminder_delete_not_found", reminder_id=reminder_id)
            return deleted
        except aiosqlite.Error as e:
            log.error("reminder_delete_failed", reminder_id=reminder_id, error=str(e))
            return False

    # --- Disclosure Log (v3.0.0) ---

    async def store_disclosure(
        self, channel_id: str, user_id: str, category: str, severity: int, signals: str, excerpt: str
    ) -> None:
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO disclosure_log (channel_id, user_id, category, severity, signals, message_excerpt, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (channel_id, user_id, category, severity, signals, excerpt[:200], time.time()),
            )
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("disclosure_store_failed", error=str(e))

    # --- Emotional Arcs (v3.0.0) ---

    async def get_arc(self, channel_id: str, user_id: str) -> dict | None:
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT phase, phase_since, crisis_depth, recovery_signals, turns_in_phase, updated_at "
            "FROM emotional_arcs WHERE channel_id = ? AND user_id = ?",
            (channel_id, user_id),
        )
        row = await cursor.fetchone()
        if row:
            return {
                "phase": row[0],
                "phase_since": row[1],
                "crisis_depth": row[2],
                "recovery_signals": row[3],
                "turns_in_phase": row[4],
                "updated_at": row[5],
            }
        return None

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
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO emotional_arcs (channel_id, user_id, phase, phase_since, crisis_depth, "
                "recovery_signals, turns_in_phase, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(channel_id, user_id) DO UPDATE SET phase=excluded.phase, "
                "phase_since=excluded.phase_since, crisis_depth=excluded.crisis_depth, "
                "recovery_signals=excluded.recovery_signals, turns_in_phase=excluded.turns_in_phase, "
                "updated_at=excluded.updated_at",
                (channel_id, user_id, phase, phase_since, crisis_depth, recovery_signals, turns_in_phase, time.time()),
            )
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("arc_upsert_failed", error=str(e))

    # --- Stance Log (v3.0.0) ---

    async def store_stance(self, channel_id: str, user_id: str, topic: str, position: str, confidence: float) -> None:
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO stance_log (channel_id, user_id, topic, position, confidence, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (channel_id, user_id, topic, position[:200], confidence, time.time()),
            )
            await self._db.commit()
            # FIFO eviction: keep max 20 per context
            await self._db.execute(
                "DELETE FROM stance_log WHERE id NOT IN ("
                "SELECT id FROM stance_log WHERE channel_id = ? AND user_id = ? "
                "ORDER BY timestamp DESC LIMIT 20)",
                (channel_id, user_id),
            )
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("stance_store_failed", error=str(e))

    async def get_stances(self, channel_id: str, user_id: str, limit: int = 5) -> list[dict]:
        await self._ensure_connection()
        cursor = await self._db.execute(
            "SELECT topic, position, confidence, timestamp FROM stance_log "
            "WHERE channel_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (channel_id, user_id, limit),
        )
        rows = await cursor.fetchall()
        return [{"topic": r[0], "position": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]

    # --- Contradiction Log (v3.0.0) ---

    async def store_contradiction(self, user_id: str, prior: str, contradicting: str, topic: str) -> None:
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO contradiction_log (user_id, prior_statement, contradicting_statement, "
                "topic, called_out, timestamp) VALUES (?, ?, ?, ?, 0, ?)",
                (user_id, prior[:300], contradicting[:300], topic, time.time()),
            )
            await self._db.commit()
        except aiosqlite.Error as e:
            log.error("contradiction_store_failed", error=str(e))

    # --- Guild Config (v3.3.0) ---

    async def get_guild_config(self, guild_id: str) -> dict | None:
        """Get guild config (system channel IDs). Returns None if not set up."""
        await self._ensure_connection()
        cursor = await self._db.execute(
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
        """Save or update guild config with system channel IDs."""
        await self._ensure_connection()
        try:
            await self._db.execute(
                "INSERT INTO guild_config (guild_id, category_id, facts_channel_id, "
                "reminders_channel_id, setup_complete) VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(guild_id) DO UPDATE SET category_id=excluded.category_id, "
                "facts_channel_id=excluded.facts_channel_id, "
                "reminders_channel_id=excluded.reminders_channel_id, setup_complete=1",
                (guild_id, category_id, facts_channel_id, reminders_channel_id),
            )
            await self._db.commit()
            log.info("guild_config_saved", guild_id=guild_id)
        except aiosqlite.Error as e:
            log.error("guild_config_save_failed", guild_id=guild_id, error=str(e))
