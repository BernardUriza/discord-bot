"""Owns the aiosqlite connection, schema creation, and migrations.

Single process-wide connection, shared out to all repositories via
BaseRepository. See `base.py` for the rationale (SQLite single-writer +
WAL interaction). Schema lives here because all tables must exist before
any repository performs its first query — and because the migrations
(`ALTER TABLE ... ADD COLUMN`) are point-in-time decisions, not per-repo.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger()


class ConnectionManager:
    """Lifecycle manager for the shared aiosqlite connection.

    Callers use `get_connection()` to receive the live handle and
    `close()` to drain the WAL and disconnect. Schema initialization runs
    once inside `connect()` — migrations are idempotent (`IF NOT EXISTS`
    + `ALTER TABLE ... ADD COLUMN` wrapped in `suppress`) so boot order
    is safe across redeploys even when the schema evolves.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._vectors_available: bool = False

    @property
    def vectors_available(self) -> bool:
        return self._vectors_available

    @property
    def raw_db(self) -> aiosqlite.Connection | None:
        """Escape hatch for code that needs the raw handle (vectors helpers)."""
        return self._db

    async def connect(self) -> None:
        """Open the connection and initialize the full schema.

        Idempotent: safe to call on boot of an existing DB. Columns added
        post-initial-design are wrapped in `suppress(OperationalError)`
        so a second call (column already exists) doesn't crash.
        """
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")

        await self._create_messages_table()
        await self._create_user_profiles_table()
        await self._create_user_facts_table()
        await self._create_world_scans_table()
        await self._create_channel_summaries_table()
        await self._create_reminders_table()
        await self._create_phase1_tables()
        await self._create_guild_config_table()

        await self._db.commit()

        await self._init_vectors()

        log.info("memory_connected", db_path=str(self.db_path))

    async def get_connection(self) -> aiosqlite.Connection:
        """Return the live connection, reconnecting if it was dropped."""
        if self._db is None:
            log.warning("memory_reconnecting")
            await self.connect()
        assert self._db is not None
        return self._db

    async def close(self) -> None:
        """Checkpoint WAL and close. Critical for Azure Blob backup to be consistent."""
        if self._db is None:
            return

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

    # -- Schema creation (split into one method per logical group) --

    async def _create_messages_table(self) -> None:
        assert self._db is not None
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
        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN for_user_id TEXT")
        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN guild_id TEXT")
        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN channel_name TEXT")
        with contextlib.suppress(aiosqlite.OperationalError):
            await self._db.execute("ALTER TABLE messages ADD COLUMN model_used TEXT")

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

    async def _create_user_profiles_table(self) -> None:
        assert self._db is not None
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

    async def _create_user_facts_table(self) -> None:
        assert self._db is not None
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
            await self._db.execute("ALTER TABLE user_facts ADD COLUMN source TEXT NOT NULL DEFAULT 'auto'")
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_facts
            ON user_facts(user_id)
        """)

    async def _create_world_scans_table(self) -> None:
        assert self._db is not None
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

    async def _create_channel_summaries_table(self) -> None:
        assert self._db is not None
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

    async def _create_reminders_table(self) -> None:
        assert self._db is not None
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

    async def _create_phase1_tables(self) -> None:
        """Phase 1 (v3.0.0): disclosure, emotional arcs, stance, contradictions."""
        assert self._db is not None
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

    async def _create_guild_config_table(self) -> None:
        """v3.3.0: system channels for facts/reminders."""
        assert self._db is not None
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS guild_config (
                guild_id TEXT PRIMARY KEY,
                category_id TEXT,
                facts_channel_id TEXT,
                reminders_channel_id TEXT,
                setup_complete INTEGER NOT NULL DEFAULT 0
            )
        """)

    async def _init_vectors(self) -> None:
        """Initialize sqlite-vec + FTS5 for semantic search. Best-effort."""
        assert self._db is not None
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
