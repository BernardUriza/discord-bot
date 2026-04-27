"""Repository for ``dream_diary`` rows.

Thin SQL wrappers — no business logic, no LLM calls. The schema lives
in :mod:`insult.core.memory.connection`. Keeping this module dumb on
purpose so it can be unit-tested against an in-memory SQLite without
mocking the LLM client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from insult.core.memory import MemoryStore

log = structlog.get_logger()

VALID_STATUS = ("ok", "partial", "failed")


@dataclass(frozen=True)
class DreamEntry:
    """One row in ``dream_diary``."""

    id: int
    run_ts: float
    duration_ms: int
    users_total: int
    users_processed: int
    facts_in_total: int
    facts_out_total: int
    deletes_total: int
    updates_total: int
    status: str
    content: str
    error: str | None


def _row_to_entry(row: tuple) -> DreamEntry:
    return DreamEntry(
        id=row[0],
        run_ts=row[1],
        duration_ms=row[2],
        users_total=row[3],
        users_processed=row[4],
        facts_in_total=row[5],
        facts_out_total=row[6],
        deletes_total=row[7],
        updates_total=row[8],
        status=row[9],
        content=row[10],
        error=row[11],
    )


async def insert_entry(
    memory: MemoryStore,
    *,
    duration_ms: int,
    users_total: int,
    users_processed: int,
    facts_in_total: int,
    facts_out_total: int,
    deletes_total: int,
    updates_total: int,
    status: str,
    content: str,
    error: str | None = None,
    run_ts: float | None = None,
) -> int | None:
    """Persist one diary row. Returns the new id, or None on failure."""
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status {status!r}; must be one of {VALID_STATUS}")
    db = memory._db
    if db is None:
        log.warning("dream_diary_insert_skipped_no_db")
        return None
    cursor = await db.execute(
        "INSERT INTO dream_diary "
        "(run_ts, duration_ms, users_total, users_processed, "
        " facts_in_total, facts_out_total, deletes_total, updates_total, "
        " status, content, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_ts if run_ts is not None else time.time(),
            duration_ms,
            users_total,
            users_processed,
            facts_in_total,
            facts_out_total,
            deletes_total,
            updates_total,
            status,
            content,
            error,
        ),
    )
    await db.commit()
    return cursor.lastrowid


async def latest_entry(memory: MemoryStore) -> DreamEntry | None:
    """Most recent diary row, or None if the table is empty."""
    db = memory._db
    if db is None:
        return None
    cursor = await db.execute(
        "SELECT id, run_ts, duration_ms, users_total, users_processed, "
        "facts_in_total, facts_out_total, deletes_total, updates_total, "
        "status, content, error "
        "FROM dream_diary ORDER BY run_ts DESC LIMIT 1"
    )
    row = await cursor.fetchone()
    return _row_to_entry(row) if row else None


async def recent_entries(memory: MemoryStore, limit: int = 5) -> list[DreamEntry]:
    """Most recent ``limit`` entries, newest first."""
    db = memory._db
    if db is None:
        return []
    cursor = await db.execute(
        "SELECT id, run_ts, duration_ms, users_total, users_processed, "
        "facts_in_total, facts_out_total, deletes_total, updates_total, "
        "status, content, error "
        "FROM dream_diary ORDER BY run_ts DESC LIMIT ?",
        (limit,),
    )
    rows = await cursor.fetchall()
    return [_row_to_entry(r) for r in rows]
