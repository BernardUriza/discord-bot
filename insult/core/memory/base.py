"""Base class for all memory repositories.

Every repository holds a reference to the shared `ConnectionManager` and
ALWAYS calls `await self._conn()` at the top of a data-access method. This
keeps the auto-reconnect logic in exactly one place — previously duplicated
40+ times across MemoryStore as `await self._ensure_connection()` calls,
each of which was a possible bug surface.

The shared-connection design (one aiosqlite.Connection per process, fanned
out to all repos) is intentional: SQLite allows only one writer anyway, so
maintaining a pool would just queue writes behind each other. If this ever
grows into a distributed store, the `ConnectionManager` abstraction lets us
swap in a backend without touching each repository's SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from insult.core.memory.connection import ConnectionManager


class BaseRepository:
    """Shared superclass for every domain repository.

    Subclasses should never open their own aiosqlite connection — they call
    `await self._conn()` which returns the live, auto-reconnected handle
    owned by the `ConnectionManager`. This is critical for data consistency:
    two independent writers on the same SQLite file will deadlock on WAL.
    """

    def __init__(self, manager: ConnectionManager):
        self._manager = manager

    async def _conn(self) -> aiosqlite.Connection:
        """Return the live connection, reconnecting if it was dropped."""
        return await self._manager.get_connection()

    @property
    def vectors_available(self) -> bool:
        """Whether sqlite-vec + FTS5 initialized successfully at boot."""
        return self._manager.vectors_available
