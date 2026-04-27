"""Siesta — auto-pause coordination between consolidator and bot.

While the memory consolidator runs (a separate Azure Container App Job),
the bot replica goes silent: it stores incoming messages, reacts with
🛌 to acknowledge them, but does not call the LLM. This avoids
mtime-collision races on the blob (the upload-side problem fixed in
:mod:`insult.core.backup`) and saves tokens on responses generated
against state that's about to be replaced.

Public API:
- :class:`SiestaSnapshot`, :class:`SiestaPhase`: in-memory state types.
- :class:`SiestaPoller`: bot-side periodic blob-metadata reader.
- ``mark_started`` / ``mark_progress`` / ``mark_finished``: consolidator-side writers.
"""

from insult.core.siesta.coordination.blob_metadata import (
    mark_finished,
    mark_progress,
    mark_started,
    read_snapshot,
)
from insult.core.siesta.coordination.poller import (
    DEFAULT_INTERVAL_SECONDS,
    SiestaPoller,
)
from insult.core.siesta.state import AWAKE, SiestaPhase, SiestaSnapshot

__all__ = [
    "AWAKE",
    "DEFAULT_INTERVAL_SECONDS",
    "SiestaPhase",
    "SiestaPoller",
    "SiestaSnapshot",
    "mark_finished",
    "mark_progress",
    "mark_started",
    "read_snapshot",
]
