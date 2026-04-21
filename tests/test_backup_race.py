"""Unit tests for the shutdown-upload race policy (pure function).

The full upload_db path needs azure-storage-blob installed, which isn't a
test-time dependency. The race decision itself is a pure function
(`should_abort_upload`) so we test THAT directly — the caller just wires
it to the blob properties.

Scenario reproduced tonight (2026-04-21 05:05 UTC): operator pushed a new
memory.db to the blob, triggered a container restart. New container
downloaded + uploaded (16 facts). Three seconds later the OLD container's
graceful_shutdown fired and called upload_db(), blindly overwriting the
blob with its stale in-memory DB (3 facts). 13 manual facts lost.

The fix encoded here: upload MUST abort when the current remote mtime is
strictly newer than the mtime we observed at our last download/upload.
"""

from __future__ import annotations

from datetime import UTC, datetime

from insult.core.backup import should_abort_upload

T_DOWNLOAD = datetime(2026, 4, 21, 5, 5, 39, tzinfo=UTC)
T_REMOTE_NEWER = datetime(2026, 4, 21, 5, 5, 40, tzinfo=UTC)
T_REMOTE_SAME = T_DOWNLOAD
T_REMOTE_OLDER = datetime(2026, 4, 21, 5, 5, 30, tzinfo=UTC)


def test_aborts_when_remote_is_strictly_newer():
    # The exact race from 2026-04-21: old container about to overwrite a
    # remote that the new container already wrote.
    assert should_abort_upload(T_REMOTE_NEWER, T_DOWNLOAD) is True


def test_proceeds_when_remote_equals_last_known():
    assert should_abort_upload(T_REMOTE_SAME, T_DOWNLOAD) is False


def test_proceeds_when_remote_is_older():
    # Defensive: clocks or cache weirdness. Err on the side of uploading.
    assert should_abort_upload(T_REMOTE_OLDER, T_DOWNLOAD) is False


def test_proceeds_on_first_ever_upload_no_tracked_mtime():
    assert should_abort_upload(T_REMOTE_NEWER, None) is False


def test_proceeds_when_remote_mtime_is_unknown():
    # If Azure didn't return last_modified for some reason, we don't stall backups.
    assert should_abort_upload(None, T_DOWNLOAD) is False


def test_override_force_overwrite_bypasses_check():
    # Escape hatch for operators: skip_if_remote_newer=False forces through.
    assert should_abort_upload(T_REMOTE_NEWER, T_DOWNLOAD, skip_if_remote_newer=False) is False
