"""Azure Blob Storage backup for SQLite database.

Optional: only activates if AZURE_STORAGE_CONNECTION_STRING is set.
Downloads DB on startup, uploads on shutdown + periodic backup.

Race-condition handling: `download_db` tracks the blob's last_modified
timestamp in a module-level _last_known_blob_mtime. `upload_db` checks
before writing — if the remote blob is newer than what we last saw,
we abort instead of overwriting. This stops the shutdown hook of a
dying container from clobbering the new container's already-uploaded
version during a restart.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import structlog

log = structlog.get_logger()

CONTAINER_NAME = "insult-bot"
BLOB_NAME = "memory.db"

# Tracks the blob's last_modified as of our most recent read. Periodic upload
# refreshes this after a successful write; shutdown compares against it.
_last_known_blob_mtime: datetime | None = None


def is_azure_configured() -> bool:
    """Check if Azure Blob Storage credentials are available."""
    return bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))


def should_abort_upload(
    current_remote_mtime: datetime | None,
    last_known_local_mtime: datetime | None,
    *,
    skip_if_remote_newer: bool = True,
) -> bool:
    """Decide whether to skip an upload because the remote blob is newer.

    Pure function — no SDK, no I/O — so the shutdown-race policy can be
    unit-tested without azure-storage-blob installed locally. Returns True
    when the caller MUST abort the upload (remote has been modified by
    another writer after our last download/upload).

    Rules:
    - If skip_if_remote_newer is False, never abort.
    - If we have no last-known mtime, proceed (first-ever upload / race-free).
    - If we can't determine the remote mtime, proceed (fail-open — better a
      stale overwrite than no backup).
    - Otherwise, abort iff the remote mtime is STRICTLY newer than what we
      last observed.
    """
    if not skip_if_remote_newer:
        return False
    if last_known_local_mtime is None:
        return False
    if current_remote_mtime is None:
        return False
    return current_remote_mtime > last_known_local_mtime


async def download_db(db_path: Path) -> bool:
    """Download memory.db from Azure Blob Storage if it exists.

    Records the blob's last_modified so upload_db can detect if the blob
    has been modified by someone else (new container, manual operator)
    before overwriting it.

    Returns True if downloaded, False if not found or not configured.
    """
    global _last_known_blob_mtime

    if not is_azure_configured():
        return False

    try:
        from azure.storage.blob.aio import BlobServiceClient

        conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        async with BlobServiceClient.from_connection_string(conn_str) as client:
            container = client.get_container_client(CONTAINER_NAME)

            # Create container if it doesn't exist
            try:
                await container.create_container()
                log.info("azure_container_created", container=CONTAINER_NAME)
            except Exception:  # noqa: S110 — expected when container already exists
                pass

            blob = container.get_blob_client(BLOB_NAME)
            try:
                stream = await blob.download_blob()
                data = await stream.readall()
                props = await blob.get_blob_properties()
                _last_known_blob_mtime = props.last_modified

                # Don't overwrite a local DB that already has data with an empty/smaller blob
                if db_path.exists() and db_path.stat().st_size > len(data):
                    log.warning(
                        "azure_download_skipped_larger_local",
                        local_size=db_path.stat().st_size,
                        blob_size=len(data),
                    )
                    return False

                db_path.parent.mkdir(parents=True, exist_ok=True)
                db_path.write_bytes(data)
                log.info(
                    "azure_db_downloaded",
                    size=len(data),
                    path=str(db_path),
                    blob_mtime=props.last_modified.isoformat() if props.last_modified else None,
                )
                return True
            except Exception:
                log.info("azure_db_not_found", blob=BLOB_NAME)
                return False

    except ImportError:
        log.warning("azure_sdk_not_installed", hint="pip install azure-storage-blob")
        return False
    except Exception:
        log.exception("azure_download_failed")
        return False


async def upload_db(db_path: Path, *, skip_if_remote_newer: bool = True) -> bool:
    """Upload memory.db to Azure Blob Storage.

    Creates a consistent snapshot via SQLite backup API (safe even with WAL mode),
    then uploads the snapshot. This avoids issues with WAL journal not being flushed.

    If `skip_if_remote_newer` is True and the remote blob has a last_modified
    timestamp after our last known one (i.e. someone else — typically the new
    container that just arrived during a restart — wrote to it), we skip the
    upload and log the reason. This prevents the shutdown race where the dying
    container overwrites the fresh upload.

    Returns True if uploaded, False if not configured, failed, or skipped.
    """
    global _last_known_blob_mtime

    if not is_azure_configured():
        return False

    if not db_path.exists():
        log.warning("azure_upload_skipped", reason="db file does not exist")
        return False

    try:
        import sqlite3

        from azure.storage.blob.aio import BlobServiceClient

        # Create a consistent snapshot using SQLite backup API
        # This merges WAL into a single file — safe while DB is open
        snapshot_path = db_path.parent / "memory_backup.db"
        src = sqlite3.connect(str(db_path))
        dst = sqlite3.connect(str(snapshot_path))
        src.backup(dst)
        src.close()
        dst.close()

        conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        async with BlobServiceClient.from_connection_string(conn_str) as client:
            container = client.get_container_client(CONTAINER_NAME)
            blob = container.get_blob_client(BLOB_NAME)

            # Freshness check: has the remote changed since we last saw it?
            if skip_if_remote_newer and _last_known_blob_mtime is not None:
                current_mtime: datetime | None = None
                try:
                    current_props = await blob.get_blob_properties()
                    current_mtime = current_props.last_modified
                except Exception:
                    # Fail-open: better a stale overwrite than no backup.
                    log.warning("azure_upload_freshness_check_failed_proceeding")
                if should_abort_upload(
                    current_mtime,
                    _last_known_blob_mtime,
                    skip_if_remote_newer=True,
                ):
                    log.warning(
                        "azure_upload_aborted_remote_newer",
                        last_known=_last_known_blob_mtime.isoformat(),
                        current=current_mtime.isoformat() if current_mtime else None,
                        reason="remote was updated by another writer (likely the replacement container)",
                    )
                    snapshot_path.unlink(missing_ok=True)
                    return False

            data = snapshot_path.read_bytes()
            resp = await blob.upload_blob(data, overwrite=True)
            # Refresh our tracked mtime so later uploads from this same
            # process compare against our own write, not a stale download.
            _last_known_blob_mtime = resp.get("last_modified") or datetime.now(UTC)
            log.info("azure_db_uploaded", size=len(data), blob=BLOB_NAME)

            # Cleanup snapshot
            snapshot_path.unlink(missing_ok=True)
            return True

    except ImportError:
        log.warning("azure_sdk_not_installed", hint="pip install azure-storage-blob")
        return False
    except Exception:
        log.exception("azure_upload_failed")
        return False
