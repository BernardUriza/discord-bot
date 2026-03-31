"""Azure Blob Storage backup for SQLite database.

Optional: only activates if AZURE_STORAGE_CONNECTION_STRING is set.
Downloads DB on startup, uploads on shutdown + periodic backup.
"""

import os
from pathlib import Path

import structlog

log = structlog.get_logger()

CONTAINER_NAME = "insult-bot"
BLOB_NAME = "memory.db"


def is_azure_configured() -> bool:
    """Check if Azure Blob Storage credentials are available."""
    return bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))


async def download_db(db_path: Path) -> bool:
    """Download memory.db from Azure Blob Storage if it exists.

    Returns True if downloaded, False if not found or not configured.
    """
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
                log.info("azure_db_downloaded", size=len(data), path=str(db_path))
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


async def upload_db(db_path: Path) -> bool:
    """Upload memory.db to Azure Blob Storage.

    Creates a consistent snapshot via SQLite backup API (safe even with WAL mode),
    then uploads the snapshot. This avoids issues with WAL journal not being flushed.

    Returns True if uploaded, False if not configured or failed.
    """
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

            data = snapshot_path.read_bytes()
            await blob.upload_blob(data, overwrite=True)
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
