"""Dashboard configuration."""

VERSION = "1.0.0"

# Azure Blob Storage URLs (public read access)
BLOB_BASE = "https://insultstorage.blob.core.windows.net/insult-bot"
METRICS_URL = f"{BLOB_BASE}/metrics.json"
LOGS_URL = f"{BLOB_BASE}/logs.json"

# Refresh interval (ms)
REFRESH_INTERVAL = 30_000  # 30 seconds
