"""CLI entry point: python -m insult run

IMPORTANT: structlog must be configured BEFORE any insult module is imported,
because modules do `log = structlog.get_logger()` at import time.
"""

import asyncio
import os

import structlog


def _metrics_processor(logger, method_name, event_dict):
    """Structlog processor that feeds events to the dashboard metrics collector."""
    from insult.core.metrics import record_event

    record_event(event_dict)
    return event_dict


# Pick renderer based on LOG_FORMAT env var (json for KQL, console for local dev)
_log_format = os.environ.get("LOG_FORMAT", "console").lower()
_renderer = structlog.processors.JSONRenderer() if _log_format == "json" else structlog.dev.ConsoleRenderer()

# Configure structlog FIRST — before any insult.* import
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        _metrics_processor,
        _renderer,
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,  # Don't cache — ensures all loggers use this config
)

# NOW import everything else
import typer  # noqa: E402

log = structlog.get_logger()
app = typer.Typer(help="Insult — Discord bot con memoria longitudinal + Claude API")


@app.command()
def run():
    """Start the Discord bot."""
    from insult.bot import run as bot_run

    bot_run()


@app.command()
def db_stats():
    """Show memory database statistics."""
    from insult.config import settings
    from insult.core.memory import MemoryStore

    async def _stats():
        store = MemoryStore(settings.db_path)
        await store.connect()
        stats = await store.get_stats()
        await store.close()
        return stats

    stats = asyncio.run(_stats())
    typer.echo(f"Total messages: {stats['total_messages']}")
    typer.echo(f"Unique users:   {stats['unique_users']}")
    typer.echo(f"Channels:       {stats['unique_channels']}")


@app.command()
def db_clean(
    before_days: int = typer.Option(90, help="Delete messages older than N days"),
    dry_run: bool = typer.Option(True, help="Preview without deleting"),
):
    """Clean old messages from memory database."""
    import time

    from insult.config import settings
    from insult.core.memory import MemoryStore

    cutoff = time.time() - (before_days * 86400)

    async def _clean():
        store = MemoryStore(settings.db_path)
        await store.connect()

        if dry_run:
            await store._ensure_connection()
            cursor = await store._db.execute("SELECT COUNT(*) FROM messages WHERE timestamp < ?", (cutoff,))
            row = await cursor.fetchone()
            typer.echo(f"[DRY RUN] Would delete {row[0]} messages older than {before_days} days")
        else:
            count = await store.delete_before(cutoff)
            typer.echo(f"Deleted {count} messages older than {before_days} days")

        await store.close()

    asyncio.run(_clean())


if __name__ == "__main__":
    app()
