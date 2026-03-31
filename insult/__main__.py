"""CLI entry point: python -m insult run"""

import asyncio

import typer
import structlog

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
        await store._ensure_connection()

        cursor = await store._db.execute(
            "SELECT COUNT(*) FROM messages WHERE timestamp < ?", (cutoff,)
        )
        row = await cursor.fetchone()
        count = row[0]

        if dry_run:
            typer.echo(f"[DRY RUN] Would delete {count} messages older than {before_days} days")
        else:
            await store._db.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
            await store._db.commit()
            typer.echo(f"Deleted {count} messages older than {before_days} days")

        await store.close()

    asyncio.run(_clean())


if __name__ == "__main__":
    app()
