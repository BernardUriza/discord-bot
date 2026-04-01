"""Discord bot setup: events, lifecycle, health check."""

import asyncio
import signal

import structlog
from discord.ext import commands, tasks

from insult.app import Container, create_app
from insult.cogs import ChatCog, UtilityCog
from insult.cogs.voice import VoiceCog
from insult.core.backup import download_db, is_azure_configured, upload_db
from insult.core.character import _get_current_time_context, strip_metadata
from insult.core.errors import get_error_response
from insult.core.proactive import generate_proactive_message, should_send_now

log = structlog.get_logger()


def _build(container: Container):
    """Register cogs and event handlers on the bot."""
    bot = container.bot
    memory = container.memory

    # --- Graceful Shutdown ---
    async def graceful_shutdown(sig: signal.Signals):
        log.info("shutdown_signal", signal=sig.name)
        _health_check.cancel()
        if _backup_task.is_running():
            _backup_task.cancel()
        await memory.close()
        await upload_db(container.settings.db_path)
        await bot.close()
        log.info("shutdown_complete")

    def _bind_signals():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(graceful_shutdown(s)))

    # --- Azure Backup (every 10 min) ---
    @tasks.loop(minutes=10)
    async def _backup_task():
        if is_azure_configured():
            try:
                # Checkpoint WAL without closing — safe while DB is in use
                import contextlib

                with contextlib.suppress(Exception):
                    await memory._db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                await upload_db(container.settings.db_path)
            except Exception:
                log.exception("azure_backup_failed")

    # --- Proactive Messaging (check every 30 min, send ~every 2-3h) ---
    _last_proactive_ts: float | None = None

    @tasks.loop(minutes=30)
    async def _proactive_task():
        nonlocal _last_proactive_ts
        from datetime import datetime as dt
        from zoneinfo import ZoneInfo

        now = dt.now(ZoneInfo("America/Mexico_City"))

        if not should_send_now(now.hour, _last_proactive_ts):
            return

        # Find the most recently active text channel
        target_channel = None
        latest_ts = 0
        for guild in bot.guilds:
            for channel in guild.text_channels:
                try:
                    stats = await memory.get_stats(str(channel.id))
                    if stats["total_messages"] and stats["total_messages"] > latest_ts:
                        latest_ts = stats["total_messages"]
                        target_channel = channel
                except Exception:
                    log.debug("proactive_channel_skip", channel=channel.name)

        if not target_channel:
            return

        # Gather user facts for all known users
        try:
            all_user_data = await memory.get_all_user_messages(limit_per_user=5)
            user_facts = {}
            for uid, data in all_user_data.items():
                facts = await memory.get_facts(uid)
                if facts:
                    user_facts[data["user_name"]] = facts

            recent = await memory.get_recent(str(target_channel.id), limit=5)
            recent_summary = "\n".join(f"{m['user_name']}: {m['content'][:80]}" for m in recent) if recent else ""
        except Exception:
            log.exception("proactive_context_failed")
            return

        time_str = _get_current_time_context()
        msg = await generate_proactive_message(
            container.llm.client, container.settings.llm_model, time_str, user_facts, recent_summary
        )

        if msg:
            msg = strip_metadata(msg)
            try:
                parts = [p.strip() for p in msg.split("[SEND]") if p.strip()]
                for part in parts:
                    await target_channel.send(part)
                _last_proactive_ts = dt.now().timestamp()
                # Store in memory
                await memory.store(
                    str(target_channel.id), str(bot.user.id), bot.user.name, "assistant", msg.replace("[SEND]", "\n")
                )
                log.info("proactive_message_sent", channel=target_channel.name, length=len(msg))
            except Exception:
                log.exception("proactive_send_failed")

    # --- Health Check ---
    @tasks.loop(seconds=60)
    async def _health_check():
        try:
            stats = await memory.get_stats()
            log.info(
                "health_check",
                latency_ms=round(bot.latency * 1000),
                guilds=len(bot.guilds),
                total_messages=stats["total_messages"],
                unique_users=stats["unique_users"],
            )
        except Exception:
            log.exception("health_check_failed")

    # --- Events ---
    _ready_fired = False

    @bot.event
    async def on_ready():
        nonlocal _ready_fired
        # Download DB from Azure on first startup (if configured)
        if not _ready_fired and is_azure_configured():
            await download_db(container.settings.db_path)
        await memory.connect()
        if not _ready_fired:
            _bind_signals()
            await bot.add_cog(ChatCog(container))
            await bot.add_cog(UtilityCog(container))
            await bot.add_cog(VoiceCog(container))
            _health_check.start()
            _proactive_task.start()
            if is_azure_configured():
                _backup_task.start()
            _ready_fired = True
        log.info(
            "bot_ready",
            user=str(bot.user),
            guilds=len(bot.guilds),
            model=container.settings.llm_model,
            prefix=container.settings.command_prefix,
            memory_recent=container.settings.memory_recent_limit,
            memory_relevant=container.settings.memory_relevant_limit,
        )

    @bot.event
    async def on_disconnect():
        log.warning("bot_disconnected")

    @bot.event
    async def on_resumed():
        log.info("bot_resumed")

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"Calmate, espera {error.retry_after:.0f}s. Cual es la urgencia?")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"Y el resto del mensaje? Te falto `{error.param.name}`. Intenta otra vez, completo.")
        elif isinstance(error, commands.CommandNotFound):
            pass
        else:
            log.error("command_error", command=str(ctx.command), error=str(error), user=str(ctx.author))
            await ctx.send(get_error_response("generic"))

    return bot


def run():
    """Create app, build bot, and run."""
    container = create_app()
    bot = _build(container)
    bot.run(container.settings.discord_token.get_secret_value(), log_handler=None)
