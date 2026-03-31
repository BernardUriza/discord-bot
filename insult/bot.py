"""Discord bot setup: events, lifecycle, health check."""

import asyncio
import signal

import structlog
from discord.ext import commands, tasks

from insult.app import Container, create_app
from insult.cogs import ChatCog, UtilityCog
from insult.core.errors import get_error_response

log = structlog.get_logger()


def _build(container: Container):
    """Register cogs and event handlers on the bot."""
    bot = container.bot
    memory = container.memory

    # --- Graceful Shutdown ---
    async def graceful_shutdown(sig: signal.Signals):
        log.info("shutdown_signal", signal=sig.name)
        _health_check.cancel()
        await memory.close()
        await bot.close()
        log.info("shutdown_complete")

    def _bind_signals():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(graceful_shutdown(s))
            )

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
    @bot.event
    async def on_ready():
        await memory.connect()
        _bind_signals()
        await bot.add_cog(ChatCog(container))
        await bot.add_cog(UtilityCog(container))
        _health_check.start()
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
