"""Discord bot setup: events, lifecycle, health check."""

import asyncio
import signal
import time as _time

import structlog
from discord.ext import commands, tasks

from insult.app import Container, create_app
from insult.cogs import ChatCog, UtilityCog
from insult.cogs.voice import VoiceCog
from insult.core.backup import download_db, is_azure_configured, upload_db
from insult.core.character import _get_current_time_context, strip_metadata
from insult.core.delivery import MESSAGE_DELIMITER, split_response
from insult.core.errors import get_error_response
from insult.core.proactive import (
    generate_proactive_message,
    generate_world_scan_message,
    should_send_now,
    should_world_scan,
)
from insult.core.reminders import compute_next_occurrence

log = structlog.get_logger()


def _build(container: Container):
    """Register cogs and event handlers on the bot."""
    bot = container.bot
    memory = container.memory

    # --- Graceful Shutdown ---
    async def graceful_shutdown(sig: signal.Signals):
        log.info("shutdown_signal", signal=sig.name)
        _health_check.cancel()
        if _reminder_check_task.is_running():
            _reminder_check_task.cancel()
        if _summarize_channels_task.is_running():
            _summarize_channels_task.cancel()
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

        # ~30% of the time: world scan (search the web and comment)
        # ~70% of the time: social check-in (ask about users)
        is_world_scan = should_world_scan()
        log.info("proactive_mode_selected", mode="world_scan" if is_world_scan else "social")

        if is_world_scan:
            scan_result = await generate_world_scan_message(
                container.llm.client, container.settings.llm_model, time_str, user_facts
            )
            msg = scan_result.commentary if scan_result else None
        else:
            scan_result = None
            msg = await generate_proactive_message(
                container.llm.client, container.settings.llm_model, time_str, user_facts, recent_summary
            )

        if msg:
            msg = strip_metadata(msg)
            try:
                parts = split_response(msg)
                for part in parts:
                    await target_channel.send(part)
                _last_proactive_ts = dt.now().timestamp()

                # Store in conversation memory
                await memory.store(
                    str(target_channel.id),
                    str(bot.user.id),
                    bot.user.name,
                    "assistant",
                    msg.replace(MESSAGE_DELIMITER, "\n"),
                )

                # World scan: persist to internal DB + post to feed channel
                if scan_result:
                    await memory.store_world_scan(scan_result.topic, scan_result.findings, scan_result.commentary)
                    # Post to dedicated feed channel if configured
                    feed_channel_name = "insult-world-feed"
                    for guild in bot.guilds:
                        feed = next((ch for ch in guild.text_channels if ch.name == feed_channel_name), None)
                        if feed and feed.id != target_channel.id:
                            feed_parts = split_response(msg)
                            for part in feed_parts:
                                await feed.send(part)
                            log.info("world_scan_feed_posted", channel=feed.name)

                log.info(
                    "proactive_message_sent",
                    channel=target_channel.name,
                    length=len(msg),
                    mode="world_scan" if is_world_scan else "social",
                )
            except Exception:
                log.exception("proactive_send_failed")

    # --- Channel Summarization (cross-channel awareness, every 15 min) ---
    @tasks.loop(minutes=container.settings.summary_interval_minutes)
    async def _summarize_channels_task():
        from insult.core.summaries import summarize_channel

        try:
            now_ts = __import__("time").time()
            # Summarize channels with 10+ new messages since last summary
            channels_processed = 0
            max_per_tick = 10

            for guild in bot.guilds:
                guild_id = str(guild.id)
                # Get activity since 1 hour ago (fallback window for new channels)
                since_ts = now_ts - 3600
                activity = await memory.get_channel_activity_since(guild_id, since_ts)

                for item in activity:
                    if channels_processed >= max_per_tick:
                        break
                    if item["count"] < 10:
                        continue

                    ch_id = item["channel_id"]
                    # Find the Discord channel object for name + privacy info
                    channel = guild.get_channel(int(ch_id))
                    if channel is None:
                        continue

                    ch_name = channel.name
                    is_private = not channel.permissions_for(guild.default_role).read_messages

                    messages = await memory.get_recent_for_summary(ch_id, limit=50)
                    if not messages:
                        continue

                    summary = await summarize_channel(
                        container.llm.client,
                        container.settings.summary_model,
                        ch_name,
                        messages,
                    )
                    if summary:
                        last_ts = messages[-1]["timestamp"] if messages else now_ts
                        await memory.upsert_channel_summary(
                            guild_id, ch_id, ch_name, summary, item["count"], last_ts, is_private
                        )
                        channels_processed += 1

                if channels_processed >= max_per_tick:
                    break

            if channels_processed > 0:
                log.info("channel_summaries_updated", count=channels_processed)
        except Exception:
            log.exception("channel_summarization_task_failed")

    # --- Reminder Delivery (check every 30s for due reminders) ---
    @tasks.loop(seconds=30)
    async def _reminder_check_task():
        try:
            now = _time.time()
            pending = await memory.get_pending_reminders(now)
            for reminder in pending:
                channel = bot.get_channel(int(reminder["channel_id"]))
                if not channel:
                    await memory.mark_reminder_delivered(reminder["id"])
                    log.warning("reminder_channel_gone", reminder_id=reminder["id"], channel_id=reminder["channel_id"])
                    continue

                # Build mention string
                mentions = ""
                if reminder["mention_user_ids"]:
                    user_ids = reminder["mention_user_ids"].split(",")
                    mentions = " ".join(f"<@{uid.strip()}>" for uid in user_ids if uid.strip())

                # Generate in-character reminder via LLM
                reminder_prompt = (
                    f"{container.settings.system_prompt[:2000]}\n\n"
                    "## Special Task: Deliver a Reminder\n"
                    f"You need to deliver a reminder. The reminder is: '{reminder['description']}'. "
                    "Deliver it in-character — short, punchy, maybe a bit aggressive. "
                    "Don't explain you're a bot delivering a reminder. Just remind them naturally. "
                    "1-2 sentences max."
                )

                try:
                    response = await container.llm.chat(
                        reminder_prompt,
                        [{"role": "user", "content": f"Recordatorio: {reminder['description']}"}],
                    )
                    text = response.text.strip()
                except Exception:
                    log.exception("reminder_llm_failed", reminder_id=reminder["id"])
                    text = f"\u23f0 Recordatorio: {reminder['description']}"

                msg = f"{mentions} {text}".strip() if mentions else text
                try:
                    await channel.send(msg)
                except Exception:
                    log.exception("reminder_send_failed", reminder_id=reminder["id"])

                # Handle recurring
                if reminder["recurring"] != "none":
                    next_time = compute_next_occurrence(reminder["remind_at"], reminder["recurring"])
                    if next_time:
                        await memory.update_reminder_time(reminder["id"], next_time)
                    else:
                        await memory.mark_reminder_delivered(reminder["id"])
                else:
                    await memory.mark_reminder_delivered(reminder["id"])

                log.info("reminder_delivered", reminder_id=reminder["id"], channel_id=reminder["channel_id"])
        except Exception:
            log.exception("reminder_check_failed")

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
            _reminder_check_task.start()
            _proactive_task.start()
            _summarize_channels_task.start()
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
