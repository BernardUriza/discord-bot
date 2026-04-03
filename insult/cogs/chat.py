"""Chat cog — responds to all messages, no prefix needed."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import discord
import structlog
from discord.ext import commands

from insult.core.actions import (
    CHANNEL_TOOLS,
    execute_create_channel,
    execute_edit_channel,
    execute_get_channel_info,
)
from insult.core.attachments import process_attachments
from insult.core.character import build_adaptive_prompt
from insult.core.delivery import MESSAGE_DELIMITER, send_response
from insult.core.errors import classify_error, get_error_response
from insult.core.facts import build_facts_prompt, extract_facts
from insult.core.flows import ExpressionHistory, analyze_flows, build_flow_prompt, validate_flow_adherence
from insult.core.llm import WEB_SEARCH_TOOL
from insult.core.presets import PresetMode, PresetModifier
from insult.core.reactions import add_reactions, parse_reactions, strip_reactions
from insult.core.reminders import REMINDER_TOOLS, format_reminder_list, parse_remind_at
from insult.core.summaries import build_server_pulse, filter_by_permissions

if TYPE_CHECKING:
    from insult.app import Container

log = structlog.get_logger()

# Static tool list — built once, reused every message
_ALL_TOOLS = list(CHANNEL_TOOLS) + list(REMINDER_TOOLS)

MAX_MESSAGE_LENGTH = 4000
BATCH_WAIT_SECONDS = 3.0  # Wait this long after last message before responding
MIN_RESPONSE_GAP = 5.0  # Minimum seconds between bot responses to same user (token protection)


@dataclass
class _MessageBatch:
    """Accumulates rapid-fire messages from one user before responding."""

    messages: list[discord.Message] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None


class ChatCog(commands.Cog):
    def __init__(self, container: Container):
        self.memory = container.memory
        self.llm = container.llm
        self.settings = container.settings
        self.bot = container.bot
        self._background_tasks: set[asyncio.Task] = set()
        self._expression_history = ExpressionHistory()
        self._processed_messages: set[int] = set()  # dedup: prevent double responses
        self._processed_max = 1000  # rotate after this many to prevent memory leak
        # Message batching: accumulate rapid messages, respond to the batch
        self._pending_batches: dict[str, _MessageBatch] = {}  # key: "{channel_id}:{user_id}"
        self._last_response_time: dict[int, float] = {}  # user_id → monotonic timestamp

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Respond to every message — no !chat prefix needed.

        Messages are batched: rapid-fire messages from the same user are
        accumulated and responded to together after a short pause, like a
        human waiting for someone to finish typing.
        """
        # Ignore bots (including ourselves)
        if message.author.bot:
            return

        # Dedup: prevent double responses from gateway replays or reconnects
        if message.id in self._processed_messages:
            return
        self._processed_messages.add(message.id)
        if len(self._processed_messages) > self._processed_max:
            # Discard oldest half to prevent memory leak
            to_keep = sorted(self._processed_messages)[self._processed_max // 2 :]
            self._processed_messages = set(to_keep)

        # Ignore other commands (!ping, !memoria, !buscar, !perfil, !chat)
        if message.content.startswith(self.settings.command_prefix):
            return

        # Transcribe voice messages via Whisper
        text = message.content.strip()
        if message.flags.voice and message.attachments:
            text = await self._transcribe_voice(message) or ""

        # Ignore empty messages (e.g. only attachments with no text, stickers)
        if not text and not message.attachments:
            return

        if len(text) > MAX_MESSAGE_LENGTH:
            await message.channel.send(get_error_response("too_long"))
            return

        # Token protection: hard minimum gap between bot responses per user
        now = time.monotonic()
        last_response = self._last_response_time.get(message.author.id, 0)
        if now - last_response < MIN_RESPONSE_GAP:
            # Still accumulate in memory — don't lose context
            try:
                await self.memory.store(
                    str(message.channel.id),
                    str(message.author.id),
                    message.author.display_name,
                    "user",
                    text,
                    guild_id=str(message.guild.id) if message.guild else None,
                    channel_name=message.channel.name if hasattr(message.channel, "name") else None,
                )
            except Exception:
                log.exception("chat_store_cooldown_failed")
            return

        # Batch messages: accumulate rapid-fire messages, respond after a pause
        batch_key = f"{message.channel.id}:{message.author.id}"
        batch = self._pending_batches.get(batch_key)

        if batch is None:
            batch = _MessageBatch()
            self._pending_batches[batch_key] = batch

        batch.messages.append(message)
        batch.texts.append(text)

        # Cancel previous timer if it exists (user sent another message)
        if batch.timer is not None:
            batch.timer.cancel()

        # Schedule batch processing after BATCH_WAIT_SECONDS of silence
        loop = asyncio.get_running_loop()
        batch.timer = loop.call_later(
            BATCH_WAIT_SECONDS,
            lambda k=batch_key: asyncio.create_task(self._flush_batch(k)),
        )

    async def _flush_batch(self, batch_key: str):
        """Process accumulated messages as a single response."""
        batch = self._pending_batches.pop(batch_key, None)
        if not batch or not batch.messages:
            return

        # Use the LAST message as the target for reactions and replies
        last_message = batch.messages[-1]
        # Combine all texts into one (preserving order)
        combined_text = "\n".join(batch.texts)

        await self._respond(last_message, combined_text)

    @commands.command(name="chat")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def chat(self, ctx: commands.Context, *, message: str):
        """Fallback: !chat still works for explicit invocation."""
        await self._respond(ctx.message, message)

    async def _respond(self, message: discord.Message, text: str):
        """Core response logic — shared by on_message and !chat command."""
        self._last_response_time[message.author.id] = time.monotonic()

        channel_id = str(message.channel.id)
        user_id = str(message.author.id)
        user_name = message.author.display_name

        # Process attachments
        attachment_blocks = []
        if message.attachments:
            blocks, errors = await process_attachments(message.attachments)
            attachment_blocks = blocks
            for err in errors:
                await message.channel.send(err)

        guild_id = str(message.guild.id) if message.guild else None
        channel_name = message.channel.name if hasattr(message.channel, "name") else None

        try:
            await self.memory.store(
                channel_id,
                user_id,
                user_name,
                "user",
                text,
                guild_id=guild_id,
                channel_name=channel_name,
            )
        except Exception:
            log.exception("chat_store_user_failed", channel_id=channel_id)

        try:
            profile = await self.memory.update_profile(user_id, text)
        except Exception:
            log.exception("chat_profile_update_failed", user_id=user_id)
            profile = None

        context, recent = await self._build_context(channel_id, text, attachment_blocks)
        if context is None:
            await message.channel.send(get_error_response("context_failed"))
            return

        # Load user facts — use semantic search for relevance when many facts exist
        user_facts = await self._load_facts_smart(user_id, text)

        # Build server pulse (cross-channel awareness)
        server_pulse = ""
        if message.guild:
            try:
                summaries = await self.memory.get_channel_summaries(
                    str(message.guild.id), exclude_channel_id=channel_id
                )
                if summaries:
                    accessible = {
                        str(ch.id)
                        for ch in message.guild.text_channels
                        if ch.permissions_for(message.author).read_messages
                    }
                    summaries = filter_by_permissions(summaries, accessible)
                    server_pulse = build_server_pulse(summaries, text)
            except Exception:
                log.exception("chat_server_pulse_failed", guild_id=str(message.guild.id))

        # Run flow analysis (depends on preset, so build_adaptive_prompt goes first)
        context_key = f"{channel_id}:{user_id}"
        # First pass: get preset from build_adaptive_prompt
        system_prompt, preset = build_adaptive_prompt(
            self.settings.system_prompt,
            profile,
            len(context),
            current_message=text,
            recent_messages=recent,
            user_facts=user_facts,
            server_pulse=server_pulse,
        )
        # Run 4-flow behavioral analysis
        flow_analysis = analyze_flows(text, recent, preset, self._expression_history, context_key)
        # Inject flow guidance as Layer 3.5 (appended after preset, before facts)
        flow_prompt = build_flow_prompt(flow_analysis)
        if flow_prompt:
            system_prompt += f"\n\n{flow_prompt}"
        system_prompt += build_facts_prompt(user_name, user_facts)

        if profile and profile.is_confident:
            log.info(
                "style_adapted",
                user_id=user_id,
                preset=preset.mode.value,
                preset_modifiers=[m.value for m in preset.modifiers],
                language=profile.detected_language,
                formality=round(profile.formality, 2),
                technical=round(profile.technical_level, 2),
                verbosity=round(profile.avg_word_count, 1),
            )

        # Build tools list: static tools + web search (disabled during crisis)
        tools = list(_ALL_TOOLS)
        if preset.mode != PresetMode.RESPECTFUL_SERIOUS:
            tools.append(WEB_SEARCH_TOOL)

        # If user wants a server action, force tool_choice="any" so Claude MUST use the tool.
        force_tool = PresetModifier.ACTION_INTENT in preset.modifiers
        tool_choice = {"type": "any"} if force_tool else None

        try:
            async with message.channel.typing():
                llm_response = await self.llm.chat(system_prompt, context, tools=tools, tool_choice=tool_choice)
        except Exception as e:
            log.exception("chat_llm_failed", channel_id=channel_id, error_type=type(e).__name__)
            await message.channel.send(get_error_response(classify_error(e)))
            return

        response = llm_response.text

        # Extract and strip emoji reactions
        reactions = parse_reactions(response)
        response = strip_reactions(response)

        # Store the full response in memory
        clean_response = response.replace(MESSAGE_DELIMITER, "\n")
        try:
            await self.memory.store(
                channel_id,
                str(self.bot.user.id),
                self.bot.user.name,
                "assistant",
                clean_response,
                for_user_id=user_id,
                guild_id=guild_id,
                channel_name=channel_name,
            )
        except Exception:
            log.exception("chat_store_response_failed", channel_id=channel_id)

        # Fire background tasks: reactions, fact extraction
        if reactions:
            self._spawn_task(add_reactions(message, reactions))

        # Execute tool calls — reminders and channels run in background
        if llm_response.tool_calls:
            reminder_names = {"create_reminder", "list_reminders", "cancel_reminder"}
            reminder_calls = [tc for tc in llm_response.tool_calls if tc.name in reminder_names]
            other_calls = [tc for tc in llm_response.tool_calls if tc.name not in reminder_names]
            for rc in reminder_calls:
                self._spawn_task(self._execute_reminder_call(message, rc))
            if other_calls and message.guild:
                self._spawn_task(self._execute_tool_calls(message, other_calls))

        # Fallback: if LLM produced no text AND no reactions AND no tool calls, send in-character error
        # (tool calls run in background — empty text is expected when LLM only produced tool_use)
        if not response.strip() and not reactions and not llm_response.tool_calls:
            response = get_error_response("generic")

        # Send text response with [SEND] splitting, chunking, and typing delays
        has_side_effects = bool(reactions or llm_response.tool_calls)
        await send_response(message.channel, response, has_side_effects=has_side_effects)

        # Post-generation flow adherence validation (telemetry only)
        validate_flow_adherence(response, flow_analysis)

        # Background fact extraction
        self._spawn_task(self._extract_user_facts(user_id, user_name, user_facts, recent))

    # --- Private helpers ---

    async def _build_context(
        self, channel_id: str, text: str, attachment_blocks: list
    ) -> tuple[list[dict], list[dict]] | tuple[None, list]:
        """Build conversation context from memory. Returns (context, recent) or (None, []) on failure."""
        try:
            recent = await self.memory.get_recent(channel_id, self.settings.memory_recent_limit)
            relevant = await self.memory.search(channel_id, text, self.settings.memory_relevant_limit)
            context = self.memory.build_context(recent, relevant)
        except Exception:
            log.exception("chat_context_failed", channel_id=channel_id)
            return None, []

        if attachment_blocks and context:
            last_msg = context[-1]
            if last_msg["role"] == "user":
                text_block = {"type": "text", "text": last_msg["content"]}
                context[-1] = {"role": "user", "content": [text_block, *attachment_blocks]}

        return context, recent

    async def _load_facts(self, user_id: str) -> list[dict]:
        """Load user facts, returning empty list on failure."""
        try:
            return await self.memory.get_facts(user_id)
        except Exception:
            log.exception("chat_facts_load_failed", user_id=user_id)
            return []

    async def _load_facts_smart(self, user_id: str, query: str) -> list[dict]:
        """Load user facts with semantic search when there are many facts.

        Uses vector similarity search to return only the most relevant facts
        for the current message. Falls back to all facts if semantic search
        is unavailable or there are few facts.
        """
        try:
            all_facts = await self.memory.get_facts(user_id)
        except Exception:
            log.exception("chat_facts_load_failed", user_id=user_id)
            return []

        # Only use semantic search if there are enough facts to warrant filtering
        if len(all_facts) > 5:
            try:
                return await self.memory.search_facts_semantic(user_id, query, limit=10)
            except Exception:
                log.exception("chat_facts_semantic_failed", user_id=user_id)
                return all_facts

        return all_facts

    async def _transcribe_voice(self, message: discord.Message) -> str | None:
        """Transcribe a Discord voice message via Azure OpenAI Whisper."""
        from insult.core.transcribe import transcribe_voice_message

        try:
            audio_data = await message.attachments[0].read()
            return await transcribe_voice_message(
                audio_data,
                endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_key.get_secret_value(),
                deployment=self.settings.azure_openai_whisper_deployment,
            )
        except Exception:
            log.exception("voice_transcription_failed")
            return None

    def _spawn_task(self, coro) -> None:
        """Create a background task with automatic cleanup."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _execute_reminder_call(self, message: discord.Message, tool_call) -> None:
        """Execute a reminder tool call (create, list, or cancel)."""
        try:
            if tool_call.name == "create_reminder":
                description = tool_call.input.get("description", "")
                remind_at_str = tool_call.input.get("remind_at", "")
                mention_ids = tool_call.input.get("mention_user_ids", [])
                recurring = tool_call.input.get("recurring", "none")

                remind_at = parse_remind_at(remind_at_str)
                if remind_at is None:
                    log.warning("reminder_invalid_time", remind_at=remind_at_str)
                    return

                mention_str = ",".join(mention_ids) if mention_ids else ""
                guild_id = str(message.guild.id) if message.guild else None

                reminder_id = await self.memory.save_reminder(
                    channel_id=str(message.channel.id),
                    guild_id=guild_id,
                    created_by=str(message.author.id),
                    description=description,
                    remind_at=remind_at,
                    mention_user_ids=mention_str,
                    recurring=recurring,
                )
                log.info(
                    "reminder_created",
                    reminder_id=reminder_id,
                    description=description[:80],
                    remind_at=remind_at_str,
                    recurring=recurring,
                )

            elif tool_call.name == "list_reminders":
                channel_id = tool_call.input.get("channel_id", str(message.channel.id))
                reminders = await self.memory.get_channel_reminders(channel_id)
                formatted = format_reminder_list(reminders)
                await message.channel.send(formatted)

            elif tool_call.name == "cancel_reminder":
                reminder_id = tool_call.input.get("reminder_id", 0)
                deleted = await self.memory.delete_reminder(reminder_id)
                if not deleted:
                    log.warning("reminder_cancel_not_found", reminder_id=reminder_id)

        except Exception:
            log.exception("reminder_tool_call_failed", tool=tool_call.name)

    async def _execute_tool_calls(self, message: discord.Message, tool_calls: list):
        """Background task: execute tool_use calls from Claude (channel creation, etc.)."""
        for tool_call in tool_calls:
            try:
                if tool_call.name == "create_channel" and message.guild:
                    channel = await execute_create_channel(message.guild, tool_call, message.author)
                    if channel:
                        await message.channel.send(f"Listo, ahí está: {channel.mention}")
                        self._spawn_task(
                            self._inaugurate_channel(channel, tool_call.input.get("name", ""), message.author)
                        )
                    else:
                        log.warning("tool_call_returned_none", tool=tool_call.name)

                elif tool_call.name == "get_channel_info" and isinstance(message.channel, discord.TextChannel):
                    info = execute_get_channel_info(message.channel)
                    topic_display = info["topic"] or "(sin descripción)"
                    await message.channel.send(f"**#{info['name']}** — {topic_display}")

                elif tool_call.name == "edit_channel" and isinstance(message.channel, discord.TextChannel):
                    success = await execute_edit_channel(message.channel, tool_call)
                    if not success:
                        log.warning("tool_call_edit_failed", tool=tool_call.name)

                else:
                    log.warning("tool_call_unknown", tool=tool_call.name)
            except Exception:
                log.exception("tool_call_execution_failed", tool=tool_call.name)

    async def _inaugurate_channel(self, channel: discord.abc.GuildChannel, channel_name: str, creator: discord.Member):
        """Background task: generate a philosophical opening message for a newly created channel."""
        from insult.core.character import _get_current_time_context

        time_ctx = _get_current_time_context()

        # Load facts about the creator for personalization
        creator_facts = await self._load_facts(str(creator.id))
        facts_str = ", ".join(f["fact"] for f in creator_facts) if creator_facts else "no los conozco todavia"

        inaugural_prompt = (
            f"{self.settings.system_prompt}\n\n"
            f"## Current Time\n{time_ctx}\n\n"
            "## Special Task: Channel Inauguration\n"
            f"You just created a new channel called #{channel_name}. "
            f"The person who asked for it is {creator.display_name}. "
            f"What you know about them: {facts_str}.\n\n"
            "Write an opening message for this channel. This is YOUR territory — make it count.\n\n"
            "Requirements:\n"
            "- Open with something philosophical, provocative, or deeply interesting about the channel's topic\n"
            "- Ask 2-3 questions that would make people WANT to participate — uncomfortable, interesting, real questions\n"
            "- Keep your personality: sharp, system-critical, anti-domination, curious, never bland\n"
            "- Reference the time of day, your mood, the creator — make it feel alive, not templated\n"
            "- If the topic relates to ethics, power, systems, animals, capitalism — go deeper\n"
            "- DO NOT use markdown headers, bullet points, or structured formatting. Talk like a person.\n"
            "- Length: medium to long. This is a manifesto for the channel, not a tweet.\n"
            "- You can use [SEND] to split into multiple messages for dramatic effect.\n"
            "- DO NOT call any tools. Just write the text directly.\n"
        )

        try:
            llm_response = await self.llm.chat(
                inaugural_prompt,
                [{"role": "user", "content": f"Inaugura el canal #{channel_name}"}],
            )
            text = llm_response.text
            if text:
                await send_response(channel, text)
                log.info("channel_inaugurated", channel=channel_name, length=len(text))
        except Exception:
            log.exception("channel_inauguration_failed", channel=channel_name)

    async def _extract_user_facts(self, user_id: str, user_name: str, existing_facts: list[dict], recent: list[dict]):
        """Background task: extract user facts from recent conversation."""
        try:
            new_facts = await extract_facts(self.llm.client, self.settings.llm_model, user_name, existing_facts, recent)
            if new_facts != existing_facts:
                await self.memory.save_facts(user_id, new_facts)
        except Exception:
            log.exception("facts_extraction_background_failed", user_id=user_id)
