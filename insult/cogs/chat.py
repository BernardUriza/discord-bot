"""Chat cog — responds to all messages, no prefix needed."""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import discord
import structlog
from discord.ext import commands

from insult.core.actions import CHANNEL_TOOLS, execute_create_channel
from insult.core.attachments import process_attachments
from insult.core.character import build_adaptive_prompt
from insult.core.errors import classify_error, get_error_response
from insult.core.facts import build_facts_prompt, extract_facts
from insult.core.presets import PresetModifier

if TYPE_CHECKING:
    from insult.app import Container

log = structlog.get_logger()

MAX_MESSAGE_LENGTH = 4000
BATCH_WAIT_SECONDS = 3.0  # Wait this long after last message before responding
MIN_RESPONSE_GAP = 5.0  # Minimum seconds between bot responses to same user (token protection)
MESSAGE_DELIMITER = "[SEND]"
TYPING_CHARS_PER_SECOND = 50  # ~250 CPM, fast mobile typing speed
MIN_TYPING_DELAY = 0.8
MAX_TYPING_DELAY = 5.0
VERSION_TAG = "ᵇᵉᵗᵃ ᵛ⁰·⁸·⁰"  # superscript unicode — visible but unobtrusive

# Reaction system — [REACT:💀,🔥] parsed from LLM response
REACTION_PATTERN = re.compile(r"\[REACT:([^\]]*)\]", re.IGNORECASE)
MAX_REACTIONS = 3
REACTION_DELAY_MIN = 0.5  # seconds before first reaction (human-like pause)
REACTION_DELAY_MAX = 2.0
REACTION_INTERVAL = 0.35  # seconds between multiple reactions (rate limit safety)


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

        # Ignore empty messages (e.g. only attachments with no text, stickers)
        text = message.content.strip()
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
        # Record response time for token protection
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

        try:
            await self.memory.store(channel_id, user_id, user_name, "user", text)
        except Exception:
            log.exception("chat_store_user_failed", channel_id=channel_id)

        try:
            profile = await self.memory.update_profile(user_id, text)
        except Exception:
            log.exception("chat_profile_update_failed", user_id=user_id)
            profile = None

        try:
            # Full channel context (all users) so the bot sees the whole conversation flow.
            # Style adaptation is per-user (via profile), but context must be shared.
            recent = await self.memory.get_recent(channel_id, self.settings.memory_recent_limit)
            relevant = await self.memory.search(channel_id, text, self.settings.memory_relevant_limit)
            context = self.memory.build_context(recent, relevant)
        except Exception:
            log.exception("chat_context_failed", channel_id=channel_id)
            await message.channel.send(get_error_response("context_failed"))
            return

        if attachment_blocks and context:
            last_msg = context[-1]
            if last_msg["role"] == "user":
                text_block = {"type": "text", "text": last_msg["content"]}
                context[-1] = {"role": "user", "content": [text_block, *attachment_blocks]}

        # Load user facts for injection into prompt
        user_facts = []
        try:
            user_facts = await self.memory.get_facts(user_id)
        except Exception:
            log.exception("chat_facts_load_failed", user_id=user_id)

        system_prompt, preset = build_adaptive_prompt(
            self.settings.system_prompt,
            profile,
            len(context),
            current_message=text,
            recent_messages=recent,
            user_facts=user_facts,
        )
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

        # If user wants a server action, force tool_choice="any" so Claude MUST use the tool.
        # With "any", Claude won't emit text — we generate confirmation ourselves.
        force_tool = PresetModifier.ACTION_INTENT in preset.modifiers
        tool_choice = {"type": "any"} if force_tool else None

        try:
            async with message.channel.typing():
                llm_response = await self.llm.chat(system_prompt, context, tools=CHANNEL_TOOLS, tool_choice=tool_choice)
        except Exception as e:
            log.exception("chat_llm_failed", channel_id=channel_id, error_type=type(e).__name__)
            await message.channel.send(get_error_response(classify_error(e)))
            return

        response = llm_response.text

        # Extract emoji reactions from text (cosmetic — text markers are fine for this)
        reactions = parse_reactions(response)
        response = strip_reactions(response)

        # Store the full response (without delimiters) in memory
        clean_response = response.replace(MESSAGE_DELIMITER, "\n")
        try:
            await self.memory.store(
                channel_id, str(self.bot.user.id), self.bot.user.name, "assistant", clean_response, for_user_id=user_id
            )
        except Exception:
            log.exception("chat_store_response_failed", channel_id=channel_id)

        # Fire emoji reactions in background (on the USER's message, with human-like delay)
        if reactions:
            task = asyncio.create_task(self._add_reactions(message, reactions))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Execute tool_use actions in background (channel creation, etc.)
        if llm_response.tool_calls and message.guild:
            task = asyncio.create_task(self._execute_tool_calls(message, llm_response.tool_calls))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Send as multiple messages with typing delay if LLM used [SEND] delimiter
        parts = [p.strip() for p in response.split(MESSAGE_DELIMITER) if p.strip()]
        if not parts:
            parts = [] if (reactions or llm_response.tool_calls) else [response.strip() or "..."]
        for i, part in enumerate(parts):
            # Append version tag to the very last chunk of the last part
            is_last_part = i == len(parts) - 1
            chunks = [part[j : j + 1990] for j in range(0, len(part), 1990)]
            for ci, chunk in enumerate(chunks):
                if is_last_part and ci == len(chunks) - 1:
                    chunk += f"\n-# {VERSION_TAG}"
                await message.channel.send(chunk)
            # Typing delay between parts (not after the last one)
            if not is_last_part:
                next_part = parts[i + 1]
                delay = max(MIN_TYPING_DELAY, min(len(next_part) / TYPING_CHARS_PER_SECOND, MAX_TYPING_DELAY))
                async with message.channel.typing():
                    await asyncio.sleep(delay)

        # Async fact extraction — runs in background, doesn't block response
        task = asyncio.create_task(self._extract_user_facts(user_id, user_name, user_facts, recent))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _execute_tool_calls(self, message: discord.Message, tool_calls: list):
        """Background task: execute tool_use calls from Claude (channel creation, etc.)."""
        for tool_call in tool_calls:
            if tool_call.name == "create_channel" and message.guild:
                try:
                    channel = await execute_create_channel(message.guild, tool_call, message.author)
                    if channel:
                        await message.channel.send(f"Listo, ahí está: {channel.mention}")
                    else:
                        log.warning("tool_call_returned_none", tool=tool_call.name)
                except Exception:
                    log.exception("tool_call_execution_failed", tool=tool_call.name)
            else:
                log.warning("tool_call_unknown", tool=tool_call.name)

    async def _add_reactions(self, message: discord.Message, emojis: list[str]):
        """Background task: add emoji reactions to the user's message with human-like delay."""
        try:
            # Initial delay — humans don't react instantly
            await asyncio.sleep(random.uniform(REACTION_DELAY_MIN, REACTION_DELAY_MAX))

            for i, emoji in enumerate(emojis):
                try:
                    await message.add_reaction(emoji)
                    log.info("reaction_added", emoji=emoji, message_id=message.id)
                except (discord.HTTPException, discord.NotFound) as e:
                    log.warning("reaction_failed", emoji=emoji, error=str(e))
                    break  # Don't try remaining if one fails
                # Small delay between multiple reactions (rate limit safety)
                if i < len(emojis) - 1:
                    await asyncio.sleep(REACTION_INTERVAL)
        except Exception:
            log.exception("reaction_task_failed", message_id=message.id)

    async def _extract_user_facts(self, user_id: str, user_name: str, existing_facts: list[dict], recent: list[dict]):
        """Background task: extract user facts from recent conversation."""
        try:
            new_facts = await extract_facts(self.llm.client, self.settings.llm_model, user_name, existing_facts, recent)
            if new_facts != existing_facts:
                await self.memory.save_facts(user_id, new_facts)
        except Exception:
            log.exception("facts_extraction_background_failed", user_id=user_id)


def parse_reactions(response: str) -> list[str]:
    """Extract emoji reactions from LLM response.

    Parses [REACT:emoji1,emoji2] markers and returns a list of emoji strings.
    Returns at most MAX_REACTIONS emojis. Returns empty list if no marker found.
    """
    match = REACTION_PATTERN.search(response)
    if not match:
        return []

    raw = match.group(1).strip()
    if not raw:
        return []

    # Split by comma, strip whitespace, filter empty
    emojis = [e.strip() for e in raw.split(",") if e.strip()]
    return emojis[:MAX_REACTIONS]


def strip_reactions(response: str) -> str:
    """Remove [REACT:...] markers from the response text."""
    return REACTION_PATTERN.sub("", response).strip()
