"""Chat cog — responds to all messages, no prefix needed."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import discord
import structlog
from discord.ext import commands

from insult.core.attachments import process_attachments
from insult.core.character import build_adaptive_prompt
from insult.core.errors import classify_error, get_error_response
from insult.core.facts import build_facts_prompt, extract_facts

if TYPE_CHECKING:
    from insult.app import Container

log = structlog.get_logger()

MAX_MESSAGE_LENGTH = 4000
COOLDOWN_SECONDS = 15
MESSAGE_DELIMITER = "[SEND]"
TYPING_CHARS_PER_SECOND = 50  # ~250 CPM, fast mobile typing speed
MIN_TYPING_DELAY = 0.8
MAX_TYPING_DELAY = 5.0
VERSION_TAG = "ᵇᵉᵗᵃ ᵛ⁰·³·¹"  # superscript unicode — visible but unobtrusive


class ChatCog(commands.Cog):
    def __init__(self, container: Container):
        self.memory = container.memory
        self.llm = container.llm
        self.settings = container.settings
        self.bot = container.bot
        self._cooldowns: dict[int, float] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._processed_messages: set[int] = set()  # dedup: prevent double responses
        self._processed_max = 1000  # rotate after this many to prevent memory leak

    def _check_cooldown(self, user_id: int) -> float:
        """Returns 0 if ready, or seconds remaining."""
        now = time.monotonic()
        last = self._cooldowns.get(user_id, 0)
        remaining = COOLDOWN_SECONDS - (now - last)
        if remaining <= 0:
            self._cooldowns[user_id] = now
            return 0
        return remaining

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Respond to every message — no !chat prefix needed."""
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

        # Per-user cooldown
        remaining = self._check_cooldown(message.author.id)
        if remaining > 0:
            await message.channel.send(f"Calmate, espera {remaining:.0f}s. Cual es la urgencia?")
            return

        await self._respond(message, text)

    @commands.command(name="chat")
    @commands.cooldown(1, COOLDOWN_SECONDS, commands.BucketType.user)
    async def chat(self, ctx: commands.Context, *, message: str):
        """Fallback: !chat still works for explicit invocation."""
        await self._respond(ctx.message, message)

    async def _respond(self, message: discord.Message, text: str):
        """Core response logic — shared by on_message and !chat command."""
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

        system_prompt = build_adaptive_prompt(self.settings.system_prompt, profile, len(context))
        system_prompt += build_facts_prompt(user_name, user_facts)

        if profile and profile.is_confident:
            log.info(
                "style_adapted",
                user_id=user_id,
                language=profile.detected_language,
                formality=round(profile.formality, 2),
                technical=round(profile.technical_level, 2),
                verbosity=round(profile.avg_word_count, 1),
            )

        try:
            async with message.channel.typing():
                response = await self.llm.chat(system_prompt, context)
        except Exception as e:
            log.exception("chat_llm_failed", channel_id=channel_id, error_type=type(e).__name__)
            await message.channel.send(get_error_response(classify_error(e)))
            return

        # Store the full response (without delimiters) in memory
        clean_response = response.replace(MESSAGE_DELIMITER, "\n")
        try:
            await self.memory.store(
                channel_id, str(self.bot.user.id), self.bot.user.name, "assistant", clean_response, for_user_id=user_id
            )
        except Exception:
            log.exception("chat_store_response_failed", channel_id=channel_id)

        # Send as multiple messages with typing delay if LLM used [SEND] delimiter
        parts = [p.strip() for p in response.split(MESSAGE_DELIMITER) if p.strip()]
        if not parts:
            parts = [response.strip() or "..."]
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

    async def _extract_user_facts(self, user_id: str, user_name: str, existing_facts: list[dict], recent: list[dict]):
        """Background task: extract user facts from recent conversation."""
        try:
            new_facts = await extract_facts(self.llm.client, self.settings.llm_model, user_name, existing_facts, recent)
            if new_facts != existing_facts:
                await self.memory.save_facts(user_id, new_facts)
        except Exception:
            log.exception("facts_extraction_background_failed", user_id=user_id)
