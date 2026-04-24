"""Chat cog — responds to all messages, no prefix needed."""

from __future__ import annotations

import asyncio
import secrets
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
from insult.core.arc_tracker import ArcState, arc_from_dict, arc_to_dict, build_arc_prompt, update_arc
from insult.core.attachments import process_attachments
from insult.core.character import (
    build_adaptive_prompt,
    deduplicate_opener,
    enforce_length_variation,
    strip_echoed_quotes,
)
from insult.core.delivery import MESSAGE_DELIMITER, send_response
from insult.core.disclosure import scan_disclosure
from insult.core.errors import classify_error, get_error_response
from insult.core.facts import build_facts_prompt, extract_facts
from insult.core.flows import ExpressionHistory, analyze_flows, build_flow_prompt, validate_flow_adherence
from insult.core.guild_setup import post_facts_to_channel, post_reminder_set
from insult.core.image_summary import summarize_images
from insult.core.llm import MEDICAL_WEB_SEARCH_TOOL, WEB_SEARCH_TOOL
from insult.core.presets import PresetMode, PresetModifier
from insult.core.reactions import add_reactions, parse_reactions, strip_reactions
from insult.core.reminders import REMINDER_TOOLS, format_reminder_list, parse_remind_at
from insult.core.routing import ModelTier, OpusBudget, select_model
from insult.core.summaries import build_server_pulse, filter_by_permissions
from insult.core.triviality import is_trivial

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
        # Opus budget counter — per-user rolling 24h, used by the 3-tier router
        self._opus_budget = OpusBudget(cap=getattr(self.settings, "opus_24h_cap", 20))

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
            log.debug("msg_dropped_duplicate", message_id=message.id, user_id=message.author.id)
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
            log.debug(
                "msg_dropped_empty",
                message_id=message.id,
                user_id=message.author.id,
                attachments=len(message.attachments),
            )
            return

        # Reset proactive backoff: a user spoke, so proactive timing resets
        if hasattr(self.bot, "_reset_proactive_backoff"):
            self.bot._reset_proactive_backoff()

        if len(text) > MAX_MESSAGE_LENGTH:
            log.info(
                "msg_dropped_too_long",
                message_id=message.id,
                user_id=message.author.id,
                text_len=len(text),
                limit=MAX_MESSAGE_LENGTH,
            )
            await message.channel.send(get_error_response("too_long"))
            return

        # Token protection: hard minimum gap between bot responses per user
        now = time.monotonic()
        last_response = self._last_response_time.get(message.author.id, 0)
        if now - last_response < MIN_RESPONSE_GAP:
            log.info(
                "msg_dropped_cooldown",
                message_id=message.id,
                user_id=message.author.id,
                channel_id=message.channel.id,
                since_last_response_s=round(now - last_response, 2),
                min_gap_s=MIN_RESPONSE_GAP,
            )
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

        log.debug(
            "msg_batched",
            batch_key=batch_key,
            batch_size=len(batch.messages),
            text_len=len(text),
        )

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
            log.debug("batch_flush_empty", batch_key=batch_key)
            return

        # Use the LAST message as the target for reactions and replies
        last_message = batch.messages[-1]
        # Combine all texts into one (preserving order)
        combined_text = "\n".join(batch.texts)

        log.debug(
            "batch_flush_start",
            batch_key=batch_key,
            batch_size=len(batch.messages),
            combined_len=len(combined_text),
        )

        await self._respond(last_message, combined_text)

    @commands.command(name="chat")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def chat(self, ctx: commands.Context, *, message: str):
        """Fallback: !chat still works for explicit invocation."""
        await self._respond(ctx.message, message)

    async def _respond(self, message: discord.Message, text: str):
        """Core response logic — shared by on_message and !chat command.

        A `request_id` is bound to structlog contextvars at the start of the
        turn so every log downstream (context build, preset, flows, llm.py,
        delivery.py, background tasks spawned from here) tags its events with
        the same id. `grep request_id=<id>` then reconstructs the full turn.

        A terminal `chat_turn_end` event is ALWAYS emitted before the binding
        is cleared — even on early-return paths (trivial / context_failed /
        llm_failed / delivery_failed / unhandled exception) — so that any
        request_id grep produces exactly one start and exactly one end line.
        """
        request_id = secrets.token_hex(4)
        channel_id = str(message.channel.id)
        user_id = str(message.author.id)
        user_name = message.author.display_name
        turn_start = time.monotonic()

        # Bind once — every downstream log in this task inherits these.
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            channel_id=channel_id,
            user_id=user_id,
        )
        outcome = "unknown"
        try:
            outcome = await self._respond_inner(message, text, user_name, channel_id, user_id, turn_start)
        except BaseException as e:
            outcome = f"unhandled:{type(e).__name__}"
            log.exception("chat_turn_unhandled_exception", error_type=type(e).__name__)
            raise
        finally:
            log.info(
                "chat_turn_end",
                outcome=outcome,
                total_ms=int((time.monotonic() - turn_start) * 1000),
            )
            structlog.contextvars.unbind_contextvars("request_id", "channel_id", "user_id")

    async def _respond_inner(
        self,
        message: discord.Message,
        text: str,
        user_name: str,
        channel_id: str,
        user_id: str,
        turn_start: float,
    ) -> str:
        """Actual response logic. Called under bound contextvars.

        Returns an outcome string consumed by `_respond` for the terminal
        `chat_turn_end` log. Possible values:
        - "ok"                — LLM responded and delivery succeeded
        - "trivial_skipped"   — message matched triviality rules, no LLM call
        - "context_failed"    — memory read failed, user notified
        - "too_long_skipped"  — rejected pre-LLM (set by on_message, kept for completeness)
        - "llm_failed"        — LLM client raised after retries exhausted
        - "delivery_failed"   — Discord rejected the send
        - "delivery_crashed"  — unexpected exception in delivery path
        """
        self._last_response_time[message.author.id] = turn_start

        log.info(
            "chat_turn_start",
            text_len=len(text),
            text_preview=text[:120],
            attachments=len(message.attachments),
            is_voice=bool(message.flags.voice),
            guild_id=str(message.guild.id) if message.guild else None,
            channel_name=message.channel.name if hasattr(message.channel, "name") else None,
        )

        def _stage_elapsed() -> int:
            return int((time.monotonic() - turn_start) * 1000)

        # Process attachments
        attachment_blocks = []
        if message.attachments:
            blocks, errors = await process_attachments(message.attachments)
            attachment_blocks = blocks
            for err in errors:
                await message.channel.send(err)
            log.info(
                "stage_attachments_processed",
                blocks=len(attachment_blocks),
                errors=len(errors),
                elapsed_ms=_stage_elapsed(),
            )

        # Generate a short text description of any image attachments so future
        # turns can reference them from SQLite context (which is text only).
        # The main LLM call still sees the raw image blocks below.
        text_for_memory = text
        image_blocks = [b for b in attachment_blocks if isinstance(b, dict) and b.get("type") == "image"]
        if image_blocks:
            summary_started = time.monotonic()
            try:
                summary = await summarize_images(
                    image_blocks,
                    client=self.llm.client,
                    model=self.settings.summary_model,
                )
                if summary:
                    text_for_memory = f"{text}\n[Imagen: {summary}]" if text else f"[Imagen: {summary}]"
                log.info(
                    "image_summary_ok",
                    duration_ms=int((time.monotonic() - summary_started) * 1000),
                    image_count=len(image_blocks),
                    summary_len=len(summary or ""),
                    summary_preview=(summary or "")[:120],
                )
            except Exception:
                log.exception(
                    "image_summary_wrapper_failed",
                    duration_ms=int((time.monotonic() - summary_started) * 1000),
                    image_count=len(image_blocks),
                )

        guild_id = str(message.guild.id) if message.guild else None
        channel_name = message.channel.name if hasattr(message.channel, "name") else None

        try:
            await self.memory.store(
                channel_id,
                user_id,
                user_name,
                "user",
                text_for_memory,
                guild_id=guild_id,
                channel_name=channel_name,
            )
        except Exception:
            log.exception("chat_store_user_failed", channel_id=channel_id)

        # Update style profile BEFORE the trivial gate so short-message users
        # still accumulate signal for language/formality/emoji detection —
        # otherwise profiles of users who mostly send "ok/gracias/jaja"
        # stay under the is_confident threshold forever.
        try:
            profile = await self.memory.update_profile(user_id, text)
        except Exception:
            log.exception("chat_profile_update_failed", user_id=user_id)
            profile = None

        log.info(
            "stage_memory_stored",
            elapsed_ms=_stage_elapsed(),
            profile_confident=bool(profile and profile.is_confident),
        )

        # Cost guard: skip LLM call for trivial messages (ok, gracias, lone emojis, etc.)
        # Message is already stored + profile already updated above.
        if not message.attachments and is_trivial(text):
            log.info("skipped_trivial_message", text=text[:40])
            return "trivial_skipped"

        context, recent = await self._build_context(channel_id, text, attachment_blocks)
        if context is None:
            log.warning("chat_turn_aborted", reason="context_failed")
            await message.channel.send(get_error_response("context_failed"))
            return "context_failed"

        log.info(
            "stage_context_built",
            context_len=len(context),
            recent_count=len(recent),
            elapsed_ms=_stage_elapsed(),
        )

        # Load user facts — use semantic search for relevance when many facts exist
        user_facts = await self._load_facts_smart(user_id, text)
        log.info("stage_facts_loaded", facts_count=len(user_facts), elapsed_ms=_stage_elapsed())

        # Load facts for ALL other participants in channel (group chat awareness)
        other_participants_facts: dict[str, list[dict]] = {}
        try:
            participants = await self.memory.get_channel_participants(channel_id, limit=10)
            for p in participants:
                if p["user_id"] != user_id:  # skip current speaker
                    facts = await self.memory.get_facts(p["user_id"])
                    if facts:
                        other_participants_facts[p["user_name"]] = facts[:5]
        except Exception:
            log.exception("chat_participants_facts_failed")

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

        # --- Phase 1 (v3.0.0): Pre-LLM disclosure scan ---
        disclosure = scan_disclosure(text)
        if disclosure.detected:
            import json as _json

            await self.memory.store_disclosure(
                channel_id,
                user_id,
                disclosure.category,
                disclosure.severity,
                _json.dumps(disclosure.signals),
                text[:200],
            )

        # --- Phase 1 (v3.0.0): Load emotional arc state ---
        arc_data = await self.memory.get_arc(channel_id, user_id)
        arc_state = arc_from_dict(arc_data) if arc_data else ArcState()

        # Compute recent response lengths for Fix #3 (length variation hint)
        recent_response_lengths = [len(m.get("content", "").split()) for m in recent if m["role"] == "assistant"][-5:]

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
            recent_response_lengths=recent_response_lengths,
        )
        # Always log the preset — the old `style_adapted` only fired for confident profiles,
        # so new users with <5 messages never showed a preset in logs.
        log.info(
            "preset_classified",
            preset=preset.mode.value,
            modifiers=[m.value for m in preset.modifiers],
            disclosure_severity=disclosure.severity,
            disclosure_category=disclosure.category,
            arc_phase=arc_state.phase,
            elapsed_ms=_stage_elapsed(),
        )

        # Run 4-flow behavioral analysis
        flow_analysis = analyze_flows(text, recent, preset, self._expression_history, context_key)
        log.info(
            "stage_flows_analyzed",
            pressure=flow_analysis.pressure.pressure_level,
            user_state=flow_analysis.pressure.detected_state.value,
            shape=flow_analysis.expression.selected_shape.value,
            flavor=flow_analysis.expression.selected_flavor.value,
            awareness=flow_analysis.awareness.detected_pattern.value,
            epistemic_move=flow_analysis.epistemic.recommended_move.value,
            agreement_streak=flow_analysis.agreement_streak,
            elapsed_ms=_stage_elapsed(),
        )
        # Inject flow guidance as Layer 3.5 (appended after preset, before facts)
        flow_prompt = build_flow_prompt(flow_analysis)
        if flow_prompt:
            system_prompt += f"\n\n{flow_prompt}"

        # --- Phase 1 (v3.0.0): Inject arc + stance guidance ---
        arc_prompt = build_arc_prompt(arc_state)
        if arc_prompt:
            system_prompt += f"\n\n{arc_prompt}"
        stances = await self.memory.get_stances(channel_id, user_id, limit=5)
        if stances:
            from insult.core.stance_log import build_stance_prompt

            stance_prompt = build_stance_prompt(stances)
            if stance_prompt:
                system_prompt += f"\n\n{stance_prompt}"

        system_prompt += build_facts_prompt(user_name, user_facts)

        # Inject other participants' facts (group chat awareness)
        if other_participants_facts:
            parts = ["## Other People in This Channel (they are REAL — never say they don't exist)"]
            for name, facts in other_participants_facts.items():
                fact_lines = ", ".join(f["fact"] for f in facts[:3])
                parts.append(f"- {name}: {fact_lines}")
            system_prompt += "\n\n" + "\n".join(parts)

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

        # Build tools list: static tools + a web_search variant.
        # For RESPECTFUL_SERIOUS (clinical topics / vulnerable users) the
        # search tool is DOMAIN-RESTRICTED to authoritative medical sources
        # (MedlinePlus, AEMPS CIMA, NIH, WHO). This exists because a bot
        # answering medication or diagnosis questions with a random blog
        # post is a documented harm mode (APA Health Advisory) — so we
        # swap the tool definition rather than disable search entirely,
        # which used to leave the model with no way to verify pharmacology.
        tools = list(_ALL_TOOLS)
        if preset.mode == PresetMode.RESPECTFUL_SERIOUS:
            tools.append(MEDICAL_WEB_SEARCH_TOOL)
        else:
            tools.append(WEB_SEARCH_TOOL)

        # If user wants a server action, force tool_choice="any" so Claude MUST use the tool.
        force_tool = PresetModifier.ACTION_INTENT in preset.modifiers
        tool_choice = {"type": "any"} if force_tool else None

        # 3-tier router: pick primary/fallback models. Gated by feature flag —
        # when disabled the bot uses self.llm.model for every turn (legacy path).
        model_choice = None
        if getattr(self.settings, "model_router_enabled", False):
            model_choice = select_model(
                preset,
                flow_analysis,
                disclosure.severity,
                casual_model=self.settings.casual_model,
                depth_model=self.settings.llm_model,
                crisis_model=self.settings.crisis_model,
                opus_24h_count=self._opus_budget.count(user_id),
                opus_24h_cap=self._opus_budget.cap,
            )
            log.info(
                "model_routed",
                tier=model_choice.tier.value,
                primary=model_choice.primary,
                fallback=model_choice.fallback,
                reason=model_choice.reason,
                preset=preset.mode.value,
                disclosure_severity=disclosure.severity,
                user_id=user_id,
            )

        async def _notify_retry():
            notify_start = time.monotonic()
            try:
                await message.channel.send(get_error_response("retry_notice"))
                log.info(
                    "retry_notice_sent",
                    delivery_ms=int((time.monotonic() - notify_start) * 1000),
                )
            except Exception:
                log.exception(
                    "retry_notice_send_failed",
                    delivery_ms=int((time.monotonic() - notify_start) * 1000),
                )

        llm_start = time.monotonic()
        log.info(
            "llm_call_start",
            prompt_chars=len(system_prompt),
            context_messages=len(context),
            tools=[t.get("name", t.get("type", "?")) for t in tools],
            tool_choice=(tool_choice or {}).get("type"),
            primary_model=model_choice.primary if model_choice else self.settings.llm_model,
            fallback_model=model_choice.fallback if model_choice else None,
        )
        try:
            async with message.channel.typing():
                llm_kwargs = {"tools": tools, "tool_choice": tool_choice, "on_timeout": _notify_retry}
                if model_choice is not None:
                    llm_kwargs["model"] = model_choice.primary
                    llm_kwargs["fallback_model"] = model_choice.fallback
                llm_response = await self.llm.chat(system_prompt, context, **llm_kwargs)
        except Exception as e:
            log.exception(
                "chat_llm_failed",
                error_type=type(e).__name__,
                error_msg=str(e)[:200],
                llm_ms=int((time.monotonic() - llm_start) * 1000),
            )
            await message.channel.send(get_error_response(classify_error(e)))
            return "llm_failed"

        llm_ms = int((time.monotonic() - llm_start) * 1000)
        log.info(
            "llm_call_complete",
            llm_ms=llm_ms,
            text_len=len(llm_response.text),
            tool_calls=len(llm_response.tool_calls),
            tool_names=[tc.name for tc in llm_response.tool_calls],
            model_used=llm_response.model_used,
            elapsed_ms=_stage_elapsed(),
        )

        # Opus budget: record only after a successful call so transient failures
        # don't spend the user's 24h cap on requests that never produced output.
        if model_choice is not None and model_choice.tier == ModelTier.CRISIS:
            self._opus_budget.record(user_id)

        response = llm_response.text
        post_llm_len = len(response)

        # Each of the post-generation passes below can silently mangle output.
        # Log whenever a pass actually changes length so "responde raro" turns
        # can be traced to the exact transformation that mutated the text.
        def _log_mutation(stage: str, before: str, after: str) -> None:
            if before != after:
                log.info(
                    "text_mutated",
                    stage=stage,
                    before_len=len(before),
                    after_len=len(after),
                    delta=len(after) - len(before),
                )

        # Anti-parrot: strip verbatim quotes of user's words from response
        before = response
        response = strip_echoed_quotes(response, text)
        _log_mutation("strip_echoed_quotes", before, response)

        # Length enforcer — truncate if 3+ consecutive mediums
        before = response
        response = enforce_length_variation(response, recent_response_lengths)
        _log_mutation("enforce_length_variation", before, response)

        # Fix #4: Opener deduplication — strip repetitive openers
        recent_openers = [m["content"].split("\n")[0] for m in recent if m["role"] == "assistant"][-5:]
        before = response
        response = deduplicate_opener(response, recent_openers)
        _log_mutation("deduplicate_opener", before, response)

        # Extract and strip emoji reactions
        reactions = parse_reactions(response)
        before = response
        response = strip_reactions(response)
        _log_mutation("strip_reactions", before, response)

        log.info(
            "stage_post_llm_done",
            raw_llm_len=post_llm_len,
            final_text_len=len(response),
            reactions=reactions,
            elapsed_ms=_stage_elapsed(),
        )

        # Store the full response in memory. Skip if the turn produced NO text
        # (reaction-only responses) — writing empty rows pollutes the conversation
        # context the next build pulls in, and the reaction itself is a side
        # effect on the USER'S message, not a conversational turn of its own.
        clean_response = response.replace(MESSAGE_DELIMITER, "\n")
        if clean_response.strip():
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
                    model_used=llm_response.model_used or None,
                )
            except Exception:
                log.exception("chat_store_response_failed", channel_id=channel_id)

        # --- Phase 1 (v3.0.0): Update arc state + extract stances ---
        new_arc = update_arc(
            arc_state,
            disclosure_severity=disclosure.severity,
            user_state=flow_analysis.pressure.detected_state.value,
            preset_mode=preset.mode.value,
        )
        arc_dict = arc_to_dict(new_arc)
        await self.memory.upsert_arc(
            channel_id,
            user_id,
            arc_dict["phase"],
            arc_dict["phase_since"],
            arc_dict["crisis_depth"],
            arc_dict["recovery_signals"],
            arc_dict["turns_in_phase"],
        )
        # Extract and store stances from bot response (background-safe)
        if clean_response and flow_analysis.epistemic.assertion_density >= 0.4:
            from insult.core.stance_log import extract_stances

            extraction = extract_stances(
                clean_response, flow_analysis.epistemic.assertion_density, __import__("time").time()
            )
            for entry in extraction.entries:
                await self.memory.store_stance(channel_id, user_id, entry.topic, entry.position, entry.confidence)

        # Fire background tasks: reactions, fact extraction
        if reactions:
            self._spawn_task(add_reactions(message, reactions), name="reactions")

        # Execute tool calls — reminders and channels run in background
        if llm_response.tool_calls:
            reminder_names = {"create_reminder", "list_reminders", "cancel_reminder"}
            reminder_calls = [tc for tc in llm_response.tool_calls if tc.name in reminder_names]
            other_calls = [tc for tc in llm_response.tool_calls if tc.name not in reminder_names]
            for rc in reminder_calls:
                self._spawn_task(self._execute_reminder_call(message, rc), name=f"reminder:{rc.name}")
            if other_calls and message.guild:
                self._spawn_task(
                    self._execute_tool_calls(message, other_calls),
                    name=f"tool_calls:{','.join(tc.name for tc in other_calls)}",
                )

        # Fallback: if LLM produced no text AND no reactions AND no tool calls, send in-character error
        # (tool calls run in background — empty text is expected when LLM only produced tool_use)
        if not response.strip() and not reactions and not llm_response.tool_calls:
            log.warning(
                "empty_response_fallback",
                raw_llm_len=post_llm_len,
                raw_llm_preview=llm_response.text[:200] if llm_response.text else "",
                final_len=len(response),
                tool_calls=len(llm_response.tool_calls),
            )
            response = get_error_response("generic")

        # Send text response with [SEND] splitting, chunking, and typing delays
        has_side_effects = bool(reactions or llm_response.tool_calls)
        delivery_start = time.monotonic()
        delivery_outcome = "ok"
        try:
            await send_response(message.channel, response, has_side_effects=has_side_effects)
            log.info(
                "chat_delivery_ok",
                final_text_len=len(response),
                delivery_ms=int((time.monotonic() - delivery_start) * 1000),
                llm_ms=llm_ms,
            )
        except discord.HTTPException as e:
            delivery_outcome = "delivery_failed"
            log.error(
                "chat_delivery_failed",
                error_type=type(e).__name__,
                status=getattr(e, "status", None),
                code=getattr(e, "code", None),
                error_msg=str(e)[:200],
                final_text_len=len(response),
                delivery_ms=int((time.monotonic() - delivery_start) * 1000),
            )
            # Don't raise — we still want the post-generation telemetry below to run
            # so the turn shows up in traces.
        except Exception:
            delivery_outcome = "delivery_crashed"
            log.exception(
                "chat_delivery_crashed",
                final_text_len=len(response),
                delivery_ms=int((time.monotonic() - delivery_start) * 1000),
            )

        # Post-generation flow adherence validation (telemetry only)
        validate_flow_adherence(response, flow_analysis)

        # Phase 2 (v3.1.0): Quality check (telemetry only — does not block)
        from insult.core.quality import check_quality

        recent_shapes = self._expression_history.recent_shapes(context_key, n=5)
        check_quality(response, text, recent_shapes, agreement_streak=flow_analysis.agreement_streak)

        # Record message trace for dashboard
        from insult.core.metrics import record_message_trace

        record_message_trace(
            {
                "user": user_name,
                "user_id": user_id,
                "channel": channel_id,
                "input": text[:200],
                "response": response[:300],
                "preset": preset.mode.value,
                "preset_modifiers": [m.value for m in preset.modifiers],
                "pressure": flow_analysis.pressure.pressure_level,
                "expression_shape": flow_analysis.expression.selected_shape.value,
                "expression_flavor": flow_analysis.expression.selected_flavor.value,
                "epistemic_move": flow_analysis.epistemic.recommended_move.value,
                "awareness_pattern": flow_analysis.awareness.detected_pattern.value,
                "tools": [tc.name for tc in llm_response.tool_calls] if llm_response.tool_calls else [],
                "reactions": reactions,
            }
        )

        # Background fact extraction (with channel posting)
        guild_id = str(message.guild.id) if message.guild else None
        ch_name = getattr(message.channel, "name", "")
        self._spawn_task(
            self._extract_user_facts(user_id, user_name, user_facts, recent, guild_id, ch_name),
            name="fact_extraction",
        )

        return delivery_outcome

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

        started = time.monotonic()
        audio_bytes = 0
        try:
            audio_data = await message.attachments[0].read()
            audio_bytes = len(audio_data)
            result = await transcribe_voice_message(
                audio_data,
                endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_key.get_secret_value(),
                deployment=self.settings.azure_openai_whisper_deployment,
            )
            log.info(
                "voice_transcription_ok",
                duration_ms=int((time.monotonic() - started) * 1000),
                audio_bytes=audio_bytes,
                text_len=len(result or ""),
                text_preview=(result or "")[:120],
            )
            return result
        except Exception:
            log.exception(
                "voice_transcription_failed",
                duration_ms=int((time.monotonic() - started) * 1000),
                audio_bytes=audio_bytes,
            )
            return None

    def _spawn_task(self, coro, *, name: str | None = None) -> None:
        """Create a background task with automatic cleanup + completion logging.

        `name` is the semantic label logged on completion (e.g. "reactions",
        "fact_extraction", "tool_calls"). When omitted we fall back to the
        coroutine's __qualname__ so nothing ever goes unlabeled.
        """
        label = name or getattr(coro, "__qualname__", "unknown")
        spawn_started = time.monotonic()

        async def _tracked() -> None:
            try:
                await coro
                log.debug(
                    "background_task_ok",
                    task=label,
                    duration_ms=int((time.monotonic() - spawn_started) * 1000),
                )
            except asyncio.CancelledError:
                log.info(
                    "background_task_cancelled",
                    task=label,
                    duration_ms=int((time.monotonic() - spawn_started) * 1000),
                )
                raise
            except Exception:
                # Inner task bodies already log.exception; this one is a
                # belt-and-suspenders so a background failure ALWAYS leaves
                # a terminal event in the trace.
                log.exception(
                    "background_task_failed",
                    task=label,
                    duration_ms=int((time.monotonic() - spawn_started) * 1000),
                )

        task = asyncio.create_task(_tracked())
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
                # Post to system reminders channel
                if guild_id:
                    mention_display = " ".join(f"<@{uid.strip()}>" for uid in mention_ids if uid) if mention_ids else ""
                    await post_reminder_set(
                        self.bot,
                        self.memory,
                        guild_id,
                        description,
                        remind_at_str,
                        mention_display,
                        recurring,
                        reminder_id,
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
                            self._inaugurate_channel(channel, tool_call.input.get("name", ""), message.author),
                            name=f"inaugurate:{channel.name}",
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

    async def _extract_user_facts(
        self,
        user_id: str,
        user_name: str,
        existing_facts: list[dict],
        recent: list[dict],
        guild_id: str | None = None,
        channel_name: str = "",
    ):
        """Background task: extract user facts from recent conversation."""
        try:
            new_facts = await extract_facts(
                self.llm.client, self.settings.summary_model, user_name, existing_facts, recent
            )
            if new_facts != existing_facts:
                await self.memory.save_facts(user_id, new_facts)
                # Post new safe facts to system channel
                if guild_id:
                    await post_facts_to_channel(
                        self.bot,
                        self.memory,
                        guild_id,
                        user_name,
                        new_facts,
                        existing_facts,
                        channel_name,
                    )
        except Exception:
            log.exception("facts_extraction_background_failed", user_id=user_id)
