"""Claude API client with character break detection and tool_use support."""

import asyncio
from dataclasses import dataclass, field

import anthropic
import structlog
from anthropic.types import MessageParam

from insult.core.actions import ToolCall
from insult.core.character import (
    CACHE_BOUNDARY,
    CHARACTER_REINFORCEMENT,
    detect_anti_patterns,
    detect_break,
    normalize_formatting,
    sanitize,
    strip_lists,
    strip_metadata,
)

log = structlog.get_logger()

# --- Token usage tracking (in-memory, resets on redeploy) ---
# Pricing per million tokens (Sonnet 4, as of 2026)
_PRICING = {
    "input": 3.00,  # $/M input tokens
    "output": 15.00,  # $/M output tokens
    "cache_read": 0.30,  # $/M cache read tokens (90% discount)
    "cache_create": 3.75,  # $/M cache creation tokens (25% premium)
}

_usage_totals = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_create_tokens": 0,
    "requests": 0,
    "errors": 0,
}


def record_usage(input_tokens: int, output_tokens: int, cache_read: int = 0, cache_create: int = 0) -> None:
    """Accumulate token usage for cost tracking."""
    _usage_totals["input_tokens"] += input_tokens
    _usage_totals["output_tokens"] += output_tokens
    _usage_totals["cache_read_tokens"] += cache_read
    _usage_totals["cache_create_tokens"] += cache_create
    _usage_totals["requests"] += 1


def get_usage_report() -> dict:
    """Return accumulated usage with estimated cost in USD."""
    t = _usage_totals
    cost_input = (t["input_tokens"] / 1_000_000) * _PRICING["input"]
    cost_output = (t["output_tokens"] / 1_000_000) * _PRICING["output"]
    cost_cache_read = (t["cache_read_tokens"] / 1_000_000) * _PRICING["cache_read"]
    cost_cache_create = (t["cache_create_tokens"] / 1_000_000) * _PRICING["cache_create"]
    total_cost = cost_input + cost_output + cost_cache_read + cost_cache_create

    return {
        "tokens": {
            "input": t["input_tokens"],
            "output": t["output_tokens"],
            "cache_read": t["cache_read_tokens"],
            "cache_create": t["cache_create_tokens"],
            "total": t["input_tokens"] + t["output_tokens"],
        },
        "requests": t["requests"],
        "errors": t["errors"],
        "cost_usd": {
            "input": round(cost_input, 4),
            "output": round(cost_output, 4),
            "cache_read": round(cost_cache_read, 4),
            "cache_create": round(cost_cache_create, 4),
            "total": round(total_cost, 4),
        },
        "avg_tokens_per_request": round((t["input_tokens"] + t["output_tokens"]) / max(t["requests"], 1)),
        "note": "Resets on redeploy. Pricing based on Sonnet 4 rates.",
    }


@dataclass
class LLMResponse:
    """Structured response from the LLM — text + optional tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


# Web search tool definition — Claude's native server-side search
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 3,
}


def _build_system_blocks(system_prompt: str) -> list[dict] | str:
    """Build Anthropic system blocks with prompt caching on the stable prefix.

    If `system_prompt` contains the CACHE_BOUNDARY marker, split it into a
    cacheable stable block (everything before the marker, with
    cache_control=ephemeral) and a dynamic block (everything after, no cache).
    If the marker is absent, return the raw string (backwards compatible with
    callers that don't mark a boundary — e.g., simple utility calls).
    """
    if CACHE_BOUNDARY not in system_prompt:
        return system_prompt

    stable, dynamic = system_prompt.split(CACHE_BOUNDARY, 1)
    stable = stable.rstrip()
    dynamic = dynamic.lstrip()

    if not stable:
        return dynamic or system_prompt

    blocks: list[dict] = [
        {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
    ]
    if dynamic:
        blocks.append({"type": "text", "text": dynamic})
    return blocks


def _parse_response_content(content: list) -> LLMResponse:
    """Extract text and tool_use blocks from Claude API response content.

    Handles standard text, tool_use (channel creation), and server-side
    blocks (web_search server_tool_use / web_search_tool_result) which
    are processed transparently by the API — we just skip them.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
        # server_tool_use and web_search_tool_result are handled server-side
        # by Claude — we just skip them in parsing

    return LLMResponse(text="\n".join(text_parts).strip(), tool_calls=tool_calls)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        timeout: float = 30.0,
        max_retries: int = 5,
        cure_model: str = "",
    ):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cure_model = cure_model  # Haiku model for language cure (step 7c)

    async def _send(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        *,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> LLMResponse:
        """Raw API call with retry logic for transient errors."""
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")

        last_error: Exception | None = None
        attempt = 0
        # kwargs is initialized per-iteration inside the loop, but we bind it
        # here so the BadRequestError fallback (`kwargs.pop("tools")`) can never
        # reference an unbound name even under exotic control flow.
        kwargs: dict = {}

        for attempt in range(1, self.max_retries + 1):
            try:
                log.info("llm_request", model=self.model, attempt=attempt, messages=len(messages))
                kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": _build_system_blocks(system_prompt),
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice

                response = await self.client.messages.create(**kwargs)
                cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                log.info(
                    "llm_response",
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cache_read=cache_read,
                    cache_create=cache_create,
                    stop_reason=response.stop_reason,
                )
                record_usage(response.usage.input_tokens, response.usage.output_tokens, cache_read, cache_create)
                return _parse_response_content(response.content)

            except anthropic.RateLimitError as e:
                last_error = e
                wait = 2**attempt
                log.warning("llm_rate_limited", attempt=attempt, wait_seconds=wait)
                await asyncio.sleep(wait)

            except anthropic.BadRequestError as e:
                # If tools caused the 400, retry WITHOUT tools so the bot doesn't go down
                if tools and "tool" in str(e).lower():
                    log.warning("llm_tools_rejected_fallback", error=str(e))
                    kwargs.pop("tools", None)
                    tools = None  # Don't try tools again on retry
                    continue
                last_error = e
                log.error("llm_bad_request", error=str(e))
                break

            except anthropic.AuthenticationError as e:
                log.error("llm_auth_error", error=str(e))
                raise

            except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
                last_error = e
                log.warning("llm_timeout", attempt=attempt, error=str(e))
                if attempt == self.max_retries:
                    break
                await asyncio.sleep(1)

            except anthropic.APIStatusError as e:
                last_error = e
                # 529 Overloaded — transient, retry with backoff
                if e.status_code == 529:
                    wait = 2**attempt
                    log.warning("llm_overloaded", attempt=attempt, wait_seconds=wait, status=529)
                    if attempt == self.max_retries:
                        break
                    await asyncio.sleep(wait)
                else:
                    log.error("llm_api_error", status=e.status_code, error=str(e))
                    break

        _usage_totals["errors"] += 1
        log.error(
            "llm_failed",
            attempts=attempt,
            max_retries=self.max_retries,
            last_error_type=type(last_error).__name__ if last_error else None,
            last_error=str(last_error),
        )
        if last_error is None:
            # Unreachable with max_retries >= 1 (loop runs at least once, and
            # every path either returns, raises, or assigns last_error). Keep
            # the explicit RuntimeError instead of `raise None` so if the
            # invariant breaks in the future the failure mode is debuggable.
            raise RuntimeError("llm._send exited loop without last_error set")
        raise last_error

    async def chat(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        *,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> LLMResponse:
        """Send messages with optional tools, detect character breaks, retry if needed.

        Returns LLMResponse with text (for Discord) and tool_calls (for actions).
        """
        response = await self._send(system_prompt, messages, tools=tools, tool_choice=tool_choice)

        # Strip leaked metadata (timestamps, speaker labels) before any other processing
        response.text = strip_metadata(response.text)

        breaks = detect_break(response.text)
        if breaks:
            log.warning("character_break_detected", patterns=breaks)

            reinforced_prompt = system_prompt + CHARACTER_REINFORCEMENT
            try:
                retry_response = await self._send(reinforced_prompt, messages, tools=tools, tool_choice=tool_choice)
                retry_response.text = strip_metadata(retry_response.text)
                retry_breaks = detect_break(retry_response.text)

                if not retry_breaks:
                    log.info("character_break_fixed_on_retry")
                    return retry_response

                log.warning("character_break_persisted", retry_patterns=retry_breaks)
                retry_response.text = sanitize(retry_response.text)
                return retry_response

            except Exception:
                log.warning("character_break_retry_failed_using_sanitized_original")
                response.text = sanitize(response.text)
                return response

        # Log anti-pattern drift (soft violations — doesn't block, just monitors)
        anti_patterns = detect_anti_patterns(response.text)
        if anti_patterns:
            log.warning("anti_pattern_detected", patterns=anti_patterns)

        # Step 7c: Language Cure — normalize mixed-language output via Haiku
        if self.cure_model and response.text:
            from insult.core.language import language_cure

            response.text = await language_cure(self.client, self.cure_model, response.text)
            # Defense in depth: the cure model (Haiku) sometimes re-introduces
            # scratchpad XML or stray tags even when the root prompt avoids them.
            # Re-run strip_metadata so nothing reaches the user.
            response.text = strip_metadata(response.text)

        # Step 7d: Formatting normalization — deterministic enforcement of
        # exclamation limits (max 1) and bold limits (max 2). Runs LAST so
        # neither the LLM nor the language cure can reintroduce violations.
        if response.text:
            response.text = normalize_formatting(response.text)
            response.text = strip_lists(response.text)

        if response.tool_calls:
            log.info("tool_calls_detected", tools=[tc.name for tc in response.tool_calls])

        return response
