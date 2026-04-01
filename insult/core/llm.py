"""Claude API client with character break detection and tool_use support."""

import asyncio
from dataclasses import dataclass, field

import anthropic
import structlog
from anthropic.types import MessageParam

from insult.core.actions import ToolCall
from insult.core.character import CHARACTER_REINFORCEMENT, detect_anti_patterns, detect_break, sanitize, strip_metadata

log = structlog.get_logger()


@dataclass
class LLMResponse:
    """Structured response from the LLM — text + optional tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def _parse_response_content(content: list) -> LLMResponse:
    """Extract text and tool_use blocks from Claude API response content."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

    return LLMResponse(text="\n".join(text_parts).strip(), tool_calls=tool_calls)


class LLMClient:
    def __init__(self, api_key: str, model: str, max_tokens: int, timeout: float = 30.0, max_retries: int = 5):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    async def _send(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        *,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> LLMResponse:
        """Raw API call with retry logic for transient errors."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                log.info("llm_request", model=self.model, attempt=attempt, messages=len(messages))
                kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": system_prompt,
                    "messages": messages,
                    "cache_control": {"type": "ephemeral"},
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

            except anthropic.APIError as e:
                last_error = e
                log.error("llm_api_error", error=str(e))
                break

        log.error("llm_failed", attempts=self.max_retries, last_error=str(last_error))
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

        if response.tool_calls:
            log.info("tool_calls_detected", tools=[tc.name for tc in response.tool_calls])

        return response
