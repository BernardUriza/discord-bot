"""Claude API client with character break detection."""

import asyncio

import anthropic
import structlog
from anthropic.types import MessageParam

from insult.core.character import CHARACTER_REINFORCEMENT, detect_break, sanitize

log = structlog.get_logger()


class LLMClient:
    def __init__(self, api_key: str, model: str, max_tokens: int, timeout: float = 30.0, max_retries: int = 5):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    async def _send(self, system_prompt: str, messages: list[MessageParam]) -> str:
        """Raw API call with retry logic for transient errors."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                log.info("llm_request", model=self.model, attempt=attempt, messages=len(messages))
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
                log.info(
                    "llm_response",
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    stop_reason=response.stop_reason,
                )
                block = response.content[0]
                return block.text if hasattr(block, "text") else str(block)

            except anthropic.RateLimitError as e:
                last_error = e
                wait = 2**attempt
                log.warning("llm_rate_limited", attempt=attempt, wait_seconds=wait)
                await asyncio.sleep(wait)

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
                log.error("llm_api_error", status=e.status_code, error=str(e))
                break

        log.error("llm_failed", attempts=self.max_retries, last_error=str(last_error))
        raise last_error

    async def chat(self, system_prompt: str, messages: list[MessageParam]) -> str:
        """Send messages, detect character breaks, retry if needed."""
        response = await self._send(system_prompt, messages)

        breaks = detect_break(response)
        if breaks:
            log.warning("character_break_detected", patterns=breaks)

            reinforced_prompt = system_prompt + CHARACTER_REINFORCEMENT
            try:
                retry_response = await self._send(reinforced_prompt, messages)
                retry_breaks = detect_break(retry_response)

                if not retry_breaks:
                    log.info("character_break_fixed_on_retry")
                    return retry_response

                log.warning("character_break_persisted", retry_patterns=retry_breaks)
                return sanitize(retry_response)

            except Exception:
                log.warning("character_break_retry_failed_using_sanitized_original")
                return sanitize(response)

        return response
