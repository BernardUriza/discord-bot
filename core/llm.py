"""Cliente LLM con Anthropic Claude API."""

from typing import Any

import anthropic
from anthropic.types import MessageParam


class LLMClient:
    def __init__(self, api_key: str, model: str, max_tokens: int):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def chat(
        self,
        system_prompt: str,
        messages: list[MessageParam],
    ) -> str:
        """Envía mensajes a Claude y devuelve la respuesta."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )
        block = response.content[0]
        return block.text if hasattr(block, "text") else str(block)
