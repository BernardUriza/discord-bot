"""Image generation via Pollinations.ai — free, no API key required.

Insult uses images as visual punctuation: proactive, expressive, disruptive.
Images are fetched as bytes and sent as Discord attachments (inline display).
"""

import io
import time
from urllib.parse import quote

import aiohttp
import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
DEFAULT_MODEL = "flux"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
REQUEST_TIMEOUT = 60  # Pollinations can be slow (5-15s for Flux)
MAX_PROMPT_LENGTH = 500
IMAGE_COOLDOWN_SECONDS = 20  # Min seconds between image generations (rate limit protection)

# Global throttle state
_last_generation_time: float = 0.0


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def is_throttled() -> bool:
    """Check if we should skip image generation due to rate limiting."""
    global _last_generation_time
    now = time.monotonic()
    return (now - _last_generation_time) < IMAGE_COOLDOWN_SECONDS


async def generate_image(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int | None = None,
    safe: bool = True,
    api_key: str = "",
) -> io.BytesIO | None:
    """Fetch a generated image from Pollinations.ai.

    Returns BytesIO with image data, or None on failure.
    Respects internal cooldown to avoid rate limiting.
    """
    global _last_generation_time

    if is_throttled():
        log.warning("image_throttled", cooldown=IMAGE_COOLDOWN_SECONDS)
        return None

    # Sanitize prompt
    clean_prompt = prompt.strip()[:MAX_PROMPT_LENGTH]
    if not clean_prompt:
        log.warning("image_empty_prompt")
        return None

    # Build URL
    encoded_prompt = quote(clean_prompt)
    url = f"{POLLINATIONS_BASE}/{encoded_prompt}"

    params: dict = {
        "model": model,
        "width": width,
        "height": height,
        "nologo": "true",
        "safe": "true" if safe else "false",
    }
    if seed is not None:
        params["seed"] = str(seed)
    if api_key:
        params["key"] = api_key

    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(url, params=params) as resp,
        ):
            if resp.status != 200:
                log.error(
                    "image_fetch_failed",
                    status=resp.status,
                    prompt=clean_prompt[:80],
                )
                return None

            data = await resp.read()
            if len(data) < 1000:
                # Suspiciously small — likely an error response
                log.error("image_too_small", size=len(data))
                return None

            _last_generation_time = time.monotonic()
            log.info(
                "image_generated",
                model=model,
                size_kb=len(data) // 1024,
                width=width,
                height=height,
                prompt_length=len(clean_prompt),
            )
            return io.BytesIO(data)

    except TimeoutError:
        log.error("image_timeout", timeout=REQUEST_TIMEOUT)
        return None
    except aiohttp.ClientError as e:
        log.error("image_client_error", error=str(e))
        return None
