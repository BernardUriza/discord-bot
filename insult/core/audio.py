"""Audio clip generation — YouTube search + Freesound memes.

Insult uses audio clips as sonic punctuation: dropping music snippets,
meme sounds, and karaoke-style clips into conversations proactively.
Clips are 15s max, sent as MP3 attachments.
"""

import asyncio
import io
import re
import tempfile
import time
from pathlib import Path

import aiohttp
import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_DURATION_SECONDS = 15
AUDIO_COOLDOWN_SECONDS = 30  # Min seconds between audio clips
MAX_SEARCH_QUERY_LENGTH = 200
DOWNLOAD_TIMEOUT = 30  # yt-dlp can be slow
FREESOUND_BASE = "https://freesound.org/apiv2"

# Global throttle state
_last_audio_time: float = 0.0


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


def is_throttled() -> bool:
    """Check if we should skip audio generation due to rate limiting."""
    global _last_audio_time
    now = time.monotonic()
    return (now - _last_audio_time) < AUDIO_COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# YouTube search + clip extraction via yt-dlp
# ---------------------------------------------------------------------------


async def search_and_clip_youtube(query: str) -> io.BytesIO | None:
    """Search YouTube, download audio, extract a 15s clip.

    Runs yt-dlp in a subprocess to avoid blocking the event loop.
    Returns BytesIO with MP3 data, or None on failure.
    """
    global _last_audio_time

    if is_throttled():
        log.warning("audio_throttled", cooldown=AUDIO_COOLDOWN_SECONDS)
        return None

    clean_query = query.strip()[:MAX_SEARCH_QUERY_LENGTH]
    # Sanitize: strip shell metacharacters and yt-dlp special chars
    clean_query = re.sub(r"[;&|`$\\\"'<>(){}\[\]!#~]", "", clean_query)
    clean_query = clean_query.strip()
    if not clean_query:
        log.warning("audio_empty_query")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.%(ext)s"
        clip_path = Path(tmpdir) / "clip.mp3"

        # Step 1: Search + download audio via yt-dlp (first result only)
        ytdlp_cmd = [
            "yt-dlp",
            f"ytsearch1:{clean_query}",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "5",  # ~128kbps, good enough for 15s
            "--max-downloads",
            "1",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--js-runtimes",
            "node",  # Required for YouTube JS challenge solving
            "--output",
            str(raw_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ytdlp_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=DOWNLOAD_TIMEOUT)

            if proc.returncode != 0:
                log.error("audio_ytdlp_failed", returncode=proc.returncode, stderr=stderr.decode()[:200])
                return None
        except TimeoutError:
            proc.kill()
            log.error("audio_ytdlp_timeout", timeout=DOWNLOAD_TIMEOUT)
            return None
        except FileNotFoundError:
            log.error("audio_ytdlp_not_installed")
            return None

        # Find the downloaded file (yt-dlp adds the extension)
        downloaded = list(Path(tmpdir).glob("raw.*"))
        if not downloaded:
            log.error("audio_no_file_downloaded")
            return None
        source_file = downloaded[0]

        # Step 2: Extract 15s clip with ffmpeg (start at 30s to skip intros, or from start if short)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            "30",  # Skip intro — if song is <30s, ffmpeg handles gracefully
            "-i",
            str(source_file),
            "-t",
            str(CLIP_DURATION_SECONDS),
            "-acodec",
            "libmp3lame",
            "-ab",
            "128k",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(clip_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)

            if proc.returncode != 0:
                log.error("audio_ffmpeg_failed", returncode=proc.returncode)
                return None
        except TimeoutError:
            proc.kill()
            log.error("audio_ffmpeg_timeout")
            return None
        except FileNotFoundError:
            log.error("audio_ffmpeg_not_installed")
            return None

        # Read the clip
        if not clip_path.exists() or clip_path.stat().st_size < 1000:
            log.error("audio_clip_too_small")
            return None

        data = clip_path.read_bytes()
        _last_audio_time = time.monotonic()

        log.info(
            "audio_clip_generated",
            query=clean_query[:80],
            size_kb=len(data) // 1024,
            duration=CLIP_DURATION_SECONDS,
            source="youtube",
        )
        return io.BytesIO(data)


# ---------------------------------------------------------------------------
# Freesound meme search (free tier — requires API key, optional)
# ---------------------------------------------------------------------------


async def search_freesound(query: str, api_key: str | None = None) -> io.BytesIO | None:
    """Search Freesound.org for a sound effect and return as MP3.

    Requires FREESOUND_API_KEY in settings. Returns None if not configured.
    """
    global _last_audio_time

    if not api_key:
        log.debug("audio_freesound_not_configured")
        return None

    if is_throttled():
        log.warning("audio_throttled", cooldown=AUDIO_COOLDOWN_SECONDS)
        return None

    clean_query = query.strip()[:MAX_SEARCH_QUERY_LENGTH]
    if not clean_query:
        return None

    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Search for sounds
            search_url = f"{FREESOUND_BASE}/search/text/"
            params = {
                "query": clean_query,
                "token": api_key,
                "fields": "id,name,previews",
                "page_size": "1",
                "sort": "rating_desc",
            }
            async with session.get(search_url, params=params) as resp:
                if resp.status != 200:
                    log.error("audio_freesound_search_failed", status=resp.status)
                    return None
                data = await resp.json()

            results = data.get("results", [])
            if not results:
                log.info("audio_freesound_no_results", query=clean_query[:80])
                return None

            # Download the preview MP3
            preview_url = results[0].get("previews", {}).get("preview-hq-mp3")
            if not preview_url:
                return None

            async with session.get(preview_url) as resp:
                if resp.status != 200:
                    return None
                audio_data = await resp.read()

            _last_audio_time = time.monotonic()
            log.info(
                "audio_clip_generated",
                query=clean_query[:80],
                size_kb=len(audio_data) // 1024,
                source="freesound",
                sound_name=results[0].get("name", ""),
            )
            return io.BytesIO(audio_data)

    except Exception:
        log.exception("audio_freesound_error")
        return None
