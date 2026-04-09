"""Debug HTTP server — read-only endpoints for inspecting bot state.

Exposes a small aiohttp app alongside the Discord bot so external tools
(including Claude Code) can read recent messages, channel activity, and
memory stats without touching Discord's API.

Security model:
- Fail-closed: if DEBUG_TOKEN is unset, the server does NOT start.
- All endpoints except /debug/health require `Authorization: Bearer <token>`.
- Binds to 0.0.0.0 by default. For production, gate via network boundary
  (Azure Container Apps without ingress is private by default).

All endpoints are read-only. No write paths. Ever.
"""

from __future__ import annotations

import hmac

import structlog
from aiohttp import web

from insult.core.memory import MemoryStore

log = structlog.get_logger()

# Typed aiohttp app keys (avoid NotAppKeyWarning)
_MEMORY_KEY: web.AppKey[MemoryStore] = web.AppKey("memory", MemoryStore)
_TOKEN_KEY: web.AppKey[str] = web.AppKey("debug_token", str)


def _unauthorized() -> web.Response:
    return web.json_response({"error": "unauthorized"}, status=401)


def _bad_request(msg: str) -> web.Response:
    return web.json_response({"error": msg}, status=400)


@web.middleware
async def _auth_middleware(request: web.Request, handler):
    # /debug/health is always public (liveness probe)
    if request.path == "/debug/health":
        return await handler(request)

    expected_token = request.app[_TOKEN_KEY]
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return _unauthorized()
    provided = header[len("Bearer ") :]
    # Constant-time compare to resist timing side-channel
    if not hmac.compare_digest(provided, expected_token):
        return _unauthorized()
    return await handler(request)


async def _handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _handle_messages(request: web.Request) -> web.Response:
    channel_id = request.query.get("channel_id")
    if not channel_id:
        return _bad_request("channel_id is required")

    try:
        limit = int(request.query.get("limit", "15"))
    except ValueError:
        return _bad_request("limit must be an integer")
    if limit < 1 or limit > 500:
        return _bad_request("limit must be between 1 and 500")

    memory = request.app[_MEMORY_KEY]
    messages = await memory.get_recent(channel_id, limit=limit)
    return web.json_response({"channel_id": channel_id, "count": len(messages), "messages": messages})


async def _handle_channels(request: web.Request) -> web.Response:
    guild_id = request.query.get("guild_id")
    if not guild_id:
        return _bad_request("guild_id is required")

    try:
        since_hours = float(request.query.get("since_hours", "24"))
    except ValueError:
        return _bad_request("since_hours must be a number")

    import time as _time

    since_ts = _time.time() - (since_hours * 3600)
    memory = request.app[_MEMORY_KEY]
    activity = await memory.get_channel_activity_since(guild_id, since_ts)
    return web.json_response(
        {"guild_id": guild_id, "since_hours": since_hours, "count": len(activity), "channels": activity}
    )


async def _handle_stats(request: web.Request) -> web.Response:
    memory = request.app[_MEMORY_KEY]
    stats = await memory.get_stats()
    return web.json_response(stats)


def build_app(memory: MemoryStore, debug_token: str) -> web.Application:
    """Construct the aiohttp Application with routes and middleware."""
    app = web.Application(middlewares=[_auth_middleware])
    app[_MEMORY_KEY] = memory
    app[_TOKEN_KEY] = debug_token
    app.router.add_get("/debug/health", _handle_health)
    app.router.add_get("/debug/messages", _handle_messages)
    app.router.add_get("/debug/channels", _handle_channels)
    app.router.add_get("/debug/stats", _handle_stats)
    return app


async def start_debug_server(
    memory: MemoryStore,
    debug_token: str,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> web.AppRunner:
    """Start the debug server and return the runner for lifecycle management."""
    app = build_app(memory, debug_token)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("debug_server_started", host=host, port=port)
    return runner


async def stop_debug_server(runner: web.AppRunner) -> None:
    """Cleanly stop the debug server."""
    await runner.cleanup()
    log.info("debug_server_stopped")
