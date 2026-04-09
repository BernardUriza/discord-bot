"""Tests for the debug HTTP server."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from insult.core.debug_server import build_app

TOKEN = "test-token-xyz"  # noqa: S105 — fixture token, not a real secret


@pytest.fixture
def memory_with_data():
    mem = AsyncMock()
    mem.get_recent = AsyncMock(
        return_value=[
            {"user_name": "Bernard", "role": "user", "content": "hola", "timestamp": 1700000000.0},
            {"user_name": "Insult", "role": "assistant", "content": "que quieres", "timestamp": 1700000001.0},
        ]
    )
    mem.get_stats = AsyncMock(return_value={"total_messages": 42, "unique_users": 3, "unique_channels": 2})
    mem.get_channel_activity_since = AsyncMock(
        return_value=[{"channel_id": "111", "count": 20}, {"channel_id": "222", "count": 5}]
    )
    return mem


@pytest.fixture
async def client(memory_with_data):
    app = build_app(memory_with_data, TOKEN)
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_health_no_auth_required(client):
    resp = await client.get("/debug/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"


async def test_messages_requires_auth(client):
    resp = await client.get("/debug/messages?channel_id=123")
    assert resp.status == 401


async def test_messages_rejects_wrong_token(client):
    resp = await client.get("/debug/messages?channel_id=123", headers={"Authorization": "Bearer wrong"})
    assert resp.status == 401


async def test_messages_rejects_missing_bearer_prefix(client):
    resp = await client.get("/debug/messages?channel_id=123", headers={"Authorization": TOKEN})
    assert resp.status == 401


async def test_messages_returns_data_with_valid_token(client, memory_with_data):
    resp = await client.get("/debug/messages?channel_id=555&limit=15", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status == 200
    data = await resp.json()
    assert data["channel_id"] == "555"
    assert data["count"] == 2
    assert len(data["messages"]) == 2
    assert data["messages"][0]["user_name"] == "Bernard"
    memory_with_data.get_recent.assert_awaited_once_with("555", limit=15)


async def test_messages_missing_channel_id(client):
    resp = await client.get("/debug/messages", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status == 400


async def test_messages_invalid_limit(client):
    resp = await client.get(
        "/debug/messages?channel_id=555&limit=notanumber", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status == 400


async def test_messages_limit_out_of_range(client):
    resp = await client.get("/debug/messages?channel_id=555&limit=9999", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status == 400


async def test_messages_default_limit(client, memory_with_data):
    resp = await client.get("/debug/messages?channel_id=555", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status == 200
    memory_with_data.get_recent.assert_awaited_once_with("555", limit=15)


async def test_stats_endpoint(client):
    resp = await client.get("/debug/stats", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status == 200
    data = await resp.json()
    assert data["total_messages"] == 42
    assert data["unique_users"] == 3


async def test_stats_requires_auth(client):
    resp = await client.get("/debug/stats")
    assert resp.status == 401


async def test_channels_endpoint(client, memory_with_data):
    resp = await client.get("/debug/channels?guild_id=abc&since_hours=1", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status == 200
    data = await resp.json()
    assert data["guild_id"] == "abc"
    assert data["since_hours"] == 1
    assert data["count"] == 2
    assert data["channels"][0]["channel_id"] == "111"
    memory_with_data.get_channel_activity_since.assert_awaited_once()


async def test_channels_missing_guild_id(client):
    resp = await client.get("/debug/channels", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status == 400


async def test_channels_invalid_since_hours(client):
    resp = await client.get(
        "/debug/channels?guild_id=abc&since_hours=nope", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status == 400
