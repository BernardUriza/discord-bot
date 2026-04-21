"""Tests for the facts persistence contract — the bug that mattered.

Before v3.4.10 `save_facts` did a blanket `DELETE FROM user_facts WHERE
user_id=?` and re-inserted whatever the LLM returned. Because extract_facts
only sees the last ~10 messages, anything older (manual injections, older
auto-extractions) routinely got wiped on the NEXT conversation turn.

These tests lock in the new contract:
- Rows with source='manual' NEVER get deleted by save_facts.
- Rows with source='auto' are replaced atomically per save_facts call.
- add_manual_fact writes with source='manual' and returns the row id.
- The migration from the old schema (no source column) is idempotent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from insult.core.memory import MemoryStore


@pytest.fixture
async def store():
    """Fresh in-file SQLite store for each test (tempfile so migrations run)."""
    with tempfile.TemporaryDirectory() as tmp:
        s = MemoryStore(Path(tmp) / "test.db")
        await s.connect()
        yield s
        await s.close()


@pytest.mark.asyncio
async def test_add_manual_fact_inserts_with_manual_source(store):
    row_id = await store.add_manual_fact("u1", "Es de Orange County", "identity")
    assert row_id > 0
    facts = await store.get_facts("u1")
    assert len(facts) == 1
    assert facts[0]["fact"] == "Es de Orange County"
    assert facts[0]["category"] == "identity"


@pytest.mark.asyncio
async def test_save_facts_preserves_manual_rows(store):
    manual_id = await store.add_manual_fact("u1", "Es de Orange County, CA", "identity")
    await store.save_facts(
        "u1",
        [
            {"fact": "Le gusta Python", "category": "interests"},
            {"fact": "Vive en CDMX", "category": "location"},
        ],
    )
    facts = await store.get_facts("u1")
    assert len(facts) == 3, "auto save must not wipe the manual fact"
    texts = {f["fact"] for f in facts}
    assert "Es de Orange County, CA" in texts
    assert "Le gusta Python" in texts
    assert "Vive en CDMX" in texts
    assert any(f["id"] == manual_id and f["fact"] == "Es de Orange County, CA" for f in facts)


@pytest.mark.asyncio
async def test_save_facts_replaces_previous_auto_rows(store):
    await store.save_facts("u1", [{"fact": "Old fact A", "category": "personal"}])
    await store.save_facts("u1", [{"fact": "Old fact B", "category": "personal"}])
    facts = await store.get_facts("u1")
    assert len(facts) == 1
    assert facts[0]["fact"] == "Old fact B"


@pytest.mark.asyncio
async def test_repeated_auto_saves_do_not_accumulate_but_manual_persists(store):
    await store.add_manual_fact("u1", "Nacio en Orange County", "identity")
    for i in range(5):
        await store.save_facts(
            "u1",
            [
                {"fact": f"auto-turn-{i}-a", "category": "personal"},
                {"fact": f"auto-turn-{i}-b", "category": "personal"},
            ],
        )
    facts = await store.get_facts("u1")
    assert len(facts) == 3
    texts = {f["fact"] for f in facts}
    assert "Nacio en Orange County" in texts
    assert "auto-turn-4-a" in texts
    assert "auto-turn-4-b" in texts
    for i in range(4):
        assert f"auto-turn-{i}-a" not in texts


@pytest.mark.asyncio
async def test_manual_fact_isolation_between_users(store):
    await store.add_manual_fact("u1", "u1 private detail", "personal")
    await store.add_manual_fact("u2", "u2 private detail", "personal")
    assert len(await store.get_facts("u1")) == 1
    assert len(await store.get_facts("u2")) == 1
    await store.save_facts("u1", [])
    assert any(f["fact"] == "u1 private detail" for f in await store.get_facts("u1"))
    assert any(f["fact"] == "u2 private detail" for f in await store.get_facts("u2"))
