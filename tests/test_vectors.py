"""Tests for insult.core.vectors — embedding model, serialization, hybrid search."""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insult.core.vectors import (
    EMBEDDING_DIM,
    FTS_WEIGHT,
    RRF_K,
    VECTOR_WEIGHT,
    EmbeddingModel,
    search_facts_hybrid,
    serialize_embedding,
    upsert_fact_vectors,
)

# --- EmbeddingModel ---


class TestEmbeddingModel:
    @patch("insult.core.vectors.EmbeddingModel._ensure_model")
    def test_embed_returns_correct_dim(self, mock_ensure):
        """embed() should return a list of floats with EMBEDDING_DIM dimensions."""
        model = EmbeddingModel()
        fake_vector = [0.1] * EMBEDDING_DIM
        mock_st_model = MagicMock()
        mock_st_model.encode.return_value = MagicMock(tolist=MagicMock(return_value=fake_vector))
        model._model = mock_st_model

        result = model.embed("test text")
        assert len(result) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in result)
        mock_st_model.encode.assert_called_once_with("test text", normalize_embeddings=True)

    @patch("insult.core.vectors.EmbeddingModel._ensure_model")
    def test_embed_batch_returns_list_of_embeddings(self, mock_ensure):
        """embed_batch() should return a list of embeddings."""
        model = EmbeddingModel()
        fake_vectors = [[0.1] * EMBEDDING_DIM, [0.2] * EMBEDDING_DIM]
        mock_st_model = MagicMock()
        mock_st_model.encode.return_value = [MagicMock(tolist=MagicMock(return_value=v)) for v in fake_vectors]
        model._model = mock_st_model

        result = model.embed_batch(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == EMBEDDING_DIM

    def test_embed_batch_empty_input(self):
        """embed_batch() with empty list returns empty list without loading model."""
        model = EmbeddingModel()
        assert model.embed_batch([]) == []
        assert model._model is None  # Model should not have been loaded


# --- Serialization ---


class TestSerializeEmbedding:
    def test_produces_correct_byte_length(self):
        """serialize_embedding should produce 4 bytes per float (float32)."""
        embedding = [0.5] * EMBEDDING_DIM
        result = serialize_embedding(embedding)
        assert isinstance(result, bytes)
        assert len(result) == EMBEDDING_DIM * 4  # 4 bytes per float32

    def test_roundtrip(self):
        """Serialized embedding should roundtrip back to original values."""
        embedding = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = serialize_embedding(embedding)
        unpacked = list(struct.unpack(f"<{len(embedding)}f", blob))
        for a, b in zip(embedding, unpacked, strict=False):
            assert abs(a - b) < 1e-6


# --- RRF Scoring ---


class TestRRFScoring:
    def test_items_in_both_results_rank_higher(self):
        """Items appearing in both vector and FTS results should have higher RRF scores."""
        # Simulate: fact_id=1 appears in both, fact_id=2 only in vector, fact_id=3 only in FTS
        vector_results = [(1, 0.1), (2, 0.3)]  # (fact_id, distance)
        fts_results = [(1, -5.0), (3, -3.0)]  # (fact_id, rank)

        scores: dict[int, float] = {}
        for rank, (fact_id, _) in enumerate(vector_results):
            scores[fact_id] = scores.get(fact_id, 0.0) + VECTOR_WEIGHT / (RRF_K + rank + 1)
        for rank, (fact_id, _) in enumerate(fts_results):
            scores[fact_id] = scores.get(fact_id, 0.0) + FTS_WEIGHT / (RRF_K + rank + 1)

        # fact_id=1 should rank highest (appears in both)
        assert scores[1] > scores[2]
        assert scores[1] > scores[3]

    def test_vector_weight_dominates(self):
        """With default weights, vector match alone should score higher than FTS alone."""
        # fact_id=10 is rank 0 in vector, fact_id=20 is rank 0 in FTS
        vec_score = VECTOR_WEIGHT / (RRF_K + 1)
        fts_score = FTS_WEIGHT / (RRF_K + 1)
        assert vec_score > fts_score


# --- Hybrid Search ---


class TestSearchFactsHybrid:
    @pytest.fixture
    def mock_db(self):
        """Create a mock aiosqlite connection for search tests."""
        db = AsyncMock()
        return db

    async def test_returns_empty_when_no_user_facts(self, mock_db):
        """If user has no facts, return empty list."""
        # user_facts query returns empty
        user_cursor = AsyncMock()
        user_cursor.fetchall = AsyncMock(return_value=[])
        mock_db.execute = AsyncMock(return_value=user_cursor)

        with patch("insult.core.vectors.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.embed.return_value = [0.0] * EMBEDDING_DIM
            mock_get_model.return_value = mock_model

            results = await search_facts_hybrid(mock_db, "user123", "test query")
            assert results == []

    async def test_returns_results_with_rrf_scores(self, mock_db):
        """Hybrid search should return facts with rrf_score field."""
        user_fact_ids = [(1,), (2,), (3,)]
        vec_rows = [(1, 0.1), (2, 0.3)]
        fts_rows = [(1, -5.0), (3, -3.0)]
        fact_data = {
            1: (1, "Likes Python", "technical", 1000.0),
            2: (2, "Lives in Mexico", "location", 1001.0),
            3: (3, "Has a dog", "personal", 1002.0),
        }

        call_count = 0

        async def mock_execute(sql, params=None):
            nonlocal call_count
            cursor = AsyncMock()
            if "FROM user_facts WHERE user_id" in sql and "id" in sql.split("SELECT")[1].split("FROM")[0]:
                cursor.fetchall = AsyncMock(return_value=user_fact_ids)
            elif "vec_user_facts" in sql:
                cursor.fetchall = AsyncMock(return_value=vec_rows)
            elif "facts_fts" in sql:
                cursor.fetchall = AsyncMock(return_value=fts_rows)
            elif "FROM user_facts WHERE id" in sql:
                fid = params[0] if params else None
                row = fact_data.get(fid)
                cursor.fetchone = AsyncMock(return_value=row)
            else:
                cursor.fetchall = AsyncMock(return_value=[])
                cursor.fetchone = AsyncMock(return_value=None)
            call_count += 1
            return cursor

        mock_db.execute = mock_execute

        with patch("insult.core.vectors.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.embed.return_value = [0.0] * EMBEDDING_DIM
            mock_get_model.return_value = mock_model

            results = await search_facts_hybrid(mock_db, "user123", "Python programming")

        assert len(results) > 0
        # All results should have rrf_score
        for r in results:
            assert "rrf_score" in r
            assert "fact" in r
            assert "category" in r

        # fact_id=1 should be first (appears in both vector and FTS results)
        assert results[0]["id"] == 1


# --- Upsert Fact Vectors ---


class TestUpsertFactVectors:
    async def test_upsert_empty_facts_is_noop(self):
        """Upserting empty facts should do nothing."""
        db = AsyncMock()
        await upsert_fact_vectors(db, "user123", [])
        db.execute.assert_not_called()

    async def test_upsert_calls_embed_batch(self):
        """Upserting facts should embed them and insert into vector + FTS tables."""
        db = AsyncMock()

        # First call: SELECT from user_facts
        facts_cursor = AsyncMock()
        facts_cursor.fetchall = AsyncMock(
            return_value=[
                (10, "Likes Python", "technical"),
                (11, "Lives in Mexico", "location"),
            ]
        )

        # Second call: SELECT from facts_fts (old entries)
        fts_cursor = AsyncMock()
        fts_cursor.fetchall = AsyncMock(return_value=[])

        call_idx = 0

        async def mock_execute(sql, params=None):
            nonlocal call_idx
            call_idx += 1
            if "SELECT id, fact, category FROM user_facts" in sql:
                return facts_cursor
            if "SELECT rowid FROM facts_fts" in sql:
                return fts_cursor
            return AsyncMock()

        db.execute = mock_execute
        db.commit = AsyncMock()

        with patch("insult.core.vectors.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.embed_batch.return_value = [
                [0.1] * EMBEDDING_DIM,
                [0.2] * EMBEDDING_DIM,
            ]
            mock_get_model.return_value = mock_model

            await upsert_fact_vectors(db, "user123", [{"fact": "test", "category": "general"}])
            mock_model.embed_batch.assert_called_once()


# --- Fallback behavior (tested via memory.py mock in conftest) ---


class TestSearchFactsSemantic:
    """Test that FactsRepository.search_facts_semantic falls back gracefully.

    Post v3.5.5 the semantic-search fallback lives in FactsRepository; the
    MemoryStore facade delegates to it. This test exercises the repository
    directly because that's where the decision ("no vectors → plain get_facts")
    lives now."""

    async def test_fallback_when_vectors_unavailable(self):
        """When vectors are not available, should fall back to get_facts."""
        from insult.core.memory.repositories.facts import FactsRepository

        # Fake a ConnectionManager whose `vectors_available` is False. The
        # BaseRepository reads `self._manager.vectors_available` via its
        # `vectors_available` property; short-circuiting there is the whole
        # point of the fallback.
        manager = MagicMock()
        manager.vectors_available = False

        repo = FactsRepository(manager)
        # Replace get_facts with a fast stub so we don't hit the DB at all.
        repo.get_facts = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"id": 1, "fact": "test", "category": "general", "updated_at": 0}]
        )

        result = await repo.search_facts_semantic("user123", "test query")
        assert len(result) == 1
        assert result[0]["fact"] == "test"
        # Crucial: get_facts was awaited because the vector path was skipped.
        repo.get_facts.assert_awaited_once_with("user123")
