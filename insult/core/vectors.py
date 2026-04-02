"""Vector embedding and semantic search for user facts.

Uses sqlite-vec for in-database vector search and sentence-transformers
for local embedding generation. Hybrid search combines vector similarity
with FTS5 keyword matching via Reciprocal Rank Fusion.
"""

from __future__ import annotations

import struct
import threading
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import aiosqlite

log = structlog.get_logger()

EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"

# RRF parameters
RRF_K = 60  # smoothing constant
VECTOR_WEIGHT = 0.7
FTS_WEIGHT = 0.3


class EmbeddingModel:
    """Lazy-loaded sentence-transformers model for generating embeddings.

    Thread-safe: model loading is protected by a lock so concurrent
    callers don't race to initialize the model.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        """Load the model on first use (lazy initialization)."""
        if self._model is not None:
            return
        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
            log.info("embedding_model_loaded", model=MODEL_NAME, dim=EMBEDDING_DIM)

    def embed(self, text: str) -> list[float]:
        """Generate a 384-dimensional embedding for a single text."""
        self._ensure_model()
        # encode returns numpy array, convert to list
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts at once."""
        if not texts:
            return []
        self._ensure_model()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


# Module-level singleton (lazy — no work done until first .embed() call)
_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the singleton EmbeddingModel instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def serialize_embedding(embedding: list[float]) -> bytes:
    """Convert a list of floats to raw bytes for sqlite-vec (little-endian float32)."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


async def init_vector_tables(db: aiosqlite.Connection) -> None:
    """Create the vector search and FTS5 tables if they don't exist.

    Requires sqlite-vec extension to be loaded first.
    """
    await db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_user_facts "
        f"USING vec0(fact_id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIM}])"
    )
    await db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts "
        "USING fts5(user_id, fact, category, content=user_facts, content_rowid=id)"
    )
    await db.commit()
    log.info("vector_tables_initialized")


async def upsert_fact_vectors(db: aiosqlite.Connection, user_id: str, facts: list[dict]) -> None:
    """Embed facts and upsert into vec_user_facts + facts_fts.

    Called after save_facts() writes to user_facts table.
    facts is a list of dicts with 'id', 'fact', 'category' keys.
    """
    if not facts:
        return

    model = get_embedding_model()

    # Get fact IDs and texts from the database (just saved)
    cursor = await db.execute(
        "SELECT id, fact, category FROM user_facts WHERE user_id = ? ORDER BY id",
        (user_id,),
    )
    rows = await cursor.fetchall()
    if not rows:
        return

    fact_ids = [r[0] for r in rows]
    fact_texts = [r[1] for r in rows]
    fact_categories = [r[2] for r in rows]

    # Generate embeddings in batch
    embeddings = model.embed_batch(fact_texts)

    # Clean old entries for this user from vector table
    # (vec0 doesn't support DELETE WHERE on non-PK, so we delete by known IDs)
    # First, find old fact_ids for this user from the FTS table
    try:
        old_cursor = await db.execute(
            "SELECT rowid FROM facts_fts WHERE user_id = ?",
            (user_id,),
        )
        old_rows = await old_cursor.fetchall()
        old_ids = [r[0] for r in old_rows]
        for old_id in old_ids:
            await db.execute("DELETE FROM vec_user_facts WHERE fact_id = ?", (old_id,))
    except Exception:
        # If FTS table was empty or had issues, proceed with insert
        log.debug("vec_cleanup_old_entries_skipped")

    # Rebuild FTS entries for this user
    # Delete old FTS entries
    await db.execute(
        "DELETE FROM facts_fts WHERE user_id = ?",
        (user_id,),
    )

    # Insert new entries
    for fact_id, text, category, embedding in zip(fact_ids, fact_texts, fact_categories, embeddings, strict=False):
        blob = serialize_embedding(embedding)
        await db.execute(
            "INSERT INTO vec_user_facts (fact_id, embedding) VALUES (?, ?)",
            (fact_id, blob),
        )
        await db.execute(
            "INSERT INTO facts_fts (rowid, user_id, fact, category) VALUES (?, ?, ?, ?)",
            (fact_id, user_id, text, category),
        )

    await db.commit()
    log.info("fact_vectors_upserted", user_id=user_id, count=len(fact_ids))


async def search_facts_hybrid(
    db: aiosqlite.Connection,
    user_id: str,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Hybrid search combining vector similarity and FTS5 keyword matching.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both sources.
    Returns top-K facts sorted by RRF score.
    """
    model = get_embedding_model()
    query_embedding = model.embed(query)
    query_blob = serialize_embedding(query_embedding)

    # --- Vector search ---
    # Get all fact IDs for this user first (to filter vector results)
    user_cursor = await db.execute(
        "SELECT id FROM user_facts WHERE user_id = ?",
        (user_id,),
    )
    user_fact_ids = {r[0] for r in await user_cursor.fetchall()}

    vector_results: list[tuple[int, float]] = []
    if user_fact_ids:
        # Query more than limit to have room after user filtering
        vec_cursor = await db.execute(
            "SELECT fact_id, distance FROM vec_user_facts WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (query_blob, limit * 3),
        )
        vec_rows = await vec_cursor.fetchall()
        # Filter to only this user's facts
        vector_results = [(r[0], r[1]) for r in vec_rows if r[0] in user_fact_ids]

    # --- FTS5 search ---
    fts_results: list[tuple[int, float]] = []
    # Build FTS query: escape special chars and join with OR
    fts_terms = [w for w in query.split() if len(w) > 1]
    if fts_terms:
        # Use simple OR query for FTS5
        fts_query = " OR ".join(f'"{t}"' for t in fts_terms)
        try:
            fts_cursor = await db.execute(
                "SELECT rowid, rank FROM facts_fts WHERE facts_fts MATCH ? AND user_id = ? ORDER BY rank LIMIT ?",
                (fts_query, user_id, limit * 2),
            )
            fts_results = [(r[0], r[1]) for r in await fts_cursor.fetchall()]
        except Exception:
            # FTS query can fail on special characters — fall through gracefully
            log.debug("fts_search_failed", query=fts_query[:80])

    # --- Reciprocal Rank Fusion ---
    scores: dict[int, float] = {}

    for rank, (fact_id, _distance) in enumerate(vector_results):
        scores[fact_id] = scores.get(fact_id, 0.0) + VECTOR_WEIGHT / (RRF_K + rank + 1)

    for rank, (fact_id, _rank_score) in enumerate(fts_results):
        scores[fact_id] = scores.get(fact_id, 0.0) + FTS_WEIGHT / (RRF_K + rank + 1)

    if not scores:
        return []

    # Sort by RRF score descending, take top-K
    ranked_ids = sorted(scores.keys(), key=lambda fid: scores[fid], reverse=True)[:limit]

    # Fetch full fact data
    results = []
    for fact_id in ranked_ids:
        cursor = await db.execute(
            "SELECT id, fact, category, updated_at FROM user_facts WHERE id = ?",
            (fact_id,),
        )
        row = await cursor.fetchone()
        if row:
            results.append(
                {
                    "id": row[0],
                    "fact": row[1],
                    "category": row[2],
                    "updated_at": row[3],
                    "rrf_score": scores[fact_id],
                }
            )

    return results
