"""Pure-logic tests for the memory module: chunking, hashing, vector index, FTS query."""
import struct

import numpy as np

from cloxy import memory


# ---------------------------------------------------------------------------
# chunking + hashing
# ---------------------------------------------------------------------------

def test_short_text_is_single_chunk():
    assert memory.chunk_text("hello world") == ["hello world"]


def test_long_text_splits_and_covers_content():
    text = ("The quick brown fox. " * 400).strip()  # ~8000 chars
    chunks = memory.chunk_text(text, size=1500, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1500 + 5 for c in chunks)
    assert chunks[0].startswith("The quick brown fox")


def test_chunk_text_terminates_on_pathological_input():
    chunks = memory.chunk_text("x" * 100, size=50, overlap=60)
    assert isinstance(chunks, list) and chunks


def test_content_hash_is_stable_and_distinct():
    assert memory.content_hash("abc") == memory.content_hash("abc")
    assert memory.content_hash("abc") != memory.content_hash("abd")


def test_embedding_pack_unpack_roundtrip_and_legacy_layout():
    vec = np.arange(8, dtype=np.float32) / 3
    out = memory.unpack_embedding(memory.pack_embedding(vec))
    assert out.dtype == np.float32 and np.array_equal(out, vec)
    assert memory.pack_embedding(vec) == struct.pack("8f", *vec)


# ---------------------------------------------------------------------------
# vector index
# ---------------------------------------------------------------------------

def _unit(*vals):
    return np.array(vals, dtype=np.float32)


def test_vector_index_ranks_by_cosine_similarity():
    idx = memory.VectorIndex()
    idx.add(1, _unit(1.0, 0.0))
    idx.add(2, _unit(0.0, 1.0))
    idx.add(3, _unit(0.9, 0.1))
    results = idx.search(_unit(1.0, 0.0), top_k=3)
    assert [cid for cid, _ in results][:2] == [1, 3]
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_index_allowed_ids_filter():
    idx = memory.VectorIndex()
    idx.add_batch([1, 2, 3], [_unit(1, 0), _unit(0.9, 0.1), _unit(0, 1)])
    results = idx.search(_unit(1, 0), top_k=3, allowed_ids={2, 3})
    assert [cid for cid, _ in results] == [2, 3]
    assert idx.search(_unit(1, 0), top_k=3, allowed_ids=set()) == []
    assert idx.search(_unit(1, 0), top_k=3, allowed_ids={99}) == []


def test_vector_index_remove_keeps_state_consistent():
    idx = memory.VectorIndex()
    idx.add_batch([1, 2, 3], [_unit(1, 0), _unit(0, 1), _unit(1, 1)])
    assert idx.remove([2]) == 1
    assert idx.size == 2
    assert 2 not in [cid for cid, _ in idx.search(_unit(0, 1), top_k=5)]


def test_vector_index_empty_search_is_safe():
    assert memory.VectorIndex().search(_unit(1, 0), top_k=5) == []


def test_vector_index_lazy_rebuild_after_add():
    idx = memory.VectorIndex()
    idx.add(1, _unit(1, 0))
    idx.search(_unit(1, 0), top_k=1)
    idx.add(2, _unit(0, 1))
    assert [cid for cid, _ in idx.search(_unit(0, 1), top_k=2)][0] == 2


# ---------------------------------------------------------------------------
# FTS query builder + recency
# ---------------------------------------------------------------------------

def test_fts_query_quotes_tokens_and_survives_punctuation():
    q = memory.fts_query('what "port" did cloxy:9055 use? (WAL-mode)')
    assert q.startswith('"what" OR ')
    assert '"cloxy:9055"' in q and '"WAL-mode"' in q
    assert memory.fts_query("!!! ???") == ""


def test_recency_decays_with_age():
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    fresh = memory._recency(now.isoformat(), now)
    month = memory._recency((now - timedelta(days=30)).isoformat(), now)
    old = memory._recency("2020-01-01T00:00:00Z", now)
    assert fresh > 0.99 and abs(month - 0.5) < 0.01 and old < 0.01
    assert memory._recency(None, now) == 0.5
    assert memory._recency("not a date", now) == 0.5
