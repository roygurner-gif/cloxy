"""Pure-logic tests for the memory module: chunking, hashing, vector index, FTS query."""
import struct

import numpy as np

from cloxy import config, memory


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


# ---------------------------------------------------------------------------
# embedding prefixes + meta check + reembed
# ---------------------------------------------------------------------------

def test_query_and_passage_prefixes_are_applied(monkeypatch):
    seen = []

    class Spy:
        def embed(self, texts):
            for t in texts:
                seen.append(t)
                yield np.ones(4, dtype=np.float32)

    monkeypatch.setattr(memory, "embedder", Spy())
    monkeypatch.setattr(config, "EMBED_QUERY_PREFIX", "query: ")
    monkeypatch.setattr(config, "EMBED_PASSAGE_PREFIX", "passage: ")
    memory.embed_query("boat plan")
    memory.embed_passages(["a", "b"])
    memory.embed_text("raw")            # symmetric use (/verify) stays untouched
    assert seen == ["query: boat plan", "passage: a", "passage: b", "raw"]


def test_embed_meta_problem_cases(monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", "m")
    monkeypatch.setattr(config, "EMBED_DIM", 4)
    monkeypatch.setattr(config, "EMBED_PASSAGE_PREFIX", "passage: ")
    ok = {"embed_model": "m", "embed_dim": "4", "embed_passage_prefix": "passage: "}
    assert memory.embed_meta_problem({}, has_data=True) is None          # pre-v4 DB: adopt
    assert memory.embed_meta_problem(ok, has_data=True) is None
    assert "reembed" in memory.embed_meta_problem({**ok, "embed_model": "other"}, True)
    assert "reembed" in memory.embed_meta_problem({**ok, "embed_dim": "8"}, True)
    # A pre-5.1 DB never recorded a prefix: with data its vectors are raw text.
    legacy = {"embed_model": "m", "embed_dim": "4"}
    assert "passage" in memory.embed_meta_problem(legacy, has_data=True)
    assert memory.embed_meta_problem(legacy, has_data=False) is None
    monkeypatch.setattr(config, "EMBED_PASSAGE_PREFIX", "")
    assert memory.embed_meta_problem(legacy, has_data=True) is None


def test_reembed_round_trip(tmp_path, monkeypatch):
    """Store raw → switch to a prefix → startup refuses → reembed → startup OK, vectors changed."""
    import asyncio
    import pytest
    from conftest import FakeEmbedder

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setattr(config, "EMBED_QUERY_PREFIX", "")
    monkeypatch.setattr(config, "EMBED_PASSAGE_PREFIX", "")
    monkeypatch.setattr(memory, "embedder", FakeEmbedder())

    async def blobs_and_meta():
        rows = await memory.db.execute_fetchall("SELECT embedding FROM chunks ORDER BY id")
        meta = {r[0]: r[1] for r in await memory.db.execute_fetchall("SELECT key, value FROM meta")}
        return [r[0] for r in rows], meta

    async def build():
        await memory.init_db()
        res = await memory.store_chunks([memory.ChunkIn(text="the boat plan is paused", source="t"),
                                         memory.ChunkIn(text="power failure in the lab", source="t")])
        assert res.stored == 2
        out = await blobs_and_meta()
        await memory.close_db()
        return out
    before, meta = asyncio.run(build())
    assert meta["embed_passage_prefix"] == ""
    assert memory.preflight_embed_check() is None

    monkeypatch.setattr(config, "EMBED_PASSAGE_PREFIX", "passage: ")
    assert "reembed" in memory.preflight_embed_check()

    async def refused():
        try:
            await memory.init_db()
        finally:
            await memory.close_db()
    with pytest.raises(RuntimeError, match="reembed"):
        asyncio.run(refused())

    async def fix():
        await memory.init_db(check_embed=False)
        seen = []
        n = await memory.reembed(batch_size=1, progress=lambda d, t: seen.append((d, t)))
        out = await blobs_and_meta()
        await memory.close_db()
        return n, seen, out
    n, seen, (after, meta) = asyncio.run(fix())
    assert n == 2 and seen == [(1, 2), (2, 2)]
    assert meta["embed_passage_prefix"] == "passage: "
    assert all(a != b for a, b in zip(after, before))
    assert memory.vec_index.size == 2
    assert memory.preflight_embed_check() is None

    async def starts():
        await memory.init_db()
        await memory.close_db()
    asyncio.run(starts())
