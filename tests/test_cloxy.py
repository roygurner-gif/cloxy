"""
Unit tests for Cloxy's pure logic: chunking, hashing, the vector index,
the SSRF guard, and batched dedupe. None of these need the embedding model,
the MLX backend, or the network, so they run fast in CI.
"""
import asyncio

import numpy as np
import pytest

import cloxy


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

def test_short_text_is_single_chunk():
    chunks = cloxy.chunk_text("hello world")
    assert chunks == ["hello world"]


def test_long_text_splits_and_covers_content():
    text = ("The quick brown fox. " * 400).strip()  # ~8000 chars
    chunks = cloxy.chunk_text(text, size=1500, overlap=200)
    assert len(chunks) > 1
    # Every chunk stays within a reasonable bound of the target size.
    assert all(len(c) <= 1500 + 5 for c in chunks)
    # No content silently dropped: the first sentence survives.
    assert chunks[0].startswith("The quick brown fox")


def test_chunk_text_terminates_on_pathological_input():
    # size <= overlap must not spin forever.
    chunks = cloxy.chunk_text("x" * 100, size=50, overlap=60)
    assert isinstance(chunks, list) and chunks


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------

def test_content_hash_is_stable_and_distinct():
    assert cloxy.content_hash("abc") == cloxy.content_hash("abc")
    assert cloxy.content_hash("abc") != cloxy.content_hash("abd")


# ---------------------------------------------------------------------------
# vector index
# ---------------------------------------------------------------------------

def _unit(*vals):
    return np.array(vals, dtype=np.float32)


def test_vector_index_ranks_by_cosine_similarity():
    idx = cloxy.VectorIndex()
    idx.add(1, _unit(1.0, 0.0))
    idx.add(2, _unit(0.0, 1.0))
    idx.add(3, _unit(0.9, 0.1))

    results = idx.search(_unit(1.0, 0.0), top_k=3)
    ids = [cid for cid, _ in results]
    assert ids[0] == 1          # exact match first
    assert ids[1] == 3          # near match second
    # scores are sorted descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_index_remove_keeps_state_consistent():
    idx = cloxy.VectorIndex()
    idx.add_batch([1, 2, 3], [_unit(1, 0), _unit(0, 1), _unit(1, 1)])
    assert idx.size == 3

    removed = idx.remove([2])
    assert removed == 1
    assert idx.size == 2

    ids = [cid for cid, _ in idx.search(_unit(0, 1), top_k=5)]
    assert 2 not in ids


def test_vector_index_empty_search_is_safe():
    idx = cloxy.VectorIndex()
    assert idx.search(_unit(1, 0), top_k=5) == []


def test_vector_index_lazy_rebuild_after_add():
    idx = cloxy.VectorIndex()
    idx.add(1, _unit(1, 0))
    idx.search(_unit(1, 0), top_k=1)      # forces a rebuild
    idx.add(2, _unit(0, 1))               # marks dirty again
    ids = [cid for cid, _ in idx.search(_unit(0, 1), top_k=2)]
    assert ids[0] == 2                     # newly added vector is searchable


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

def _patch_resolve(monkeypatch, ip):
    monkeypatch.setattr(
        cloxy.socket, "getaddrinfo",
        lambda host, *a, **k: [(2, 1, 6, "", (ip, 0))],
    )


def test_ssrf_rejects_non_http_scheme():
    assert cloxy.validate_fetch_url("file:///etc/passwd") is not None
    assert cloxy.validate_fetch_url("ftp://example.com") is not None


def test_ssrf_blocks_cloud_metadata(monkeypatch):
    monkeypatch.setattr(cloxy, "ALLOW_PRIVATE_URLS", False)
    _patch_resolve(monkeypatch, "169.254.169.254")
    assert cloxy.validate_fetch_url("http://metadata.internal/latest") is not None


def test_ssrf_blocks_loopback_and_private(monkeypatch):
    monkeypatch.setattr(cloxy, "ALLOW_PRIVATE_URLS", False)
    for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.10"):
        _patch_resolve(monkeypatch, ip)
        assert cloxy.validate_fetch_url("http://internal.example/") is not None


def test_ssrf_allows_public_host(monkeypatch):
    monkeypatch.setattr(cloxy, "ALLOW_PRIVATE_URLS", False)
    _patch_resolve(monkeypatch, "93.184.216.34")  # example.com
    assert cloxy.validate_fetch_url("https://example.com") is None


def test_ssrf_opt_in_allows_private(monkeypatch):
    monkeypatch.setattr(cloxy, "ALLOW_PRIVATE_URLS", True)
    # getaddrinfo shouldn't even be consulted when private is allowed.
    assert cloxy.validate_fetch_url("http://localhost:8080") is None


# ---------------------------------------------------------------------------
# batched dedupe against the DB
# ---------------------------------------------------------------------------

async def _make_db():
    import aiosqlite
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.execute(
        "CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT, "
        "content_hash TEXT UNIQUE, source TEXT, embedding BLOB)"
    )
    await db.commit()
    return db


def test_existing_hashes_returns_only_known(monkeypatch):
    async def run():
        db = await _make_db()
        await db.execute(
            "INSERT INTO chunks (content, content_hash, source) VALUES (?, ?, ?)",
            ("hi", "hash_a", "manual"),
        )
        await db.commit()
        found = await cloxy.existing_hashes(db, ["hash_a", "hash_b"])
        await db.close()
        return found

    found = asyncio.run(run())
    assert found == {"hash_a"}


def test_existing_hashes_batches_past_sqlite_var_limit():
    async def run():
        db = await _make_db()
        # Insert 1200 rows, query for all of them — exercises the >500 batching.
        for i in range(1200):
            await db.execute(
                "INSERT INTO chunks (content, content_hash, source) VALUES (?, ?, ?)",
                (f"c{i}", f"h{i}", "manual"),
            )
        await db.commit()
        found = await cloxy.existing_hashes(db, [f"h{i}" for i in range(1200)])
        await db.close()
        return found

    found = asyncio.run(run())
    assert len(found) == 1200
