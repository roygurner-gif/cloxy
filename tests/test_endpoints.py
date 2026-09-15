"""
Endpoint tests. The app runs under Starlette's TestClient with:
  - a deterministic fake embedder (no model download),
  - a temp SQLite DB,
  - httpx.MockTransport standing in for the internet,
  - DNS patched so the SSRF guard sees whatever IP the test wants.
No MLX, no network.
"""
import hashlib

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import cloxy


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic 384-dim vectors seeded from the text's hash."""
    def embed(self, texts):
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            yield np.random.default_rng(seed).standard_normal(cloxy.EMBED_DIM).astype(np.float32)


# Everything resolves public unless listed here.
DNS = {"metadata.internal": "169.254.169.254", "169.254.169.254": "169.254.169.254"}
PUBLIC_IP = "93.184.216.34"


def _fake_getaddrinfo(host, *a, **k):
    return [(2, 1, 6, "", (DNS.get(host, PUBLIC_IP), 0))]


PAGES = {
    "https://site.example/page": httpx.Response(
        200, headers={"content-type": "text/html"},
        content=b"<html><body><h1>Title One</h1><h2>Sub Two</h2>"
                b"<p>" + b"Body text about quarterly revenue. " * 40 + b"</p></body></html>",
    ),
    "https://site.example/bounce": httpx.Response(
        302, headers={"location": "http://metadata.internal/latest/meta-data/"}),
    "https://site.example/hop": httpx.Response(
        301, headers={"location": "/page"}),
    "https://site.example/big": httpx.Response(
        200, headers={"content-type": "text/plain"}, content=b"a" * (cloxy.MAX_CONTENT_LENGTH * 2)),
    "https://site.example/file.pdf": httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4"),
}


def _handler(request: httpx.Request) -> httpx.Response:
    return PAGES.get(str(request.url), httpx.Response(404))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    data_dir = tmp_path_factory.mktemp("cloxy")
    mp.setattr(cloxy, "DATA_DIR", str(data_dir))
    mp.setattr(cloxy, "DB_PATH", str(data_dir / "memory.db"))
    mp.setattr(cloxy, "init_embedder", lambda: setattr(cloxy, "embedder", FakeEmbedder()))
    mp.setattr(cloxy, "ALLOW_PRIVATE_URLS", False)
    mp.setattr(cloxy.socket, "getaddrinfo", _fake_getaddrinfo)
    mp.setattr(cloxy, "_make_client",
               lambda: httpx.AsyncClient(transport=httpx.MockTransport(_handler)))
    with TestClient(cloxy.app) as c:
        yield c
    mp.undo()


@pytest.fixture(autouse=True)
def clear_cache():
    cloxy._cache.clear()


# ---------------------------------------------------------------------------
# /fetch
# ---------------------------------------------------------------------------

def test_fetch_clean(client):
    r = client.post("/fetch", json={"url": "https://site.example/page"})
    assert r.status_code == 200
    body = r.json()
    assert "quarterly revenue" in body["content"]
    assert body["from_cache"] is False


def test_fetch_cache_keyed_by_selector(client):
    h1 = client.post("/fetch", json={"url": "https://site.example/page",
                                     "mode": "extract", "selector": "h1"}).json()
    h2 = client.post("/fetch", json={"url": "https://site.example/page",
                                     "mode": "extract", "selector": "h2"}).json()
    assert h1["content"] == "Title One"
    assert h2["content"] == "Sub Two"          # was served the h1 result before
    assert h2["from_cache"] is False


def test_fetch_second_call_is_cached_and_original_not_mutated(client):
    first = client.post("/fetch", json={"url": "https://site.example/page"}).json()
    second = client.post("/fetch", json={"url": "https://site.example/page"}).json()
    assert first["from_cache"] is False
    assert second["from_cache"] is True
    assert cloxy._cache[cloxy.cache_key("https://site.example/page", "clean")]["from_cache"] is False


def test_fetch_blocks_redirect_to_private(client):
    r = client.post("/fetch", json={"url": "https://site.example/bounce"})
    assert r.status_code == 400
    assert "169.254.169.254" in r.json()["error"]


def test_fetch_follows_safe_relative_redirect(client):
    r = client.post("/fetch", json={"url": "https://site.example/hop", "mode": "raw"})
    assert r.status_code == 200
    assert r.json()["final_url"] == "https://site.example/page"
    assert "Title One" in r.json()["content"]


def test_fetch_caps_body(client):
    r = client.post("/fetch", json={"url": "https://site.example/big", "mode": "raw"})
    assert r.status_code == 200
    assert len(r.json()["content"]) == cloxy.MAX_CONTENT_LENGTH


def test_fetch_rejects_binary(client):
    r = client.post("/fetch", json={"url": "https://site.example/file.pdf"})
    assert r.status_code == 415


def test_fetch_extract_requires_selector(client):
    r = client.post("/fetch", json={"url": "https://site.example/page", "mode": "extract"})
    assert r.status_code == 400


def test_fetch_unknown_mode(client):
    r = client.post("/fetch", json={"url": "https://site.example/page", "mode": "pdf"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /search and /verify share the guard
# ---------------------------------------------------------------------------

def test_search_finds_pattern(client):
    r = client.post("/search", json={"url": "https://site.example/page", "pattern": "REVENUE"})
    assert r.status_code == 200
    assert r.json()["total_matches"] >= 1


def test_search_blocks_redirect_to_private(client):
    r = client.post("/search", json={"url": "https://site.example/bounce", "pattern": "x"})
    assert r.status_code == 400


def test_verify_returns_ranked_passages(client):
    r = client.post("/verify", json={"url": "https://site.example/page",
                                     "claim": "revenue was reported", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total_chunks_searched"] >= 1
    assert len(body["matches"]) <= 2
    # populated the clean-mode /fetch cache
    assert cloxy.cache_key("https://site.example/page", "clean") in cloxy._cache


# ---------------------------------------------------------------------------
# memory round trip
# ---------------------------------------------------------------------------

def test_ingest_text_then_recall_and_forget(client):
    text = "The staging cluster listens on port 9055 and uses WAL mode."
    r = client.post("/ingest_text", json={"text": text, "source": "test-notes"})
    assert r.status_code == 200
    assert r.json()["chunks_stored"] == 1

    # identical content is deduped
    assert client.post("/ingest_text", json={"text": text, "source": "test-notes"}).json()["chunks_stored"] == 0

    hits = client.post("/recall", json={"query": text, "top_k": 3}).json()["results"]
    assert hits and hits[0]["content"] == text
    assert hits[0]["source"] == "test-notes:chunk0"

    r = client.post("/forget", json={"source_prefix": "test-notes"})
    assert r.json()["deleted"] == 1
    assert client.post("/recall", json={"query": text}).json()["results"] == []


def test_recall_top_k_is_bounded(client):
    assert client.post("/recall", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/recall", json={"query": "x", "top_k": 1000}).status_code == 422


def test_ingest_text_size_cap(client):
    r = client.post("/ingest_text", json={"text": "x" * (cloxy.MAX_INGEST_CHARS + 1)})
    assert r.status_code == 422


def test_delete_missing_memory_404(client):
    assert client.delete("/memory/999999").status_code == 404


# ---------------------------------------------------------------------------
# chat endpoint request shape (no model configured → 503, but must parse)
# ---------------------------------------------------------------------------

def test_chat_accepts_content_parts_and_max_completion_tokens(client, monkeypatch, tmp_path):
    monkeypatch.setenv("CLOXY_CONFIG", str(tmp_path / "nope.json"))
    r = client.post("/v1/chat/completions", json={
        "model": "cloxy",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "max_completion_tokens": 32,
    })
    # Parsed fine (would have been 422 before); fails only because no LLM is set up.
    assert r.status_code == 503
