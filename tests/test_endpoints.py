"""Endpoint tests through the TestClient fixture in conftest.py."""
from cloxy import config, memory, proxy, server


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
    assert h2["content"] == "Sub Two"
    assert h2["from_cache"] is False


def test_fetch_second_call_is_cached_and_original_not_mutated(client):
    first = client.post("/fetch", json={"url": "https://site.example/page"}).json()
    second = client.post("/fetch", json={"url": "https://site.example/page"}).json()
    assert first["from_cache"] is False and second["from_cache"] is True
    assert proxy.cache[proxy.cache_key("https://site.example/page", "clean")]["from_cache"] is False


def test_fetch_blocks_redirect_to_private(client):
    r = client.post("/fetch", json={"url": "https://site.example/bounce"})
    assert r.status_code == 400
    assert "169.254.169.254" in r.json()["error"]


def test_fetch_follows_safe_relative_redirect(client):
    r = client.post("/fetch", json={"url": "https://site.example/hop", "mode": "raw"})
    assert r.status_code == 200
    assert r.json()["final_url"] == "https://site.example/page"


def test_fetch_caps_body(client):
    r = client.post("/fetch", json={"url": "https://site.example/big", "mode": "raw"})
    assert len(r.json()["content"]) == config.MAX_CONTENT_LENGTH


def test_fetch_rejects_binary(client):
    assert client.post("/fetch", json={"url": "https://site.example/file.pdf"}).status_code == 415


def test_fetch_validation(client):
    assert client.post("/fetch", json={"url": "https://site.example/page",
                                       "mode": "extract"}).status_code == 400
    assert client.post("/fetch", json={"url": "https://site.example/page",
                                       "mode": "pdf"}).status_code == 400


# ---------------------------------------------------------------------------
# /search and /verify
# ---------------------------------------------------------------------------

def test_search_finds_pattern(client):
    r = client.post("/search", json={"url": "https://site.example/page", "pattern": "REVENUE"})
    assert r.json()["total_matches"] >= 1


def test_search_blocks_redirect_to_private(client):
    assert client.post("/search", json={"url": "https://site.example/bounce",
                                        "pattern": "x"}).status_code == 400


def test_verify_returns_ranked_passages(client):
    r = client.post("/verify", json={"url": "https://site.example/page",
                                     "claim": "revenue was reported", "top_k": 2})
    body = r.json()
    assert body["total_chunks_searched"] >= 1 and len(body["matches"]) <= 2
    assert proxy.cache_key("https://site.example/page", "clean") in proxy.cache


# ---------------------------------------------------------------------------
# memory: ingest_text → recall (hybrid / dense / keyword, filters) → forget
# ---------------------------------------------------------------------------

NOTE_A = "The staging cluster listens on port 9055 and uses WAL mode for sqlite."
NOTE_B = "Lunch plan: tacos on Thursday, the good truck near the office."


def test_ingest_text_then_recall_and_forget(client):
    r = client.post("/ingest_text", json={"text": NOTE_A, "source": "notes", "project": "/w/infra"})
    assert r.status_code == 200 and r.json()["chunks_stored"] == 1
    assert client.post("/ingest_text", json={"text": NOTE_A, "source": "notes"}).json()["chunks_stored"] == 0
    assert client.post("/ingest_text", json={"text": NOTE_B, "source": "notes",
                                             "project": "/w/life"}).json()["chunks_stored"] == 1

    hits = client.post("/recall", json={"query": "which port does staging use"}).json()["results"]
    assert hits and hits[0]["content"] == NOTE_A
    assert hits[0]["project"] == "/w/infra" and hits[0]["ts_start"]
    assert hits[0]["score"] > 0 and "similarity" in hits[0]

    # keyword mode finds the exact token
    kw = client.post("/recall", json={"query": "9055", "mode": "keyword"}).json()["results"]
    assert [h["content"] for h in kw] == [NOTE_A]

    # dense mode still works on its own
    dense = client.post("/recall", json={"query": "tacos truck", "mode": "dense"}).json()["results"]
    assert dense[0]["content"] == NOTE_B

    # project filter
    only_life = client.post("/recall", json={"query": "port 9055", "project": "life"}).json()["results"]
    assert [h["content"] for h in only_life] == [NOTE_B]
    assert client.post("/recall", json={"query": "x", "project": "nope"}).json()["results"] == []

    # date filters (everything was ingested just now)
    assert client.post("/recall", json={"query": "9055", "since": "2999-01-01"}).json()["results"] == []
    assert client.post("/recall", json={"query": "9055", "until": "2000-01-01"}).json()["results"] == []
    assert client.post("/recall", json={"query": "9055", "since": "2000-01-01"}).json()["results"]

    projects = client.get("/projects").json()["projects"]
    assert {p["project"] for p in projects} >= {"/w/infra", "/w/life"}

    stats = client.get("/memory_stats").json()
    assert stats["keyword_index_size"] == stats["total_memories"] == stats["vector_index_size"]

    assert client.post("/forget", json={"source_prefix": "notes"}).json()["deleted"] == 2
    assert client.post("/recall", json={"query": NOTE_A}).json()["results"] == []


def test_recall_validation(client):
    assert client.post("/recall", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/recall", json={"query": "x", "top_k": 1000}).status_code == 422
    assert client.post("/recall", json={"query": "x", "mode": "psychic"}).status_code == 400


def test_ingest_text_size_cap(client):
    r = client.post("/ingest_text", json={"text": "x" * (config.MAX_INGEST_CHARS + 1)})
    assert r.status_code == 422


def test_delete_missing_memory_404(client):
    assert client.delete("/memory/999999").status_code == 404


def test_reindex_and_status_endpoints(client):
    assert client.post("/reindex").json()["status"] == "rebuilt"
    st = client.get("/ingest_status").json()
    assert st["running"] is False and "files_tracked" in st
    h = client.get("/health").json()
    assert h["status"] == "OK" and h["watcher"]["running"] is False
    assert "Give your local AI eyes and memory" in client.get("/").text


# ---------------------------------------------------------------------------
# chat endpoint: request shape + memory block (no model → 503, but must parse)
# ---------------------------------------------------------------------------

def test_chat_accepts_content_parts_and_max_completion_tokens(client):
    r = client.post("/v1/chat/completions", json={
        "model": "cloxy",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "max_completion_tokens": 32,
        "memory": True,
    })
    assert r.status_code == 503   # parsed fine; fails only because no LLM is configured
    assert client.get("/v1/models").json()["data"] == []


def test_chat_message_text_flattens_parts():
    m = server.ChatMessage(role="user", content=[
        {"type": "text", "text": "one"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
        {"type": "text", "text": "two"},
    ])
    assert m.text() == "one\ntwo"
    assert server.ChatMessage(role="assistant", content=None).text() == ""
    msgs = [{"role": "user", "content": "x"}]
    assert server.ChatCompletionRequest(messages=msgs).effective_max_tokens() == server.DEFAULT_MAX_TOKENS
    assert server.ChatCompletionRequest(messages=msgs, max_tokens=64,
                                        max_completion_tokens=128).effective_max_tokens() == 128


def test_memory_block_formatting():
    block = server._memory_block([
        {"content": "we chose WAL mode", "ts_start": "2026-09-10T10:00:00Z", "project": "/w/cloxy"},
        {"content": "x" * 1300, "ts_start": None, "project": None},
    ])
    assert "1. [2026-09-10 · cloxy] we chose WAL mode" in block
    assert block.endswith(" …") and len(block) < 1600
