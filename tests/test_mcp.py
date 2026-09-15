"""MCP tool functions against a mocked Cloxy HTTP server."""
import asyncio

import httpx
import pytest

from cloxy import mcp_server


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/recall":
        return httpx.Response(200, json={"results": [{
            "id": 7, "content": "we chose WAL mode", "score": 0.0321, "source": "convo:abc:0",
            "ts_start": "2026-09-10T10:00:00Z", "project": "/w/cloxy",
            "metadata": {"title": "DB tuning"},
        }]})
    if path == "/ingest_text":
        return httpx.Response(200, json={"chunks_stored": 1, "total_chunks": 1, "ids": [9]})
    if path == "/fetch":
        return httpx.Response(200, json={"final_url": "https://x.example/", "length": 5,
                                         "content": "hello"})
    if path == "/projects":
        return httpx.Response(200, json={"projects": [{
            "project": "/w/cloxy", "chunks": 12, "sessions": 2,
            "first_seen": "2026-09-01T00:00:00Z", "last_seen": "2026-09-14T00:00:00Z"}]})
    if path == "/verify":
        return httpx.Response(400, json={"error": "Refusing to fetch private/loopback address"})
    if path.startswith("/memory/"):
        return httpx.Response(404, json={"error": "nope"})
    return httpx.Response(404, json={"error": "unknown"})


@pytest.fixture(autouse=True)
def mock_server(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "_client",
        lambda: httpx.AsyncClient(base_url="http://cloxy.test",
                                  transport=httpx.MockTransport(_handler)))


def test_recall_formats_hits():
    out = asyncio.run(mcp_server.recall("wal", project="cloxy"))
    assert out.startswith("#1 [2026-09-10 10:00 · cloxy · DB tuning] (score 0.0321, id 7)")
    assert out.endswith("we chose WAL mode")


def test_remember_fetch_projects_forget():
    assert "Stored 1" in asyncio.run(mcp_server.remember("note"))
    assert asyncio.run(mcp_server.fetch("https://x.example/")) == "[https://x.example/ · 5 chars]\nhello"
    assert "12 chunks, 2 sessions, 2026-09-01 → 2026-09-14" in asyncio.run(mcp_server.projects())
    assert asyncio.run(mcp_server.forget(3)) == "No memory with id 3."


def test_server_errors_surface_as_messages():
    with pytest.raises(RuntimeError, match="Refusing to fetch"):
        asyncio.run(mcp_server.verify("http://localhost/", "claim"))


def test_unreachable_server_message(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("refused", request=request)
    monkeypatch.setattr(mcp_server, "_client",
                        lambda: httpx.AsyncClient(base_url="http://cloxy.test",
                                                  transport=httpx.MockTransport(boom)))
    with pytest.raises(RuntimeError, match="not reachable"):
        asyncio.run(mcp_server.recall("x"))


def test_build_server_registers_tools():
    server = mcp_server.build_server()
    assert server is not None
    # Every tool implementation has a docstring — that's the description the client sees.
    assert all(fn.__doc__ for fn in mcp_server.TOOLS)
