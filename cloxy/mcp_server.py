"""
MCP server — exposes Cloxy's eyes and memory as tools over stdio.

This is a thin client of the running Cloxy server (CLOXY_URL, default
http://127.0.0.1:9055): the server owns the database, the embedder and the
vector index, so there is exactly one of each no matter how many editors
have the MCP server open.

Register with Claude Code:   claude mcp add cloxy -- cloxy mcp
Or in .mcp.json:             {"mcpServers": {"cloxy": {"command": "cloxy", "args": ["mcp"]}}}
"""
import json
import os
from typing import Optional

import httpx

from . import __version__, config

INSTRUCTIONS = (
    "Cloxy gives you persistent memory of the user's past AI conversations and a "
    "web proxy. Use `recall` when the user refers to earlier work, decisions, or "
    "anything that might have been discussed before — especially 'what did we decide', "
    "'last time', 'that project'. Use `remember` to store facts worth keeping. Use "
    "`fetch` to read a web page as clean text and `verify` to check a claim against a page."
)


def _client() -> httpx.AsyncClient:
    headers = {"X-API-Key": config.API_KEY} if config.API_KEY else {}
    return httpx.AsyncClient(base_url=config.URL, timeout=120, headers=headers)


async def _post(path: str, payload: dict) -> dict:
    try:
        async with _client() as c:
            r = await c.post(path, json=payload)
    except httpx.ConnectError:
        raise RuntimeError(f"Cloxy server is not reachable at {config.URL}. "
                           f"Start it with `cloxy start` (or `cloxy install-service`).")
    body = r.json()
    if r.status_code >= 400:
        raise RuntimeError(body.get("error") or body.get("detail") or f"HTTP {r.status_code}")
    return body


async def _get(path: str) -> dict:
    try:
        async with _client() as c:
            r = await c.get(path)
    except httpx.ConnectError:
        raise RuntimeError(f"Cloxy server is not reachable at {config.URL}. "
                           f"Start it with `cloxy start`.")
    r.raise_for_status()
    return r.json()


def _fmt_hit(i: int, h: dict) -> str:
    when = (h.get("ts_start") or "")[:16].replace("T", " ")
    proj = os.path.basename((h.get("project") or "").rstrip("/"))
    title = (h.get("metadata") or {}).get("title")
    tag = " · ".join(b for b in (when, proj, title) if b) or h.get("source", "")
    return f"#{i} [{tag}] (score {h.get('score')}, id {h.get('id')})\n{h['content']}"


# --- Tool implementations (plain async functions; registered below) ---------

async def recall(query: str, top_k: int = 5, project: Optional[str] = None,
                 since: Optional[str] = None, until: Optional[str] = None,
                 mode: str = "hybrid") -> str:
    """
    Search the user's memory of past conversations and saved notes.

    Hybrid semantic + keyword search, most relevant first, each hit tagged with
    date and project. `project` is a substring of the project directory
    (e.g. "cloxy"); `since`/`until` are ISO dates like "2026-09-01".
    `mode` is "hybrid" (default), "dense" (meaning only) or "keyword" (exact terms).
    """
    body = await _post("/recall", {"query": query, "top_k": top_k, "project": project,
                                   "since": since, "until": until, "mode": mode})
    hits = body.get("results", [])
    if not hits:
        return f"No memories matched {query!r}" + (f" in project {project!r}" if project else "") + "."
    return "\n\n".join(_fmt_hit(i, h) for i, h in enumerate(hits, 1))


async def remember(text: str, source: str = "mcp", project: Optional[str] = None) -> str:
    """
    Store text in long-term memory so it can be recalled later. Use for
    decisions, preferences, facts, and summaries worth keeping. `source` is a
    short label (e.g. "decision", "preference"); `project` ties it to a project.
    """
    body = await _post("/ingest_text", {"text": text, "source": source, "project": project})
    return f"Stored {body['chunks_stored']} memory chunk(s) (ids {body.get('ids')})."


async def forget(memory_id: int) -> str:
    """Delete one memory by id (ids appear in `recall` results)."""
    try:
        async with _client() as c:
            r = await c.delete(f"/memory/{memory_id}")
    except httpx.ConnectError:
        raise RuntimeError(f"Cloxy server is not reachable at {config.URL}.")
    if r.status_code == 404:
        return f"No memory with id {memory_id}."
    r.raise_for_status()
    return f"Deleted memory {memory_id}."


async def fetch(url: str, mode: str = "clean", selector: Optional[str] = None) -> str:
    """
    Fetch a web page and return its content. `mode`: "clean" (main text,
    default), "markdown", "raw" (HTML), or "extract" (text of a CSS `selector`).
    Private/loopback addresses are refused.
    """
    body = await _post("/fetch", {"url": url, "mode": mode, "selector": selector})
    head = f"[{body.get('final_url', url)} · {body.get('length', len(body.get('content') or ''))} chars]\n"
    return head + (body.get("content") or "")


async def search_page(url: str, pattern: str) -> str:
    """Fetch a page and return the lines (with context) containing `pattern`."""
    body = await _post("/search", {"url": url, "pattern": pattern})
    if not body.get("matches"):
        return f"No lines matching {pattern!r} on {url}."
    return "\n---\n".join(m["context"] for m in body["matches"][:20])


async def verify(url: str, claim: str, top_k: int = 3) -> str:
    """
    Check a claim against a web page: returns the passages most relevant to
    the claim with similarity scores. You decide whether they support,
    contradict, or don't address it.
    """
    body = await _post("/verify", {"url": url, "claim": claim, "top_k": top_k})
    if not body.get("matches"):
        return f"No content extracted from {url}."
    out = [f"Claim: {claim}\nSource: {body.get('final_url', url)} "
           f"({body.get('total_chunks_searched')} passages searched)"]
    for m in body["matches"]:
        out.append(f"[similarity {m['similarity']}, {m['position_pct']}% into page]\n{m['passage']}")
    return "\n\n".join(out)


async def projects() -> str:
    """List the projects present in memory with chunk counts and date ranges."""
    body = await _get("/projects")
    rows = body.get("projects", [])
    if not rows:
        return "No project-tagged memories yet."
    return "\n".join(
        f"{p['project']}  —  {p['chunks']} chunks, {p['sessions']} sessions, "
        f"{(p['first_seen'] or '')[:10]} → {(p['last_seen'] or '')[:10]}" for p in rows)


async def memory_status() -> str:
    """Health of the memory system: counts, watcher state, model."""
    h = await _get("/health")
    s = await _get("/memory_stats")
    return json.dumps({"health": h, "stats": s}, indent=2)


TOOLS = [recall, remember, forget, fetch, search_page, verify, projects, memory_status]


def build_server():
    """Create the MCP server (mcp 2.x MCPServer, falling back to 1.x FastMCP)."""
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp < 2
        from mcp.server.fastmcp import FastMCP as Server
    server = Server("cloxy", instructions=INSTRUCTIONS)
    for fn in TOOLS:
        server.tool()(fn)
    return server


def run():
    build_server().run(transport="stdio")


if __name__ == "__main__":
    run()
