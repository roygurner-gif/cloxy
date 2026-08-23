#!/usr/bin/env python3
"""
CLOXY — Give your local AI eyes and memory.
Web proxy + conversation RAG for local LLMs and AI coding tools.

v4.0 — Adds:
  - Apple Silicon LLM bootstrap + OpenAI-compatible /v1/chat/completions (MLX)
  - Embedding runs off the event loop (non-blocking under concurrency)
  - Upsert-on-conflict ingest (no read-then-write race)
  - Delete / prune endpoints + index rebuild
  - SSRF guard on the web proxy; loopback-only bind by default
  - Embed-model/dim recorded in DB and enforced on startup

v3.1 — /verify endpoint: rank URL passages by semantic match to a claim.
v3.0 — numpy vector search, aiosqlite, SHA256 hashing, FastAPI lifespan,
       optional API key auth, TTL cache, in-memory vector index.

Usage:
    pip install fastapi uvicorn httpx trafilatura markdownify beautifulsoup4 \
                fastembed numpy aiosqlite cachetools
    python cloxy.py

    # Fetch a webpage
    curl -X POST http://localhost:9055/fetch -H "Content-Type: application/json" \
        -d '{"url": "https://example.com", "mode": "clean"}'

    # Ingest Claude Code conversations
    curl -X POST http://localhost:9055/ingest_convos -H "Content-Type: application/json" \
        -d '{"convo_dir": "~/.claude/projects"}'

    # Recall from memory
    curl -X POST http://localhost:9055/recall -H "Content-Type: application/json" \
        -d '{"query": "what did we build last week"}'

Roy Gurner | Occam Engineering | 2026
"""
import os
import sys
import time
import hashlib
import json
import re
import glob as globmod
import struct
import socket
import ipaddress
import secrets
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List
from urllib.parse import urlparse
from pathlib import Path

import httpx
import trafilatura
import numpy as np
import aiosqlite
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from fastembed import TextEmbedding
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from cachetools import TTLCache

# --- Version (single source of truth) ---
__version__ = "4.0"

# --- Config ---
PORT = int(os.environ.get("CLOXY_PORT", 9055))
# Bind loopback-only by default. Set CLOXY_HOST=0.0.0.0 to expose on the network
# (do that only behind CLOXY_API_KEY — see the SSRF note in README).
HOST = os.environ.get("CLOXY_HOST", "127.0.0.1")
DATA_DIR = os.environ.get("CLOXY_DATA_DIR", os.path.expanduser("~/.cloxy"))
DB_PATH = os.path.join(DATA_DIR, "memory.db")
API_KEY = os.environ.get("CLOXY_API_KEY", "")  # empty = no auth
USER_AGENT = os.environ.get("CLOXY_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
FETCH_TIMEOUT = float(os.environ.get("CLOXY_FETCH_TIMEOUT", 30))
MAX_CONTENT_LENGTH = 500_000
# Allow the proxy to reach private/loopback/link-local addresses. Off by default
# so an exposed instance can't be used to pivot into internal services / cloud
# metadata (169.254.169.254). Set CLOXY_ALLOW_PRIVATE_URLS=1 for local scraping.
ALLOW_PRIVATE_URLS = os.environ.get("CLOXY_ALLOW_PRIVATE_URLS", "") not in ("", "0", "false", "False")

# --- RAG Config ---
EMBED_MODEL = os.environ.get("CLOXY_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("CLOXY_EMBED_DIM", 384))
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("cloxy")

# --- Cache (TTL 15 min, max 200 entries) ---
_cache: TTLCache = TTLCache(maxsize=200, ttl=900)

# --- Globals ---
START_TIME = time.time()
embedder: TextEmbedding = None
_db_pool: aiosqlite.Connection = None


# =============================================================================
# VECTOR INDEX — In-memory numpy matrix for fast cosine similarity
# =============================================================================

class VectorIndex:
    """
    In-memory vector index backed by a normalized numpy matrix.
    Recall is a single matrix multiply — no Python loops, no full table scan.
    Auto-syncs with SQLite on insert and startup.

    Vectors are appended to a Python list and the packed matrix is rebuilt
    lazily on the next search (dirty flag). This keeps inserts O(1) amortized
    instead of O(N) per add — a large ingest was previously O(N^2) because each
    add did a full np.vstack of the whole matrix.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ids: List[int] = []
        self._vecs: List[np.ndarray] = []   # normalized (dim,) rows, append-only
        self._matrix: Optional[np.ndarray] = None  # (N, dim) rebuilt from _vecs
        self._dirty = False

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def load(self, ids: List[int], embeddings: List[np.ndarray]):
        """Bulk (re)load from database. Replaces any existing contents."""
        with self._lock:
            self._ids = list(ids)
            self._vecs = [self._normalize(e) for e in embeddings]
            self._matrix = None
            self._dirty = True

    def add(self, chunk_id: int, embedding: np.ndarray):
        """Add a single vector to the index (O(1) amortized)."""
        with self._lock:
            self._ids.append(chunk_id)
            self._vecs.append(self._normalize(embedding))
            self._dirty = True

    def add_batch(self, ids: List[int], embeddings: List[np.ndarray]):
        """Add multiple vectors at once."""
        with self._lock:
            self._ids.extend(ids)
            self._vecs.extend(self._normalize(e) for e in embeddings)
            self._dirty = True

    def remove(self, chunk_ids) -> int:
        """Drop the given chunk ids from the index. Returns count removed."""
        drop = set(chunk_ids)
        with self._lock:
            keep = [(i, v) for i, v in zip(self._ids, self._vecs) if i not in drop]
            removed = len(self._ids) - len(keep)
            self._ids = [i for i, _ in keep]
            self._vecs = [v for _, v in keep]
            self._matrix = None
            self._dirty = True
            return removed

    def _rebuild_locked(self):
        """Rebuild the packed matrix from _vecs. Caller must hold the lock."""
        if not self._vecs:
            self._matrix = None
        else:
            self._matrix = np.vstack(self._vecs).astype(np.float32)
        self._dirty = False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        """
        Returns list of (chunk_id, similarity_score) sorted by relevance.
        Single matrix multiply — O(N) but vectorized in C, not Python.
        """
        with self._lock:
            if self._dirty or self._matrix is None:
                self._rebuild_locked()
            if self._matrix is None or len(self._ids) == 0:
                return []
            q = self._normalize(query_embedding).reshape(1, -1)
            # Cosine similarity via dot product (both sides normalized)
            scores = (self._matrix @ q.T).flatten()
            k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [(self._ids[i], float(scores[i])) for i in top_indices]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)


vec_index = VectorIndex()


# =============================================================================
# AUTH
# =============================================================================

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def check_auth(api_key: Optional[str] = Depends(_api_key_header)):
    """Optional API key auth. Skipped if CLOXY_API_KEY is not set."""
    if API_KEY and not (api_key and secrets.compare_digest(api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =============================================================================
# DATABASE
# =============================================================================

async def get_db() -> aiosqlite.Connection:
    return _db_pool


async def init_db():
    """Initialize async SQLite database."""
    global _db_pool
    os.makedirs(DATA_DIR, exist_ok=True)
    _db_pool = await aiosqlite.connect(DB_PATH)
    _db_pool.row_factory = aiosqlite.Row

    await _db_pool.execute("PRAGMA journal_mode=WAL")

    await _db_pool.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            content_hash TEXT UNIQUE NOT NULL,
            source TEXT DEFAULT 'unknown',
            embedding BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await _db_pool.execute("CREATE INDEX IF NOT EXISTS idx_hash ON chunks(content_hash)")
    await _db_pool.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
    await _db_pool.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    await _db_pool.commit()
    await _check_embed_meta()
    logger.info(f"Database ready at {DB_PATH}")


async def _check_embed_meta():
    """
    Record the embedding model + dim in the DB the first time, and refuse to
    start if a later run points at a different model/dim than the stored data.
    Mixing dims silently corrupts vector search (np.vstack would blow up, or
    worse, compare incompatible spaces).
    """
    db = _db_pool
    rows = {r[0]: r[1] for r in await db.execute_fetchall("SELECT key, value FROM meta")}
    stored_model = rows.get("embed_model")
    stored_dim = rows.get("embed_dim")

    has_data = (await db.execute_fetchall("SELECT 1 FROM chunks LIMIT 1")) != []

    if stored_model is None:
        if has_data:
            # Pre-v4 DB with no meta — trust current config but warn.
            logger.warning("No embed metadata in DB; assuming current model. "
                           "If recall looks wrong, re-ingest with a clean DB.")
        await db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                         ("embed_model", EMBED_MODEL))
        await db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                         ("embed_dim", str(EMBED_DIM)))
        await db.commit()
        return

    if stored_model != EMBED_MODEL or stored_dim != str(EMBED_DIM):
        raise RuntimeError(
            f"Embedding mismatch: DB was built with {stored_model} (dim {stored_dim}) "
            f"but CLOXY_EMBED_MODEL={EMBED_MODEL} (dim {EMBED_DIM}). "
            f"Use the original model, or start with a fresh CLOXY_DATA_DIR."
        )


async def load_vector_index():
    """Load all embeddings from DB into the in-memory vector index."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
    )
    if not rows:
        vec_index.load([], [])  # clear any stale contents
        logger.info("Vector index: empty (no embeddings in DB)")
        return

    ids = []
    embeddings = []
    for row in rows:
        chunk_id = row[0]
        blob = row[1]
        n = len(blob) // 4
        vec = np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
        ids.append(chunk_id)
        embeddings.append(vec)

    vec_index.load(ids, embeddings)
    logger.info(f"Vector index: loaded {len(ids)} embeddings into memory")


async def close_db():
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None


def init_embedder():
    global embedder
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    logger.info("Embedding model loaded")


# =============================================================================
# LIFESPAN
# =============================================================================

async def _maybe_eager_load_llm():
    """If CLOXY_EAGER_LLM is set and a model is configured, load it at startup."""
    if not os.environ.get("CLOXY_EAGER_LLM"):
        return
    try:
        import json
        from pathlib import Path
        from backends import mlx_backend
        cfg_path = Path(os.environ.get("CLOXY_CONFIG", Path.home() / ".cloxy" / "config.json"))
        if not cfg_path.exists():
            logger.info("CLOXY_EAGER_LLM set but no config — run `python cli.py init` first.")
            return
        cfg = json.loads(cfg_path.read_text())
        hf_id = cfg.get("model_hf_id")
        if hf_id:
            logger.info(f"Eagerly loading LLM: {hf_id}")
            await mlx_backend.load_model(hf_id)
            logger.info("LLM ready.")
    except Exception as e:
        logger.warning(f"Eager LLM load failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    init_embedder()
    await load_vector_index()
    await _maybe_eager_load_llm()
    logger.info(f"CLOXY v{__version__} ready on {HOST}:{PORT}")
    yield
    await close_db()
    logger.info("CLOXY shut down")


app = FastAPI(title="CLOXY", version=__version__, lifespan=lifespan)


# =============================================================================
# EMBEDDING HELPERS
# =============================================================================

def embed_text(text: str) -> np.ndarray:
    return list(embedder.embed([text]))[0]


def embed_batch(texts: List[str]) -> List[np.ndarray]:
    return list(embedder.embed(texts))


async def aembed_text(text: str) -> np.ndarray:
    """Embed a single string off the event loop (fastembed is sync + CPU-bound)."""
    return await asyncio.to_thread(embed_text, text)


async def aembed_batch(texts: List[str]) -> List[np.ndarray]:
    """Embed a batch off the event loop so one request can't stall the server."""
    return await asyncio.to_thread(embed_batch, texts)


def pack_embedding(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    pos = 0
    while pos < len(text):
        end = pos + size
        chunk = text[pos:end]
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                last = chunk.rfind(sep)
                if last > size // 3:
                    chunk = chunk[:last + len(sep)]
                    end = pos + last + len(sep)
                    break
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        pos = end - overlap if end - overlap > pos else end
    return chunks


# =============================================================================
# CONVERSATION PARSER
# =============================================================================

def parse_convo_jsonl(filepath: str) -> List[dict]:
    messages = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = d.get("message", {})
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            parts.append(text)
                text = "\n".join(parts)
            elif isinstance(content, str):
                text = content.strip()
            else:
                continue

            if not text:
                continue

            text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL).strip()
            if not text:
                continue

            messages.append({
                "role": role,
                "text": text,
                "timestamp": d.get("timestamp", ""),
            })

    return messages


def chunk_conversation(messages: List[dict], session_id: str) -> List[dict]:
    dialogue = []
    for msg in messages:
        prefix = "USER" if msg["role"] == "user" else "ASSISTANT"
        dialogue.append(f"[{prefix}]: {msg['text']}")

    full_text = "\n\n".join(dialogue)
    text_chunks = chunk_text(full_text)

    return [
        {"text": c, "source": f"convo:{session_id}:chunk{i}"}
        for i, c in enumerate(text_chunks)
    ]


# =============================================================================
# HASHING
# =============================================================================

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def existing_hashes(db, hashes: List[str]) -> set:
    """
    Return the subset of `hashes` already present in the chunks table, using
    batched IN queries (respecting SQLite's bound-variable limit) instead of
    one SELECT per chunk.
    """
    found = set()
    BATCH = 500
    for i in range(0, len(hashes), BATCH):
        window = hashes[i:i + BATCH]
        placeholders = ",".join("?" * len(window))
        rows = await db.execute_fetchall(
            f"SELECT content_hash FROM chunks WHERE content_hash IN ({placeholders})",
            window,
        )
        found.update(r[0] for r in rows)
    return found


def cache_key(url: str, mode: str) -> str:
    return hashlib.sha256(f"{url}:{mode}".encode()).hexdigest()


# =============================================================================
# SSRF GUARD
# =============================================================================

def _ip_is_blocked(ip: str) -> bool:
    """Block loopback, private, link-local, and other non-global ranges."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable — refuse
    return (
        addr.is_private or addr.is_loopback or addr.is_link_local
        or addr.is_multicast or addr.is_reserved or addr.is_unspecified
    )


def validate_fetch_url(url: str) -> Optional[str]:
    """
    Return None if the URL is safe to fetch, else a human-readable reason.

    Guards the server-side proxy against SSRF: rejects non-http(s) schemes and
    (unless CLOXY_ALLOW_PRIVATE_URLS is set) any host that resolves to a
    private / loopback / link-local address — e.g. 169.254.169.254 cloud
    metadata, localhost, or RFC1918 internal services.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "URL must be http or https"

    host = parsed.hostname
    if not host:
        return "URL has no host"

    if ALLOW_PRIVATE_URLS:
        return None

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return f"Could not resolve host: {host}"

    for info in infos:
        ip = info[4][0]
        if _ip_is_blocked(ip):
            return (f"Refusing to fetch private/loopback address ({host} -> {ip}). "
                    f"Set CLOXY_ALLOW_PRIVATE_URLS=1 to allow.")
    return None


# =============================================================================
# REQUEST MODELS
# =============================================================================

class FetchRequest(BaseModel):
    url: str
    mode: str = "clean"
    selector: Optional[str] = None
    headers: Optional[dict] = None


class SearchExtract(BaseModel):
    url: str
    pattern: str


class IngestConvoRequest(BaseModel):
    convo_dir: str
    file_pattern: str = "*.jsonl"
    recursive: bool = True


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"


class RecallRequest(BaseModel):
    query: str
    top_k: int = 5


class VerifyRequest(BaseModel):
    url: str
    claim: str
    top_k: int = 3


class DeleteBySourceRequest(BaseModel):
    source_prefix: str  # matches source LIKE 'prefix%' (e.g. "convo:" or "manual")


# =============================================================================
# ENDPOINTS — WEB PROXY
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    db = await get_db()
    row = await db.execute_fetchall("SELECT COUNT(*) as cnt FROM chunks")
    doc_count = row[0][0] if row else 0
    auth_status = "ENABLED" if API_KEY else "DISABLED"
    return f"""
<pre style="color:#0f0;background:#111;padding:20px;font-family:monospace">
   _____ _     _____ __  ____   __
  / ____| |   / __ \\ \\/ /\\ \\ / /
 | |    | |  | |  | |\\  /  \\ V /
 | |    | |  | |  | |/  \\   | |
 | |____| |__| |__| / /\\ \\  | |
  \\_____|_____\\____/_/  \\_\\ |_|

  Give your local AI eyes and memory.
  v{__version__} — Port {PORT} — {doc_count} memories — Auth {auth_status}

  WEB PROXY:
    POST /fetch          — Fetch and clean a URL
    POST /search         — Fetch URL and extract around a pattern (regex/keyword)
    POST /verify         — Fetch URL and rank passages by semantic match to a claim

  MEMORY:
    POST /ingest_convos  — Parse Claude Code conversations into memory
    POST /ingest_text    — Store any text into memory
    POST /recall         — Search conversation memory (vector index)
    GET  /memory_stats   — Memory stats
    DELETE /memory/{{id}}  — Delete one memory
    POST /forget         — Delete memories by source prefix
    POST /reindex        — Rebuild the vector index from the DB

  LLM (OpenAI-compatible, MLX):
    POST /v1/chat/completions — Chat (streaming or not)
    GET  /v1/models      — The currently-loaded model

  HEALTH:
    GET  /health         — Health check

  Fetch modes: clean | raw | markdown | extract
</pre>"""


@app.get("/health")
async def health():
    db = await get_db()
    row = await db.execute_fetchall("SELECT COUNT(*) FROM chunks")
    doc_count = row[0][0] if row else 0
    return {
        "status": "OK",
        "service": "cloxy",
        "version": __version__,
        "uptime": round(time.time() - START_TIME),
        "cache_size": _cache.currsize,
        "memories": doc_count,
        "vector_index_size": vec_index.size,
        "embed_model": EMBED_MODEL,
        "auth": "enabled" if API_KEY else "disabled",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/fetch", dependencies=[Depends(check_auth)])
async def fetch(req: FetchRequest):
    logger.info(f"FETCH url={req.url} mode={req.mode}")

    blocked = validate_fetch_url(req.url)
    if blocked:
        return JSONResponse(status_code=400, content={"error": blocked})

    ckey = cache_key(req.url, req.mode)
    cached = _cache.get(ckey)
    if cached:
        cached["from_cache"] = True
        return cached

    try:
        headers = {"User-Agent": USER_AGENT}
        if req.headers:
            headers.update(req.headers)

        async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT) as client:
            resp = await client.get(req.url, headers=headers)
            resp.raise_for_status()
            html = resp.text[:MAX_CONTENT_LENGTH]
            status = resp.status_code
            final_url = str(resp.url)
    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=502, content={"error": f"HTTP {e.response.status_code}", "url": req.url})
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e), "url": req.url})

    result = {
        "url": req.url,
        "final_url": final_url,
        "status": status,
        "mode": req.mode,
        "from_cache": False,
        "fetched_at": datetime.now(timezone.utc).isoformat()
    }

    if req.mode == "raw":
        result["content"] = html

    elif req.mode == "clean":
        cleaned = trafilatura.extract(html, include_links=True, include_tables=True)
        if not cleaned:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            cleaned = soup.get_text(separator="\n", strip=True)
        result["content"] = cleaned
        result["length"] = len(cleaned) if cleaned else 0

    elif req.mode == "markdown":
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        markdown = md(str(soup), heading_style="ATX", strip=["img"])
        result["content"] = markdown.strip()
        result["length"] = len(result["content"])

    elif req.mode == "extract":
        if not req.selector:
            return JSONResponse(status_code=400, content={"error": "selector required for extract mode"})
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(req.selector)
        result["content"] = "\n---\n".join(el.get_text(strip=True) for el in elements)
        result["matches"] = len(elements)
        result["length"] = len(result["content"])

    _cache[ckey] = result
    return result


@app.post("/search", dependencies=[Depends(check_auth)])
async def search_extract(req: SearchExtract):
    logger.info(f"SEARCH url={req.url} pattern={req.pattern}")

    blocked = validate_fetch_url(req.url)
    if blocked:
        return JSONResponse(status_code=400, content={"error": blocked})

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT) as client:
            resp = await client.get(req.url, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            html = resp.text[:MAX_CONTENT_LENGTH]
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})

    cleaned = trafilatura.extract(html, include_links=True) or ""
    lines = cleaned.split("\n")
    pattern_lower = req.pattern.lower()

    matches = []
    for i, line in enumerate(lines):
        if pattern_lower in line.lower():
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            context = "\n".join(lines[start:end])
            matches.append({"line": i, "context": context})

    return {
        "url": req.url,
        "pattern": req.pattern,
        "matches": matches,
        "total_matches": len(matches)
    }


@app.post("/verify", dependencies=[Depends(check_auth)])
async def verify(req: VerifyRequest):
    """
    Fetch a URL and find passages semantically most relevant to a claim.

    Returns top-K cleaned passages with cosine similarity scores against
    the claim embedding. The calling agent reads the passages and decides
    whether they support, contradict, or fail to address the claim —
    Cloxy stays a tool, not a judge.

    Best for: rapid fact-check of a single URL against a single claim.
    """
    logger.info(f"VERIFY url={req.url} claim='{req.claim[:80]}'")

    blocked = validate_fetch_url(req.url)
    if blocked:
        return JSONResponse(status_code=400, content={"error": blocked})

    # Reuse /fetch clean-mode cache if present
    ckey = cache_key(req.url, "clean")
    cached = _cache.get(ckey)
    if cached and cached.get("content") is not None:
        content = cached["content"]
        final_url = cached.get("final_url", req.url)
        from_cache = True
    else:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT) as client:
                resp = await client.get(req.url, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                html = resp.text[:MAX_CONTENT_LENGTH]
                final_url = str(resp.url)
        except httpx.HTTPStatusError as e:
            return JSONResponse(status_code=502, content={"error": f"HTTP {e.response.status_code}", "url": req.url})
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": str(e), "url": req.url})

        cleaned = trafilatura.extract(html, include_links=False, include_tables=True)
        if not cleaned:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            cleaned = soup.get_text(separator="\n", strip=True)
        content = cleaned or ""
        from_cache = False

        # Populate the /fetch cache so a follow-up clean fetch is free
        _cache[ckey] = {
            "url": req.url,
            "final_url": final_url,
            "status": 200,
            "mode": "clean",
            "from_cache": False,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
            "length": len(content),
        }

    if not content:
        return {
            "url": req.url,
            "final_url": final_url,
            "claim": req.claim,
            "matches": [],
            "total_chunks_searched": 0,
            "from_cache": from_cache,
            "note": "No content extracted from URL",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    chunks = chunk_text(content)
    if not chunks:
        return {
            "url": req.url,
            "final_url": final_url,
            "claim": req.claim,
            "matches": [],
            "total_chunks_searched": 0,
            "from_cache": from_cache,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    # Batch-embed all chunks plus the claim, then cosine similarity
    all_embs = await aembed_batch(chunks + [req.claim])
    chunk_embs = all_embs[:-1]
    claim_emb = all_embs[-1]

    mat = np.vstack(chunk_embs).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    q = claim_emb.astype(np.float32).reshape(1, -1)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn

    scores = (mat @ q.T).flatten()
    k = min(req.top_k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    matches = [
        {
            "passage": chunks[int(i)],
            "similarity": round(float(scores[int(i)]), 4),
            "chunk_index": int(i),
            "position_pct": round(100 * int(i) / max(1, len(chunks) - 1), 1),
        }
        for i in top_idx
    ]

    return {
        "url": req.url,
        "final_url": final_url,
        "claim": req.claim,
        "matches": matches,
        "total_chunks_searched": len(chunks),
        "best_similarity": round(float(scores.max()), 4),
        "from_cache": from_cache,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# ENDPOINTS — MEMORY
# =============================================================================

@app.post("/ingest_convos", dependencies=[Depends(check_auth)])
async def ingest_convos(req: IngestConvoRequest):
    logger.info(f"INGEST_CONVOS dir={req.convo_dir} pattern={req.file_pattern}")

    convo_dir = os.path.expanduser(req.convo_dir)
    if not os.path.isdir(convo_dir):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {convo_dir}"})

    pattern = os.path.join(convo_dir, "**", req.file_pattern) if req.recursive else os.path.join(convo_dir, req.file_pattern)
    files = sorted(globmod.glob(pattern, recursive=req.recursive))
    if not files:
        return JSONResponse(status_code=400, content={"error": "No matching files found"})

    db = await get_db()
    total_chunks = 0
    total_stored = 0
    total_dupes = 0
    processed_files = 0
    errors = []

    for fpath in files:
        fname = os.path.basename(fpath)
        session_id = fname.replace(".jsonl", "")

        try:
            messages = parse_convo_jsonl(fpath)
            if len(messages) < 2:
                continue

            chunks = chunk_conversation(messages, session_id)
            total_chunks += len(chunks)

            # De-dupe within the file, then drop hashes already in the DB in one
            # batched query — so we never embed content we're going to discard.
            seen = set()
            candidates = []
            for chunk in chunks:
                h = content_hash(chunk["text"])
                if h in seen:
                    total_dupes += 1
                    continue
                seen.add(h)
                candidates.append((chunk["text"], h, chunk["source"]))

            already = await existing_hashes(db, [c[1] for c in candidates])
            new_chunks = [c for c in candidates if c[1] not in already]
            total_dupes += len(candidates) - len(new_chunks)

            if new_chunks:
                texts = [c[0] for c in new_chunks]
                embeddings = await aembed_batch(texts)

                new_ids, new_embs = [], []
                for (text, h, source), emb in zip(new_chunks, embeddings):
                    # ON CONFLICT is the race-safe backstop: a concurrent ingest
                    # of the same content just yields no row instead of raising.
                    cursor = await db.execute(
                        "INSERT INTO chunks (content, content_hash, source, embedding) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(content_hash) DO NOTHING RETURNING id",
                        (text, h, source, pack_embedding(emb))
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        total_dupes += 1
                    else:
                        new_ids.append(row[0])
                        new_embs.append(emb)

                await db.commit()

                if new_ids:
                    vec_index.add_batch(new_ids, new_embs)
                    total_stored += len(new_ids)

            processed_files += 1

        except Exception as e:
            errors.append({"file": fname, "error": str(e)[:200]})

    logger.info(f"INGEST_CONVOS_DONE files={processed_files} stored={total_stored} dupes={total_dupes}")
    return {
        "files_found": len(files),
        "files_processed": processed_files,
        "total_chunks": total_chunks,
        "chunks_stored": total_stored,
        "chunks_duplicate": total_dupes,
        "errors": errors[:10],
    }


@app.post("/ingest_text", dependencies=[Depends(check_auth)])
async def ingest_text(req: IngestTextRequest):
    logger.info(f"INGEST_TEXT source={req.source} len={len(req.text)}")

    db = await get_db()
    chunks = chunk_text(req.text)

    # De-dupe within the payload and against the DB before embedding.
    seen = set()
    candidates = []
    for i, chunk in enumerate(chunks):
        h = content_hash(chunk)
        if h in seen:
            continue
        seen.add(h)
        candidates.append((chunk, h, f"{req.source}:chunk{i}"))

    already = await existing_hashes(db, [c[1] for c in candidates])
    new_chunks = [c for c in candidates if c[1] not in already]

    stored = 0
    if new_chunks:
        embeddings = await aembed_batch([c[0] for c in new_chunks])
        new_ids, new_embs = [], []
        for (chunk, h, source), emb in zip(new_chunks, embeddings):
            cursor = await db.execute(
                "INSERT INTO chunks (content, content_hash, source, embedding) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(content_hash) DO NOTHING RETURNING id",
                (chunk, h, source, pack_embedding(emb))
            )
            row = await cursor.fetchone()
            if row is not None:
                new_ids.append(row[0])
                new_embs.append(emb)
        await db.commit()
        if new_ids:
            vec_index.add_batch(new_ids, new_embs)
            stored = len(new_ids)

    return {"chunks_stored": stored, "total_chunks": len(chunks)}


@app.post("/recall", dependencies=[Depends(check_auth)])
async def recall(req: RecallRequest):
    """Semantic search via in-memory numpy vector index."""
    logger.info(f"RECALL query='{req.query[:80]}' top_k={req.top_k}")

    query_emb = await aembed_text(req.query)

    # Single matrix multiply — no Python loop, no table scan
    results_raw = vec_index.search(query_emb, top_k=req.top_k)

    if not results_raw:
        return {"results": [], "query": req.query, "searched": vec_index.size}

    # Fetch content for matched IDs
    db = await get_db()
    results = []
    for chunk_id, similarity in results_raw:
        row = await db.execute(
            "SELECT content, source FROM chunks WHERE id = ?", (chunk_id,)
        )
        data = await row.fetchone()
        if data:
            results.append({
                "id": chunk_id,
                "content": data[0],
                "source": data[1],
                "similarity": round(similarity, 4)
            })

    return {
        "results": results,
        "query": req.query,
        "searched": vec_index.size,
    }


@app.get("/memory_stats", dependencies=[Depends(check_auth)])
async def memory_stats():
    db = await get_db()

    total_row = await db.execute_fetchall("SELECT COUNT(*) FROM chunks")
    total = total_row[0][0] if total_row else 0

    emb_row = await db.execute_fetchall("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
    with_emb = emb_row[0][0] if emb_row else 0

    sources = await db.execute_fetchall(
        "SELECT DISTINCT substr(source, 1, instr(source, ':') - 1) as src, COUNT(*) FROM chunks GROUP BY src"
    )

    return {
        "total_memories": total,
        "with_embeddings": with_emb,
        "vector_index_size": vec_index.size,
        "sources": {s[0] if s[0] else "other": s[1] for s in sources},
        "cache_size": _cache.currsize,
        "uptime": round(time.time() - START_TIME),
        "embed_model": EMBED_MODEL,
        "db_path": DB_PATH,
        "vector_engine": "numpy_matrix",
    }


@app.delete("/memory/{chunk_id}", dependencies=[Depends(check_auth)])
async def delete_memory(chunk_id: int):
    """Delete a single memory by id, keeping the vector index in sync."""
    db = await get_db()
    cursor = await db.execute("DELETE FROM chunks WHERE id = ? RETURNING id", (chunk_id,))
    row = await cursor.fetchone()
    await db.commit()
    if row is None:
        return JSONResponse(status_code=404, content={"error": f"No memory with id {chunk_id}"})
    removed = vec_index.remove([chunk_id])
    return {"deleted": chunk_id, "index_removed": removed}


@app.post("/forget", dependencies=[Depends(check_auth)])
async def forget(req: DeleteBySourceRequest):
    """
    Delete every memory whose source starts with `source_prefix`
    (e.g. "convo:" to drop all ingested conversations, or a single session id).
    Keeps the in-memory vector index consistent with the DB.
    """
    logger.info(f"FORGET source_prefix={req.source_prefix!r}")
    db = await get_db()
    like = req.source_prefix.replace("%", r"\%").replace("_", r"\_") + "%"
    cursor = await db.execute(
        "DELETE FROM chunks WHERE source LIKE ? ESCAPE '\\' RETURNING id", (like,)
    )
    rows = await cursor.fetchall()
    await db.commit()
    ids = [r[0] for r in rows]
    removed = vec_index.remove(ids) if ids else 0
    return {"deleted": len(ids), "index_removed": removed, "source_prefix": req.source_prefix}


@app.post("/reindex", dependencies=[Depends(check_auth)])
async def reindex():
    """
    Rebuild the in-memory vector index from the database. Use after manual DB
    edits, or if you ever suspect the index and DB have drifted.
    """
    await load_vector_index()
    return {"status": "rebuilt", "vector_index_size": vec_index.size}


# =============================================================================
# LLM CHAT (OpenAI-compatible) — Apple Silicon via MLX
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None              # ignored; we serve the loaded model
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False


async def _ensure_llm_loaded():
    """Lazy-load the LLM the first time a chat request comes in."""
    from backends import mlx_backend
    if mlx_backend.is_loaded():
        return
    import json
    from pathlib import Path
    cfg_path = Path(os.environ.get("CLOXY_CONFIG", Path.home() / ".cloxy" / "config.json"))
    if not cfg_path.exists():
        raise HTTPException(
            status_code=503,
            detail="No LLM configured. Run `python cli.py init` to pick a model.",
        )
    cfg = json.loads(cfg_path.read_text())
    hf_id = cfg.get("model_hf_id")
    if not hf_id:
        raise HTTPException(status_code=503, detail="Config has no model_hf_id.")
    await mlx_backend.load_model(hf_id)


@app.get("/v1/models", dependencies=[Depends(check_auth)])
async def list_models():
    """OpenAI-compatible model list. Returns the single currently-loaded model."""
    from backends import mlx_backend
    current = mlx_backend.current_model()
    if not current:
        return {"object": "list", "data": []}
    return {
        "object": "list",
        "data": [{
            "id": current,
            "object": "model",
            "created": int(START_TIME),
            "owned_by": "cloxy-mlx",
        }],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint, MLX-backed."""
    await _ensure_llm_loaded()
    from backends import mlx_backend
    messages = [m.model_dump() for m in req.messages]

    if not req.stream:
        return await mlx_backend.generate_chat(
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    # Streaming: Server-Sent Events in OpenAI format.
    from fastapi.responses import StreamingResponse

    async def event_stream():
        completion_id = f"cloxy-{int(time.time() * 1000)}"
        model_id = mlx_backend.current_model() or "unknown"

        async for fragment in mlx_backend.stream_chat(
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        ):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": fragment},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk + DONE sentinel
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# =============================================================================
# MAIN
# =============================================================================

def _preflight_embed_check():
    """
    Synchronous embed-model/dim check before uvicorn starts, so a mismatch
    exits the process cleanly with a clear message instead of failing inside
    the async lifespan (where uvicorn can linger).
    """
    import sqlite3
    if not os.path.exists(DB_PATH):
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = {k: v for k, v in conn.execute("SELECT key, value FROM meta").fetchall()}
        conn.close()
    except sqlite3.Error:
        return  # pre-v4 DB without a meta table; the lifespan path handles it
    stored_model, stored_dim = rows.get("embed_model"), rows.get("embed_dim")
    if stored_model and (stored_model != EMBED_MODEL or stored_dim != str(EMBED_DIM)):
        sys.exit(
            f"\nCLOXY refusing to start: embedding mismatch.\n"
            f"  DB was built with: {stored_model} (dim {stored_dim})\n"
            f"  You configured:    {EMBED_MODEL} (dim {EMBED_DIM})\n"
            f"Use the original model, or start with a fresh CLOXY_DATA_DIR.\n"
        )


if __name__ == "__main__":
    import uvicorn
    _preflight_embed_check()
    print("""
   _____ _     _____ __  ____   __
  / ____| |   / __ \\ \\/ /\\ \\ / /
 | |    | |  | |  | |\\  /  \\ V /
 | |    | |  | |  | |/  \\   | |
 | |____| |__| |__| / /\\ \\  | |
  \\_____|_____\\____/_/  \\_\\ |_|

  Local AI with eyes and memory — native to your Mac.
  v{ver} — MLX bootstrap · OpenAI-compatible /v1/chat/completions
""".format(ver=__version__))
    if HOST == "0.0.0.0" and not API_KEY:
        logger.warning("Binding 0.0.0.0 with NO api key — the proxy + RAG are "
                       "open to your whole network. Set CLOXY_API_KEY.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
