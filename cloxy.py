#!/usr/bin/env python3
"""
CLOXY — Give your local AI eyes and memory.
Web proxy + conversation RAG for local LLMs and AI coding tools.

v3.0 — Rebuilt with:
  - Numpy matrix vector search (batch cosine sim, no Python loops)
  - aiosqlite for async-safe database access
  - SHA256 content hashing
  - FastAPI lifespan (no deprecated on_event)
  - Optional API key auth
  - TTL cache via cachetools
  - In-memory vector index with auto-reload from DB

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

# --- Config ---
PORT = int(os.environ.get("CLOXY_PORT", 9055))
DATA_DIR = os.environ.get("CLOXY_DATA_DIR", os.path.expanduser("~/.cloxy"))
DB_PATH = os.path.join(DATA_DIR, "memory.db")
API_KEY = os.environ.get("CLOXY_API_KEY", "")  # empty = no auth
USER_AGENT = os.environ.get("CLOXY_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
FETCH_TIMEOUT = float(os.environ.get("CLOXY_FETCH_TIMEOUT", 30))
MAX_CONTENT_LENGTH = 500_000

# --- RAG Config ---
EMBED_MODEL = os.environ.get("CLOXY_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
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
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ids: List[int] = []
        self._matrix: Optional[np.ndarray] = None  # (N, dim) normalized

    def load(self, ids: List[int], embeddings: List[np.ndarray]):
        """Bulk load from database on startup."""
        with self._lock:
            if not ids:
                self._ids = []
                self._matrix = None
                return
            self._ids = list(ids)
            mat = np.vstack(embeddings).astype(np.float32)
            # Normalize rows for cosine similarity via dot product
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = mat / norms

    def add(self, chunk_id: int, embedding: np.ndarray):
        """Add a single vector to the index."""
        with self._lock:
            vec = embedding.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._ids.append(chunk_id)
            if self._matrix is None:
                self._matrix = vec
            else:
                self._matrix = np.vstack([self._matrix, vec])

    def add_batch(self, ids: List[int], embeddings: List[np.ndarray]):
        """Add multiple vectors at once."""
        with self._lock:
            mat = np.vstack(embeddings).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            self._ids.extend(ids)
            if self._matrix is None:
                self._matrix = mat
            else:
                self._matrix = np.vstack([self._matrix, mat])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        """
        Returns list of (chunk_id, similarity_score) sorted by relevance.
        Single matrix multiply — O(N) but vectorized in C, not Python.
        """
        with self._lock:
            if self._matrix is None or len(self._ids) == 0:
                return []
            q = query_embedding.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
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
    if API_KEY and api_key != API_KEY:
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
    await _db_pool.commit()
    logger.info(f"Database ready at {DB_PATH}")


async def load_vector_index():
    """Load all embeddings from DB into the in-memory vector index."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
    )
    if not rows:
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    init_embedder()
    await load_vector_index()
    logger.info(f"CLOXY v3.0 ready on port {PORT}")
    yield
    await close_db()
    logger.info("CLOXY shut down")


app = FastAPI(title="CLOXY", version="3.0", lifespan=lifespan)


# =============================================================================
# EMBEDDING HELPERS
# =============================================================================

def embed_text(text: str) -> np.ndarray:
    return list(embedder.embed([text]))[0]


def embed_batch(texts: List[str]) -> List[np.ndarray]:
    return list(embedder.embed(texts))


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


def cache_key(url: str, mode: str) -> str:
    return hashlib.sha256(f"{url}:{mode}".encode()).hexdigest()


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
  v3.0 — Port {PORT} — {doc_count} memories — Auth {auth_status}

  WEB PROXY:
    POST /fetch          — Fetch and clean a URL
    POST /search         — Fetch URL and extract around a pattern

  MEMORY:
    POST /ingest_convos  — Parse Claude Code conversations into memory
    POST /ingest_text    — Store any text into memory
    POST /recall         — Search conversation memory (vector index)
    GET  /memory_stats   — Memory stats

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
        "version": "3.0",
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

    parsed = urlparse(req.url)
    if parsed.scheme not in ("http", "https"):
        return JSONResponse(status_code=400, content={"error": "URL must be http or https"})

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

            new_chunks = []
            for chunk in chunks:
                h = content_hash(chunk["text"])
                row = await db.execute("SELECT 1 FROM chunks WHERE content_hash = ?", (h,))
                exists = await row.fetchone()
                if not exists:
                    new_chunks.append((chunk["text"], h, chunk["source"]))
                else:
                    total_dupes += 1

            if new_chunks:
                texts = [c[0] for c in new_chunks]
                embeddings = embed_batch(texts)

                new_ids = []
                for (text, h, source), emb in zip(new_chunks, embeddings):
                    cursor = await db.execute(
                        "INSERT INTO chunks (content, content_hash, source, embedding) VALUES (?, ?, ?, ?)",
                        (text, h, source, pack_embedding(emb))
                    )
                    new_ids.append(cursor.lastrowid)

                await db.commit()

                # Update in-memory vector index
                vec_index.add_batch(new_ids, embeddings)
                total_stored += len(new_chunks)

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
    stored = 0

    for i, chunk in enumerate(chunks):
        h = content_hash(chunk)
        row = await db.execute("SELECT 1 FROM chunks WHERE content_hash = ?", (h,))
        exists = await row.fetchone()
        if not exists:
            emb = embed_text(chunk)
            cursor = await db.execute(
                "INSERT INTO chunks (content, content_hash, source, embedding) VALUES (?, ?, ?, ?)",
                (chunk, h, f"{req.source}:chunk{i}", pack_embedding(emb))
            )
            vec_index.add(cursor.lastrowid, emb)
            stored += 1

    await db.commit()
    return {"chunks_stored": stored, "total_chunks": len(chunks)}


@app.post("/recall", dependencies=[Depends(check_auth)])
async def recall(req: RecallRequest):
    """Semantic search via in-memory numpy vector index."""
    logger.info(f"RECALL query='{req.query[:80]}' top_k={req.top_k}")

    query_emb = embed_text(req.query)

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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("""
   _____ _     _____ __  ____   __
  / ____| |   / __ \\ \\/ /\\ \\ / /
 | |    | |  | |  | |\\  /  \\ V /
 | |    | |  | |  | |/  \\   | |
 | |____| |__| |__| / /\\ \\  | |
  \\_____|_____\\____/_/  \\_\\ |_|

  Give your local AI eyes and memory.
  v3.0 — numpy vector index · aiosqlite · async
""")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
