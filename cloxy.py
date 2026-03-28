#!/usr/bin/env python3
"""
CLOXY — Give your local AI eyes and memory.
Web proxy + conversation RAG for local LLMs and AI coding tools.

Usage:
    pip install fastapi uvicorn httpx trafilatura markdownify beautifulsoup4 fastembed numpy
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
"""
import os
import sys
import time
import hashlib
import json
import re
import glob as globmod
import struct
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, List
from urllib.parse import urlparse
from pathlib import Path

import httpx
import trafilatura
import numpy as np
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from fastembed import TextEmbedding
from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# --- App ---
app = FastAPI(title="CLOXY", version="2.0")
logger = logging.getLogger("cloxy")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

# --- Config ---
PORT = int(os.environ.get("CLOXY_PORT", 9055))
DATA_DIR = os.environ.get("CLOXY_DATA_DIR", os.path.expanduser("~/.cloxy"))
DB_PATH = os.path.join(DATA_DIR, "memory.db")
START_TIME = time.time()
USER_AGENT = os.environ.get("CLOXY_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
FETCH_TIMEOUT = float(os.environ.get("CLOXY_FETCH_TIMEOUT", 30))
MAX_CONTENT_LENGTH = 500_000

# --- RAG Config ---
EMBED_MODEL = os.environ.get("CLOXY_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- Cache (in-memory, 15 min TTL) ---
_cache = {}
CACHE_TTL = 900

# --- Globals ---
embedder: TextEmbedding = None
db: sqlite3.Connection = None


# =============================================================================
# DATABASE
# =============================================================================

def init_db():
    global db
    os.makedirs(DATA_DIR, exist_ok=True)
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            content_hash TEXT UNIQUE NOT NULL,
            source TEXT DEFAULT 'unknown',
            embedding BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_hash ON chunks(content_hash)")
    db.commit()
    logger.info(f"Database ready at {DB_PATH}")


def init_embedder():
    global embedder
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    logger.info("Embedding model loaded")


# =============================================================================
# EMBEDDING HELPERS
# =============================================================================

def embed_text(text: str) -> np.ndarray:
    return list(embedder.embed([text]))[0]


def embed_batch(texts: List[str]) -> List[np.ndarray]:
    return list(embedder.embed(texts))


def pack_embedding(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_embedding(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


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
            # Try to break on paragraph or sentence boundary
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
    """Parse a Claude Code conversation JSONL into clean message turns."""
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

            # Strip system reminders
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
    """Turn conversation messages into labeled chunks for RAG."""
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
# REQUEST MODELS
# =============================================================================

class FetchRequest(BaseModel):
    url: str
    mode: str = "clean"           # clean | raw | markdown | extract
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
# CACHE
# =============================================================================

def _cache_key(url: str, mode: str) -> str:
    return hashlib.md5(f"{url}:{mode}".encode()).hexdigest()


def _get_cached(key: str) -> Optional[dict]:
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
        del _cache[key]
    return None


def _set_cache(key: str, data: dict):
    _cache[key] = {"ts": time.time(), "data": data}
    if len(_cache) > 100:
        oldest = sorted(_cache.items(), key=lambda x: x[1]["ts"])
        for k, _ in oldest[:20]:
            del _cache[k]


# =============================================================================
# ENDPOINTS — WEB PROXY
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    doc_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if db else 0
    return f"""
<pre style="color:#0f0;background:#111;padding:20px;font-family:monospace">
   _____ _     _____ __  ____   __
  / ____| |   / __ \\ \\/ /\\ \\ / /
 | |    | |  | |  | |\\  /  \\ V /
 | |    | |  | |  | |/  \\   | |
 | |____| |__| |__| / /\\ \\  | |
  \\_____|_____\\____/_/  \\_\\ |_|

  Give your local AI eyes and memory.
  v2.0 — Port {PORT} — {doc_count} memories

  WEB PROXY:
    POST /fetch          — Fetch and clean a URL
    POST /search         — Fetch URL and extract around a pattern

  MEMORY:
    POST /ingest_convos  — Parse Claude Code conversations into memory
    POST /ingest_text    — Store any text into memory
    POST /recall         — Search conversation memory
    GET  /memory_stats   — Memory stats

  HEALTH:
    GET  /health         — Health check

  Fetch modes: clean | raw | markdown | extract
</pre>"""


@app.get("/health")
async def health():
    doc_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if db else 0
    return {
        "status": "OK",
        "service": "cloxy",
        "version": "2.0",
        "uptime": round(time.time() - START_TIME),
        "cache_size": len(_cache),
        "memories": doc_count,
        "embed_model": EMBED_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/fetch")
async def fetch(req: FetchRequest):
    logger.info(f"FETCH url={req.url} mode={req.mode}")

    parsed = urlparse(req.url)
    if parsed.scheme not in ("http", "https"):
        return JSONResponse(status_code=400, content={"error": "URL must be http or https"})

    ckey = _cache_key(req.url, req.mode)
    cached = _get_cached(ckey)
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

    _set_cache(ckey, result)
    return result


@app.post("/search")
async def search_extract(req: SearchExtract):
    """Fetch a URL and return lines matching a pattern with context."""
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

@app.post("/ingest_convos")
async def ingest_convos(req: IngestConvoRequest):
    """Parse Claude Code conversation JSONLs and embed into memory."""
    logger.info(f"INGEST_CONVOS dir={req.convo_dir} pattern={req.file_pattern}")

    convo_dir = os.path.expanduser(req.convo_dir)
    if not os.path.isdir(convo_dir):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {convo_dir}"})

    pattern = os.path.join(convo_dir, "**", req.file_pattern) if req.recursive else os.path.join(convo_dir, req.file_pattern)
    files = sorted(globmod.glob(pattern, recursive=req.recursive))
    if not files:
        return JSONResponse(status_code=400, content={"error": "No matching files found"})

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

            # Dedupe and embed
            new_chunks = []
            for chunk in chunks:
                h = hashlib.md5(chunk["text"].encode()).hexdigest()
                exists = db.execute("SELECT 1 FROM chunks WHERE content_hash = ?", (h,)).fetchone()
                if not exists:
                    new_chunks.append((chunk["text"], h, chunk["source"]))
                else:
                    total_dupes += 1

            if new_chunks:
                texts = [c[0] for c in new_chunks]
                embeddings = embed_batch(texts)

                for (text, h, source), emb in zip(new_chunks, embeddings):
                    db.execute(
                        "INSERT INTO chunks (content, content_hash, source, embedding) VALUES (?, ?, ?, ?)",
                        (text, h, source, pack_embedding(emb))
                    )
                db.commit()
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


@app.post("/ingest_text")
async def ingest_text(req: IngestTextRequest):
    """Store arbitrary text into memory."""
    logger.info(f"INGEST_TEXT source={req.source} len={len(req.text)}")

    chunks = chunk_text(req.text)
    stored = 0

    for i, chunk in enumerate(chunks):
        h = hashlib.md5(chunk.encode()).hexdigest()
        exists = db.execute("SELECT 1 FROM chunks WHERE content_hash = ?", (h,)).fetchone()
        if not exists:
            emb = embed_text(chunk)
            db.execute(
                "INSERT INTO chunks (content, content_hash, source, embedding) VALUES (?, ?, ?, ?)",
                (chunk, h, f"{req.source}:chunk{i}", pack_embedding(emb))
            )
            stored += 1

    db.commit()
    return {"chunks_stored": stored, "total_chunks": len(chunks)}


@app.post("/recall")
async def recall(req: RecallRequest):
    """Semantic search over conversation memory."""
    logger.info(f"RECALL query='{req.query[:80]}' top_k={req.top_k}")

    query_emb = embed_text(req.query)

    rows = db.execute("SELECT id, content, source, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()

    scored = []
    for row_id, content, source, emb_blob in rows:
        emb = unpack_embedding(emb_blob)
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, row_id, content, source))

    scored.sort(reverse=True)
    top = scored[:req.top_k]

    return {
        "results": [
            {"id": r[1], "content": r[2], "source": r[3], "similarity": round(r[0], 4)}
            for r in top
        ],
        "query": req.query,
        "searched": len(rows),
    }


@app.get("/memory_stats")
async def memory_stats():
    total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    with_emb = db.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    sources = db.execute("SELECT DISTINCT substr(source, 1, instr(source, ':') - 1) as src, COUNT(*) FROM chunks GROUP BY src").fetchall()

    return {
        "total_memories": total,
        "with_embeddings": with_emb,
        "sources": {s[0] if s[0] else "other": s[1] for s in sources},
        "cache_size": len(_cache),
        "uptime": round(time.time() - START_TIME),
        "embed_model": EMBED_MODEL,
        "db_path": DB_PATH,
    }


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    init_db()
    init_embedder()
    logger.info(f"CLOXY v2.0 ready on port {PORT}")


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
  v2.0
""")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
