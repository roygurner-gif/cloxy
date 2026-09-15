"""
CLOXY server — FastAPI app wiring the web proxy, memory, conversation
watcher, and the MLX chat endpoint together on one port.

Run with `cloxy start` (or `python -m cloxy start`).
"""
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__, config, memory, proxy
from .convos import ingester, watcher
from .memory import ChunkIn

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("cloxy")

START_TIME = time.time()


# =============================================================================
# AUTH
# =============================================================================

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def check_auth(api_key: Optional[str] = Depends(_api_key_header)):
    """Optional API key auth. Skipped if CLOXY_API_KEY is not set."""
    if config.API_KEY and not (api_key and secrets.compare_digest(api_key, config.API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =============================================================================
# LIFESPAN
# =============================================================================

async def _maybe_eager_load_llm():
    if not config.EAGER_LLM:
        return
    try:
        from .backends import mlx_backend
        if not config.CONFIG_PATH.exists():
            logger.info("CLOXY_EAGER_LLM set but no config — run `cloxy init` first.")
            return
        hf_id = json.loads(config.CONFIG_PATH.read_text()).get("model_hf_id")
        if hf_id:
            logger.info(f"Eagerly loading LLM: {hf_id}")
            await mlx_backend.load_model(hf_id)
            logger.info("LLM ready.")
    except Exception as e:
        logger.warning(f"Eager LLM load failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory.init_db()
    memory.init_embedder()
    await memory.load_vector_index()
    await _maybe_eager_load_llm()
    if config.WATCH:
        watcher.start()
    logger.info(f"CLOXY v{__version__} ready on {config.HOST}:{config.PORT}")
    yield
    await watcher.stop()
    await memory.close_db()
    logger.info("CLOXY shut down")


app = FastAPI(title="CLOXY", version=__version__, lifespan=lifespan)


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


class VerifyRequest(BaseModel):
    url: str
    claim: str
    top_k: int = Field(3, ge=1, le=50)


class IngestConvoRequest(BaseModel):
    convo_dir: Optional[str] = None      # default: the watched directories
    force: bool = False                  # re-ingest from scratch


class IngestTextRequest(BaseModel):
    text: str = Field(..., max_length=config.MAX_INGEST_CHARS)
    source: str = "manual"
    project: Optional[str] = None
    metadata: Optional[dict] = None


class RecallRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=100)
    mode: str = "hybrid"                 # hybrid | dense | keyword
    project: Optional[str] = None        # substring match on the session's cwd
    since: Optional[str] = None          # ISO date/time lower bound
    until: Optional[str] = None          # ISO date/time upper bound
    recency_weight: float = Field(0.3, ge=0.0, le=1.0)
    rerank: Optional[bool] = None        # default: CLOXY_RERANK


class DeleteBySourceRequest(BaseModel):
    source_prefix: str  # matches source LIKE 'prefix%' (e.g. "convo:" or "manual")


# =============================================================================
# INFO + HEALTH
# =============================================================================

BANNER = r"""
   _____ _     _____ __  ____   __
  / ____| |   / __ \ \/ /\ \ / /
 | |    | |  | |  | |\  /  \ V /
 | |    | |  | |  | |/  \   | |
 | |____| |__| |__| / /\ \  | |
  \_____|_____\____/_/  \_\ |_|
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    count = await memory.count_where("1=1", ())
    auth_status = "ENABLED" if config.API_KEY else "DISABLED"
    watch = "ON" if watcher.status["running"] else "OFF"
    return f"""<pre style="color:#0f0;background:#111;padding:20px;font-family:monospace">{BANNER}
  Give your local AI eyes and memory.
  v{__version__} — Port {config.PORT} — {count} memories — Auth {auth_status} — Watcher {watch}

  WEB PROXY:
    POST /fetch          — Fetch and clean a URL (clean | raw | markdown | extract)
    POST /search         — Fetch URL and extract lines around a pattern
    POST /verify         — Fetch URL and rank passages by semantic match to a claim

  MEMORY:
    POST /recall         — Hybrid search (dense + keyword), filters, recency
    POST /ingest_text    — Store any text
    POST /ingest_convos  — Ingest Claude Code sessions now (the watcher does this live)
    GET  /projects       — Projects seen in memory
    GET  /memory_stats   — Memory stats
    GET  /ingest_status  — Watcher status
    DELETE /memory/{{id}}  — Delete one memory
    POST /forget         — Delete memories by source prefix
    POST /reindex        — Rebuild the vector index from the DB

  LLM (OpenAI-compatible, MLX):
    POST /v1/chat/completions — Chat (streaming or not; "memory": true to inject recall)
    GET  /v1/models      — The currently-loaded model

  GET  /health
</pre>"""


@app.get("/health")
async def health():
    count = await memory.count_where("1=1", ())
    return {
        "status": "OK",
        "service": "cloxy",
        "version": __version__,
        "uptime": round(time.time() - START_TIME),
        "cache_size": proxy.cache.currsize,
        "memories": count,
        "vector_index_size": memory.vec_index.size,
        "embed_model": config.EMBED_MODEL,
        "auth": "enabled" if config.API_KEY else "disabled",
        "watcher": {k: watcher.status[k] for k in ("running", "dirs", "scans", "last_scan_at")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# WEB PROXY
# =============================================================================

@app.post("/fetch", dependencies=[Depends(check_auth)])
async def fetch(req: FetchRequest):
    logger.info(f"FETCH url={req.url} mode={req.mode}")

    if req.mode not in proxy.FETCH_MODES:
        return JSONResponse(status_code=400, content={"error": f"Unknown mode: {req.mode}"})
    if req.mode == "extract" and not req.selector:
        return JSONResponse(status_code=400, content={"error": "selector required for extract mode"})

    ckey = proxy.cache_key(req.url, req.mode, req.selector, req.headers)
    cached = proxy.cache.get(ckey)
    if cached:
        return {**cached, "from_cache": True}

    headers = proxy.default_headers()
    if req.headers:
        headers.update(req.headers)
    try:
        html, status, final_url = await proxy.safe_get(req.url, headers)
    except Exception as e:
        return _fetch_error(e, req.url)

    result = {
        "url": req.url,
        "final_url": final_url,
        "status": status,
        "mode": req.mode,
        "from_cache": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        **proxy.extract(html, req.mode, req.selector),
    }
    proxy.cache[ckey] = result
    return result


def _fetch_error(e: Exception, url: str) -> JSONResponse:
    if isinstance(e, proxy.FetchError):
        return JSONResponse(status_code=e.status, content={"error": e.message, "url": url})
    return JSONResponse(status_code=502, content={"error": str(e), "url": url})


@app.post("/search", dependencies=[Depends(check_auth)])
async def search_extract(req: SearchExtract):
    logger.info(f"SEARCH url={req.url} pattern={req.pattern}")
    try:
        html, _, _ = await proxy.safe_get(req.url)
    except Exception as e:
        return _fetch_error(e, req.url)

    lines = proxy.clean_text(html).split("\n")
    needle = req.pattern.lower()
    matches = []
    for i, line in enumerate(lines):
        if needle in line.lower():
            matches.append({"line": i, "context": "\n".join(lines[max(0, i - 2):i + 3])})
    return {"url": req.url, "pattern": req.pattern, "matches": matches,
            "total_matches": len(matches)}


@app.post("/verify", dependencies=[Depends(check_auth)])
async def verify(req: VerifyRequest):
    """
    Fetch a URL and return the top-K passages most semantically relevant to a
    claim, with cosine scores. The caller decides support/contradiction —
    Cloxy stays a tool, not a judge.
    """
    import numpy as np
    logger.info(f"VERIFY url={req.url} claim='{req.claim[:80]}'")

    ckey = proxy.cache_key(req.url, "clean")
    cached = proxy.cache.get(ckey)
    if cached and cached.get("content") is not None:
        content, final_url, from_cache = cached["content"], cached.get("final_url", req.url), True
    else:
        try:
            html, status, final_url = await proxy.safe_get(req.url)
        except Exception as e:
            return _fetch_error(e, req.url)
        content = proxy.clean_text(html, include_links=False)
        from_cache = False
        proxy.cache[ckey] = {
            "url": req.url, "final_url": final_url, "status": status, "mode": "clean",
            "from_cache": False, "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content": content, "length": len(content),
        }

    base = {"url": req.url, "final_url": final_url, "claim": req.claim,
            "from_cache": from_cache, "fetched_at": datetime.now(timezone.utc).isoformat()}
    chunks = memory.chunk_text(content) if content else []
    if not chunks:
        return {**base, "matches": [], "total_chunks_searched": 0,
                "note": "No content extracted from URL"}

    all_embs = await memory.aembed_batch(chunks + [req.claim])
    mat = np.vstack(all_embs[:-1]).astype(np.float32)
    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
    q = all_embs[-1].astype(np.float32).reshape(1, -1)
    q /= max(float(np.linalg.norm(q)), 1e-9)
    scores = (mat @ q.T).flatten()
    k = min(req.top_k, len(scores))
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return {
        **base,
        "matches": [{
            "passage": chunks[int(i)],
            "similarity": round(float(scores[int(i)]), 4),
            "chunk_index": int(i),
            "position_pct": round(100 * int(i) / max(1, len(chunks) - 1), 1),
        } for i in top],
        "total_chunks_searched": len(chunks),
        "best_similarity": round(float(scores.max()), 4),
    }


# =============================================================================
# MEMORY
# =============================================================================

@app.post("/ingest_convos", dependencies=[Depends(check_auth)])
async def ingest_convos(req: IngestConvoRequest):
    """Run an ingest pass now (the watcher does this continuously)."""
    dirs = [os.path.expanduser(req.convo_dir)] if req.convo_dir else None
    if dirs and not os.path.isdir(dirs[0]):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {dirs[0]}"})
    logger.info(f"INGEST_CONVOS dirs={dirs or config.WATCH_DIRS} force={req.force}")
    summary = await ingester.scan(dirs, force=req.force, settle=0)
    return summary.as_dict()


@app.post("/ingest_text", dependencies=[Depends(check_auth)])
async def ingest_text(req: IngestTextRequest):
    logger.info(f"INGEST_TEXT source={req.source} len={len(req.text)}")
    now = datetime.now(timezone.utc).isoformat()
    chunks = memory.chunk_text(req.text)
    res = await memory.store_chunks([
        ChunkIn(text=c, source=f"{req.source}:chunk{i}", project=req.project,
                ts_start=now, ts_end=now, metadata=req.metadata)
        for i, c in enumerate(chunks)
    ])
    return {"chunks_stored": res.stored, "total_chunks": len(chunks), "ids": res.ids}


@app.post("/recall", dependencies=[Depends(check_auth)])
async def recall(req: RecallRequest):
    logger.info(f"RECALL query='{req.query[:80]}' top_k={req.top_k} mode={req.mode} "
                f"project={req.project} since={req.since} until={req.until}")
    try:
        results = await memory.search(
            req.query, top_k=req.top_k, mode=req.mode, project=req.project,
            since=req.since, until=req.until, recency_weight=req.recency_weight,
            rerank=req.rerank)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return {"results": results, "query": req.query, "mode": req.mode,
            "searched": memory.vec_index.size}


@app.get("/projects", dependencies=[Depends(check_auth)])
async def projects():
    return {"projects": await memory.projects()}


@app.get("/memory_stats", dependencies=[Depends(check_auth)])
async def memory_stats():
    return {**(await memory.stats()), "cache_size": proxy.cache.currsize,
            "uptime": round(time.time() - START_TIME)}


@app.get("/ingest_status", dependencies=[Depends(check_auth)])
async def ingest_status():
    tracked = (await memory.db.execute_fetchall("SELECT COUNT(*) FROM ingest_state"))[0][0]
    return {**watcher.status, "files_tracked": tracked}


@app.delete("/memory/{chunk_id}", dependencies=[Depends(check_auth)])
async def delete_memory(chunk_id: int):
    if await memory.delete_ids([chunk_id]) == 0:
        return JSONResponse(status_code=404, content={"error": f"No memory with id {chunk_id}"})
    return {"deleted": chunk_id}


@app.post("/forget", dependencies=[Depends(check_auth)])
async def forget(req: DeleteBySourceRequest):
    logger.info(f"FORGET source_prefix={req.source_prefix!r}")
    ids = await memory.forget_prefix(req.source_prefix)
    return {"deleted": len(ids), "source_prefix": req.source_prefix}


@app.post("/reindex", dependencies=[Depends(check_auth)])
async def reindex():
    await memory.load_vector_index()
    await memory.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    await memory.db.commit()
    return {"status": "rebuilt", "vector_index_size": memory.vec_index.size}


# =============================================================================
# LLM CHAT (OpenAI-compatible) — Apple Silicon via MLX
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    # OpenAI clients send either a string or a list of content parts.
    content: Union[str, List[dict], None] = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if not self.content:
            return ""
        return "\n".join(p.get("text", "") for p in self.content
                         if isinstance(p, dict) and p.get("type") == "text")


DEFAULT_MAX_TOKENS = 512


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None              # ignored; we serve the loaded model
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None   # current OpenAI field name
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    # Cloxy extensions: prepend relevant memory to the prompt.
    memory: Optional[bool] = None            # default: CLOXY_CHAT_MEMORY
    memory_top_k: int = Field(config.CHAT_MEMORY_TOP_K, ge=1, le=20)
    memory_project: Optional[str] = None

    def effective_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS


async def _ensure_llm_loaded():
    from .backends import mlx_backend
    if mlx_backend.is_loaded():
        return
    if not config.CONFIG_PATH.exists():
        raise HTTPException(status_code=503,
                            detail="No LLM configured. Run `cloxy init` to pick a model.")
    hf_id = json.loads(config.CONFIG_PATH.read_text()).get("model_hf_id")
    if not hf_id:
        raise HTTPException(status_code=503, detail="Config has no model_hf_id.")
    await mlx_backend.load_model(hf_id)


def _memory_block(hits: List[dict]) -> str:
    lines = ["Relevant memory from past conversations (most relevant first). "
             "Use it when it helps; ignore it when it doesn't."]
    for i, h in enumerate(hits, 1):
        tag = " · ".join(b for b in (
            (h.get("ts_start") or "")[:10],
            os.path.basename((h.get("project") or "").rstrip("/")),
        ) if b)
        body = h["content"] if len(h["content"]) <= 1200 else h["content"][:1200] + " …"
        lines.append(f"{i}. [{tag}] {body}" if tag else f"{i}. {body}")
    return "\n\n".join(lines)


async def _with_memory(req: ChatCompletionRequest, messages: List[dict]) -> List[dict]:
    """Prepend a system block of recalled memory keyed on the last user message."""
    use = config.CHAT_MEMORY if req.memory is None else req.memory
    if not use:
        return messages
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not last_user.strip():
        return messages
    hits = await memory.search(last_user, top_k=req.memory_top_k, project=req.memory_project)
    if not hits:
        return messages
    block = _memory_block(hits)
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + "\n\n" + block}] + messages[1:]
    return [{"role": "system", "content": block}] + messages


@app.get("/v1/models", dependencies=[Depends(check_auth)])
async def list_models():
    from .backends import mlx_backend
    current = mlx_backend.current_model()
    if not current:
        return {"object": "list", "data": []}
    return {"object": "list", "data": [{
        "id": current, "object": "model", "created": int(START_TIME), "owned_by": "cloxy-mlx",
    }]}


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint, MLX-backed."""
    await _ensure_llm_loaded()
    from .backends import mlx_backend

    role_map = {"developer": "system"}
    messages = [{"role": role_map.get(m.role, m.role), "content": m.text()} for m in req.messages]
    messages = await _with_memory(req, messages)
    max_tokens = req.effective_max_tokens()

    if not req.stream:
        return await mlx_backend.generate_chat(
            messages=messages, max_tokens=max_tokens,
            temperature=req.temperature, top_p=req.top_p)

    async def event_stream():
        completion_id = f"cloxy-{int(time.time() * 1000)}"
        model_id = mlx_backend.current_model() or "unknown"
        finish_reason, usage = "stop", None

        # If the client disconnects, Starlette closes this generator, which
        # closes stream_chat, which signals the MLX producer thread to stop.
        async for piece in mlx_backend.stream_chat(
            messages=messages, max_tokens=max_tokens,
            temperature=req.temperature, top_p=req.top_p,
        ):
            if piece.finish_reason:
                finish_reason, usage = piece.finish_reason, piece.usage
            if not piece.text:
                continue
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model_id,
                "choices": [{"index": 0, "delta": {"content": piece.text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        if usage:
            final["usage"] = usage
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# =============================================================================
# ENTRYPOINT
# =============================================================================

def serve():
    """Start uvicorn. Used by `cloxy start`."""
    import sys
    import uvicorn
    problem = memory.preflight_embed_check()
    if problem:
        sys.exit(f"\nCLOXY refusing to start: {problem}\n")
    print(BANNER)
    print(f"  Local AI with eyes and memory — v{__version__}\n")
    if config.HOST == "0.0.0.0" and not config.API_KEY:
        logger.warning("Binding 0.0.0.0 with NO api key — the proxy + RAG are "
                       "open to your whole network. Set CLOXY_API_KEY.")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
