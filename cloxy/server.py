"""
CLOXY server — FastAPI app wiring the web proxy, memory, conversation
watcher, and the MLX chat endpoint together on one port.

Run with `cloxy start` (or `python -m cloxy start`).
"""
import json
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__, backends, config, memory, proxy
from . import push as pushmod
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
        if config.LLM_URL:
            await backends.get().load_model(_configured_model())
            logger.info(f"LLM upstream ready: {config.LLM_URL}")
            return
        if not config.CONFIG_PATH.exists():
            logger.info("CLOXY_EAGER_LLM set but no config — run `cloxy init` first.")
            return
        hf_id = _configured_model()
        if hf_id:
            logger.info(f"Eagerly loading LLM: {hf_id}")
            await backends.get().load_model(hf_id)
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


class PushRequest(BaseModel):
    host: str = Field(..., max_length=64)
    agent: str = "claude"                # claude | codex
    path: str = Field(..., max_length=1024)   # relative to the agent's session root
    offset: int = Field(..., ge=0)       # byte offset the data starts at
    data: str = Field("", max_length=config.PUSH_MAX_BYTES * 2)   # base64, whole lines
    replace: bool = False                # rewrite the mirror from offset 0


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
    POST /ingest_convos  — Ingest agent sessions now (the watcher does this live)
    POST /push           — Receive session lines from `cloxy push` on another machine
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


@app.post("/push", dependencies=[Depends(check_auth)])
async def push(req: PushRequest):
    """Append session lines mirrored from another machine (see `cloxy push`)."""
    try:
        dest, size, rewritten = pushmod.apply_push(
            req.host, req.agent, req.path, req.offset, req.data, req.replace)
    except pushmod.PushError as e:
        body = {"error": e.message}
        if e.size is not None:
            body["size"] = e.size
        return JSONResponse(status_code=e.status, content=body)
    if rewritten:
        # The mirror no longer matches what was ingested from it: start that file over.
        await ingester.ingest_file(str(dest), force=True)
    return {"size": size}


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
    # Cite recalled memory inline as [n]; response carries `cloxy_citations`.
    citations: Optional[bool] = None         # default: CLOXY_CHAT_CITATIONS

    def effective_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS


def _configured_model() -> Optional[str]:
    try:
        return json.loads(config.CONFIG_PATH.read_text()).get("model_hf_id")
    except (OSError, json.JSONDecodeError):
        return None


async def _ensure_llm_loaded():
    backend = backends.get()
    if backend.is_loaded():
        return
    if config.LLM_URL:
        from .backends.openai_backend import UpstreamError
        try:
            await backend.load_model(_configured_model())
        except UpstreamError as e:
            raise HTTPException(status_code=e.status, detail=str(e))
        return
    if not config.CONFIG_PATH.exists():
        raise HTTPException(status_code=503,
                            detail="No LLM configured. Run `cloxy init` to pick a model.")
    hf_id = json.loads(config.CONFIG_PATH.read_text()).get("model_hf_id")
    if not hf_id:
        raise HTTPException(status_code=503, detail="Config has no model_hf_id.")
    await backend.load_model(hf_id)


CITE_INSTRUCTIONS = (
    "Numbered sources from memory (most relevant first). When a statement relies on "
    "a source, cite it inline right after the statement as [n], e.g. [2] or [1][3]. "
    "Only cite numbers listed below. If the sources don't support an answer, say so "
    "plainly instead of guessing, and don't cite anything for general knowledge.")


def _hit_tag(h: dict, with_source: bool = False) -> str:
    parts = [(h.get("ts_start") or "")[:10],
             os.path.basename((h.get("project") or "").rstrip("/"))]
    if with_source:
        parts.append(h.get("source") or "")
    return " · ".join(b for b in parts if b)


def _memory_block(hits: List[dict], cite: bool = False) -> str:
    head = CITE_INSTRUCTIONS if cite else (
        "Relevant memory from past conversations (most relevant first). "
        "Use it when it helps; ignore it when it doesn't.")
    lines = [head]
    for i, h in enumerate(hits, 1):
        tag = _hit_tag(h, with_source=cite)
        body = h["content"] if len(h["content"]) <= 1200 else h["content"][:1200] + " …"
        lines.append(f"{i}. [{tag}] {body}" if tag else f"{i}. {body}")
    return "\n\n".join(lines)


_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def _citations(hits: List[dict], answer: str) -> dict:
    """Map each [n] in the answer back to the memory it points at.

    `sources` lists every injected memory with `cited` set when the answer used
    it; `unknown` lists numbers the model cited that don't exist (a red flag
    worth surfacing to the reader).
    """
    used = {int(n) for n in _CITE_RE.findall(answer or "")}
    sources = []
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h["content"].split())
        sources.append({
            "n": i,
            "id": h.get("id"),
            "source": h.get("source"),
            "project": h.get("project"),
            "session_id": h.get("session_id"),
            "date": (h.get("ts_start") or "")[:10] or None,
            "snippet": snippet if len(snippet) <= 280 else snippet[:280] + " …",
            "cited": i in used,
        })
    return {"sources": sources,
            "unknown": sorted(n for n in used if not 1 <= n <= len(hits))}


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _cited_claims(answer: str) -> List[Tuple[str, List[int]]]:
    """Split the answer into sentences and keep the ones carrying [n] markers."""
    out = []
    for sent in _SENTENCE_RE.split(answer or ""):
        nums = [int(n) for n in _CITE_RE.findall(sent)]
        text = " ".join(_CITE_RE.sub("", sent).split())
        if nums and text:
            out.append((text, sorted(set(nums))))
    return out


async def _check_citations(hits: List[dict], answer: str) -> List[dict]:
    """Does each cited sentence actually match the source it cites?

    Embeds every cited sentence and small windows of every injected source, then
    asks which source matches the sentence best. A citation is flagged when the
    best-matching source isn't one of the cited ones (beyond a small margin), or
    when it points at a number that doesn't exist. Heuristic, not proof: it
    catches mis-attributed citations, not every wrong claim.
    """
    import numpy as np
    claims = _cited_claims(answer)
    if not claims or not hits:
        return []
    windows, owner = [], []
    for i, h in enumerate(hits, 1):
        for w in memory.chunk_text(h["content"], size=config.CITE_WINDOW,
                                   overlap=config.CITE_WINDOW // 5) or [h["content"]]:
            windows.append(config.EMBED_PASSAGE_PREFIX + w)
            owner.append(i)
    embs = await memory.aembed_batch(
        windows + [config.EMBED_QUERY_PREFIX + c for c, _ in claims])
    mat = np.vstack(embs).astype(np.float32)
    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
    win, qs = mat[:len(windows)], mat[len(windows):]
    sims = qs @ win.T                                   # (claims, windows)
    owner_arr = np.array(owner)

    report = []
    for (text, cited), row in zip(claims, sims):
        per_src = {n: round(float(row[owner_arr == n].max()), 4) for n in range(1, len(hits) + 1)}
        best_n = max(per_src, key=per_src.get)
        valid = [n for n in cited if n in per_src]
        cited_best = max((per_src[n] for n in valid), default=None)
        if len(valid) < len(cited):
            verdict = "unknown_source"
        elif cited_best is not None and per_src[best_n] - cited_best > config.CITE_MARGIN:
            verdict = "better_match_elsewhere"
        else:
            verdict = "ok"
        report.append({
            "claim": text if len(text) <= 300 else text[:300] + " …",
            "cited": cited,
            "support": {n: per_src[n] for n in valid},
            "best_match": best_n,
            "verdict": verdict,
        })
    return report


async def _citation_payload(hits: List[dict], answer: str) -> dict:
    out = _citations(hits, answer)
    try:
        out["checks"] = await _check_citations(hits, answer)
    except Exception as e:                      # never fail the chat over the check
        logger.warning(f"citation check failed: {e}")
        out["checks"], out["check_error"] = [], str(e)
    out["flagged"] = sum(c["verdict"] != "ok" for c in out["checks"])
    return out


async def _with_memory(req: ChatCompletionRequest,
                       messages: List[dict]) -> Tuple[List[dict], List[dict], bool]:
    """Prepend a system block of recalled memory keyed on the last user message.

    Returns (messages, hits, cite). `cite` is True only when citations were asked
    for and there was memory to cite.
    """
    use = config.CHAT_MEMORY if req.memory is None else req.memory
    cite = config.CHAT_CITATIONS if req.citations is None else req.citations
    if req.citations and req.memory is None:
        use = True                          # asking for citations implies memory
    if not use:
        return messages, [], False
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not last_user.strip():
        return messages, [], False
    hits = await memory.search(last_user, top_k=req.memory_top_k, project=req.memory_project)
    if not hits:
        return messages, [], False
    block = _memory_block(hits, cite=cite)
    if messages and messages[0]["role"] == "system":
        merged = [{"role": "system", "content": messages[0]["content"] + "\n\n" + block}] + messages[1:]
    else:
        merged = [{"role": "system", "content": block}] + messages
    return merged, hits, cite


@app.get("/v1/models", dependencies=[Depends(check_auth)])
async def list_models():
    current = backends.get().current_model()
    if not current:
        return {"object": "list", "data": []}
    owner = "cloxy-upstream" if config.LLM_URL else "cloxy-mlx"
    return {"object": "list", "data": [{
        "id": current, "object": "model", "created": int(START_TIME), "owned_by": owner,
    }]}


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions: in-process MLX, or the CLOXY_LLM_URL upstream."""
    await _ensure_llm_loaded()
    backend = backends.get()
    from .backends.openai_backend import UpstreamError

    role_map = {"developer": "system"}
    messages = [{"role": role_map.get(m.role, m.role), "content": m.text()} for m in req.messages]
    messages, hits, cite = await _with_memory(req, messages)
    max_tokens = req.effective_max_tokens()

    if not req.stream:
        try:
            out = await backend.generate_chat(
                messages=messages, max_tokens=max_tokens,
                temperature=req.temperature, top_p=req.top_p)
        except UpstreamError as e:
            raise HTTPException(status_code=e.status, detail=str(e))
        if cite:
            choices = out.get("choices") or [{}]
            answer = (choices[0].get("message") or {}).get("content") or ""
            out["cloxy_citations"] = await _citation_payload(hits, answer)
        return out

    # Pull the first piece before answering, so an unreachable upstream is an
    # HTTP error and not a 200 that dies mid-stream.
    pieces = backend.stream_chat(
        messages=messages, max_tokens=max_tokens,
        temperature=req.temperature, top_p=req.top_p,
    )
    try:
        first = await pieces.__anext__()
    except StopAsyncIteration:
        first = None
    except UpstreamError as e:
        raise HTTPException(status_code=e.status, detail=str(e))

    async def _all():
        try:
            if first is not None:
                yield first
                async for p in pieces:
                    yield p
        finally:
            await pieces.aclose()   # stops the MLX producer thread on disconnect

    async def event_stream():
        completion_id = f"cloxy-{int(time.time() * 1000)}"
        model_id = backend.current_model() or "unknown"
        finish_reason, usage = "stop", None
        answer = []

        # If the client disconnects, Starlette closes this generator, which
        # closes stream_chat, which signals the MLX producer thread to stop.
        async for piece in _all():
            if piece.finish_reason:
                finish_reason, usage = piece.finish_reason, piece.usage
            if not piece.text:
                continue
            answer.append(piece.text)
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
        if cite:
            # Citations need the whole answer, so they ride on the last chunk.
            final["cloxy_citations"] = await _citation_payload(hits, "".join(answer))
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
