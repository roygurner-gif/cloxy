"""
Memory: SQLite store + in-memory vector index + FTS5 keyword index, with
hybrid (dense + keyword) search, metadata filters, and recency weighting.

Owned by the server process. Everything else (CLI, MCP server) talks to it
over HTTP so there is exactly one embedder and one index in memory.
"""
import asyncio
import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import aiosqlite
import numpy as np
from fastembed import TextEmbedding

from . import config

logger = logging.getLogger("cloxy.memory")

# --- Process-wide state (one embedder, one connection, one index) ---
embedder: Optional[TextEmbedding] = None
reranker = None
db: Optional[aiosqlite.Connection] = None


# =============================================================================
# VECTOR INDEX — In-memory numpy matrix for fast cosine similarity
# =============================================================================

class VectorIndex:
    """
    In-memory vector index backed by a normalized numpy matrix.
    Recall is a single matrix multiply — no Python loops, no full table scan.

    Vectors are appended to a Python list and the packed matrix is rebuilt
    lazily on the next search (dirty flag), so inserts stay O(1) amortized.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ids: List[int] = []
        self._vecs: List[np.ndarray] = []          # normalized (dim,) rows, append-only
        self._matrix: Optional[np.ndarray] = None  # (N, dim) rebuilt from _vecs
        self._ids_arr: Optional[np.ndarray] = None
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
        with self._lock:
            self._ids.append(chunk_id)
            self._vecs.append(self._normalize(embedding))
            self._dirty = True

    def add_batch(self, ids: List[int], embeddings: List[np.ndarray]):
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
        if not self._vecs:
            self._matrix = None
            self._ids_arr = None
        else:
            self._matrix = np.vstack(self._vecs).astype(np.float32)
            self._ids_arr = np.asarray(self._ids, dtype=np.int64)
        self._dirty = False

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               allowed_ids: Optional[Iterable[int]] = None) -> List[tuple]:
        """
        Returns [(chunk_id, cosine_similarity)] sorted by relevance.
        `allowed_ids` restricts the search to a subset (metadata filters).
        """
        with self._lock:
            if self._dirty or self._matrix is None:
                self._rebuild_locked()
            if self._matrix is None or len(self._ids) == 0:
                return []
            q = self._normalize(query_embedding).reshape(1, -1)
            scores = (self._matrix @ q.T).flatten()
            if allowed_ids is not None:
                mask = np.isin(self._ids_arr, np.fromiter(allowed_ids, dtype=np.int64))
                if not mask.any():
                    return []
                scores = np.where(mask, scores, -np.inf)
            k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [(self._ids[i], float(scores[i])) for i in top_indices
                    if np.isfinite(scores[i])]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)


vec_index = VectorIndex()


# =============================================================================
# EMBEDDING
# =============================================================================

def init_embedder():
    global embedder
    logger.info(f"Loading embedding model: {config.EMBED_MODEL}")
    embedder = TextEmbedding(model_name=config.EMBED_MODEL)
    logger.info("Embedding model loaded")


def embed_text(text: str) -> np.ndarray:
    return list(embedder.embed([text]))[0]


def embed_batch(texts: List[str]) -> List[np.ndarray]:
    return list(embedder.embed(texts))


async def aembed_text(text: str) -> np.ndarray:
    """Embed off the event loop (fastembed is sync + CPU-bound)."""
    return await asyncio.to_thread(embed_text, text)


async def aembed_batch(texts: List[str]) -> List[np.ndarray]:
    return await asyncio.to_thread(embed_batch, texts)


def pack_embedding(vec: np.ndarray) -> bytes:
    # Native-endian float32, byte-identical to the pre-4.1 struct.pack("Nf") layout.
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _get_reranker():
    """Lazy-load the cross-encoder the first time a rerank is requested."""
    global reranker
    if reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        logger.info(f"Loading reranker: {config.RERANK_MODEL}")
        reranker = TextCrossEncoder(model_name=config.RERANK_MODEL)
    return reranker


# =============================================================================
# TEXT CHUNKING + HASHING
# =============================================================================

def chunk_text(text: str, size: int = None, overlap: int = None) -> List[str]:
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
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


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# =============================================================================
# DATABASE
# =============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE NOT NULL,
    source TEXT DEFAULT 'unknown',
    embedding BLOB,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    project TEXT,
    session_id TEXT,
    ts_start TEXT,
    ts_end TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_source ON chunks(source);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS ingest_state (
    path TEXT PRIMARY KEY,
    session_id TEXT,
    project TEXT,
    title TEXT,
    git_branch TEXT,
    committed_offset INTEGER DEFAULT 0,
    next_index INTEGER DEFAULT 0,
    tail_chunk_id INTEGER,
    tail_hash TEXT,
    size INTEGER,
    mtime REAL,
    chunks INTEGER DEFAULT 0,
    legacy_cleared INTEGER DEFAULT 0,
    updated_at TEXT
);
"""

# v5 columns added to a pre-v5 `chunks` table (name -> DDL type).
V5_COLUMNS = {
    "project": "TEXT", "session_id": "TEXT", "ts_start": "TEXT",
    "ts_end": "TEXT", "metadata": "TEXT",
}

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, content='chunks', content_rowid='id',
    tokenize="unicode61 tokenchars '_'"
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


async def init_db():
    """Open the database, create/migrate the schema, and enforce embed metadata."""
    global db
    import os
    os.makedirs(config.DATA_DIR, exist_ok=True)
    db = await aiosqlite.connect(config.DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.executescript(SCHEMA)
    await _migrate()
    await db.commit()
    await _check_embed_meta()
    logger.info(f"Database ready at {config.DB_PATH}")


async def _migrate():
    """Bring a v3/v4 database up to the v5 schema in place."""
    cols = {r[1] for r in await db.execute_fetchall("PRAGMA table_info(chunks)")}
    for name, ddl in V5_COLUMNS.items():
        if name not in cols:
            await db.execute(f"ALTER TABLE chunks ADD COLUMN {name} {ddl}")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_project ON chunks(project)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_ts_start ON chunks(ts_start)")

    had_fts = await db.execute_fetchall(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
    await db.executescript(FTS_SCHEMA)
    if not had_fts:
        # First time: index whatever is already in `chunks`.
        await db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        logger.info("Built FTS5 keyword index from existing memories")


async def _check_embed_meta():
    """
    Record the embedding model + dim in the DB the first time, and refuse to
    start if a later run points at a different model/dim than the stored data.
    """
    rows = {r[0]: r[1] for r in await db.execute_fetchall("SELECT key, value FROM meta")}
    stored_model = rows.get("embed_model")
    stored_dim = rows.get("embed_dim")
    has_data = (await db.execute_fetchall("SELECT 1 FROM chunks LIMIT 1")) != []

    if stored_model is None:
        if has_data:
            logger.warning("No embed metadata in DB; assuming current model. "
                           "If recall looks wrong, re-ingest with a clean DB.")
        await db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                         ("embed_model", config.EMBED_MODEL))
        await db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                         ("embed_dim", str(config.EMBED_DIM)))
        await db.commit()
        return

    if stored_model != config.EMBED_MODEL or stored_dim != str(config.EMBED_DIM):
        raise RuntimeError(
            f"Embedding mismatch: DB was built with {stored_model} (dim {stored_dim}) "
            f"but CLOXY_EMBED_MODEL={config.EMBED_MODEL} (dim {config.EMBED_DIM}). "
            f"Use the original model, or start with a fresh CLOXY_DATA_DIR."
        )


def preflight_embed_check() -> Optional[str]:
    """
    Synchronous embed-model/dim check before uvicorn starts. Returns an error
    message on mismatch so the CLI can exit cleanly instead of failing inside
    the async lifespan.
    """
    import os
    import sqlite3
    if not os.path.exists(config.DB_PATH):
        return None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        rows = {k: v for k, v in conn.execute("SELECT key, value FROM meta").fetchall()}
        conn.close()
    except sqlite3.Error:
        return None  # pre-v4 DB without a meta table; the lifespan path handles it
    stored_model, stored_dim = rows.get("embed_model"), rows.get("embed_dim")
    if stored_model and (stored_model != config.EMBED_MODEL or stored_dim != str(config.EMBED_DIM)):
        return (f"embedding mismatch: DB was built with {stored_model} (dim {stored_dim}) "
                f"but you configured {config.EMBED_MODEL} (dim {config.EMBED_DIM}). "
                f"Use the original model, or start with a fresh CLOXY_DATA_DIR.")
    return None


async def load_vector_index():
    rows = await db.execute_fetchall(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")
    vec_index.load([r[0] for r in rows], [unpack_embedding(r[1]) for r in rows])
    logger.info(f"Vector index: loaded {len(rows)} embeddings into memory")


async def close_db():
    global db
    if db:
        await db.close()
        db = None


# =============================================================================
# WRITE PATH
# =============================================================================

@dataclass
class ChunkIn:
    text: str
    source: str
    project: Optional[str] = None
    session_id: Optional[str] = None
    ts_start: Optional[str] = None
    ts_end: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class StoreResult:
    ids: List[int] = field(default_factory=list)
    duplicates: int = 0

    @property
    def stored(self) -> int:
        return len(self.ids)


async def existing_hashes(conn, hashes: List[str]) -> set:
    """Subset of `hashes` already in the chunks table (batched IN queries)."""
    found = set()
    BATCH = 500
    for i in range(0, len(hashes), BATCH):
        window = hashes[i:i + BATCH]
        placeholders = ",".join("?" * len(window))
        rows = await conn.execute_fetchall(
            f"SELECT content_hash FROM chunks WHERE content_hash IN ({placeholders})", window)
        found.update(r[0] for r in rows)
    return found


async def store_chunks(items: List[ChunkIn]) -> StoreResult:
    """
    Dedupe (within the batch and against the DB), embed only what's new, insert
    with ON CONFLICT DO NOTHING as the race-safe backstop, and add to the index.
    """
    import json
    result = StoreResult()
    seen = set()
    candidates = []
    for item in items:
        h = content_hash(item.text)
        if h in seen:
            result.duplicates += 1
            continue
        seen.add(h)
        candidates.append((item, h))

    already = await existing_hashes(db, [h for _, h in candidates])
    new = [(item, h) for item, h in candidates if h not in already]
    result.duplicates += len(candidates) - len(new)
    if not new:
        return result

    embeddings = await aembed_batch([item.text for item, _ in new])
    new_embs = []
    for (item, h), emb in zip(new, embeddings):
        cursor = await db.execute(
            "INSERT INTO chunks (content, content_hash, source, embedding, project, "
            "session_id, ts_start, ts_end, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(content_hash) DO NOTHING RETURNING id",
            (item.text, h, item.source, pack_embedding(emb), item.project,
             item.session_id, item.ts_start, item.ts_end,
             json.dumps(item.metadata) if item.metadata else None),
        )
        row = await cursor.fetchone()
        if row is None:
            result.duplicates += 1
        else:
            result.ids.append(row[0])
            new_embs.append(emb)
    await db.commit()
    if result.ids:
        vec_index.add_batch(result.ids, new_embs)
    return result


async def delete_ids(ids: List[int]) -> int:
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    cursor = await db.execute(
        f"DELETE FROM chunks WHERE id IN ({placeholders}) RETURNING id", ids)
    gone = [r[0] for r in await cursor.fetchall()]
    await db.commit()
    vec_index.remove(gone)
    return len(gone)


async def delete_where(where: str, params: tuple) -> List[int]:
    cursor = await db.execute(f"DELETE FROM chunks WHERE {where} RETURNING id", params)
    ids = [r[0] for r in await cursor.fetchall()]
    await db.commit()
    if ids:
        vec_index.remove(ids)
    return ids


async def forget_prefix(prefix: str) -> List[int]:
    like = prefix.replace("%", r"\%").replace("_", r"\_") + "%"
    return await delete_where("source LIKE ? ESCAPE '\\'", (like,))


# =============================================================================
# READ PATH — hybrid search
# =============================================================================

RRF_K = 60
RECENCY_HALF_LIFE_DAYS = 30.0


def fts_query(text: str) -> str:
    """
    Turn free text into a lenient FTS5 MATCH expression: every token quoted
    and OR'd, so punctuation can't break the parser and partial overlap still
    ranks (bm25 rewards more matched terms).
    """
    tokens = re.findall(r"[\w][\w\-.:/]*", text)
    tokens = [t.strip("-.:/") for t in tokens]
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens[:32])


def _recency(ts: Optional[str], now: datetime) -> float:
    """1.0 for right now, 0.5 at the half-life, → 0 for ancient. 0.5 if unknown."""
    if not ts:
        return 0.5
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    age_days = max(0.0, (now - t).total_seconds() / 86400)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _filter_sql(project: Optional[str], since: Optional[str], until: Optional[str],
                alias: str = "") -> tuple:
    p = f"{alias}." if alias else ""
    clauses, params = [], []
    if project:
        clauses.append(f"{p}project LIKE ?")
        params.append(f"%{project}%")
    if since:
        clauses.append(f"COALESCE({p}ts_end, {p}created_at) >= ?")
        params.append(since)
    if until:
        clauses.append(f"COALESCE({p}ts_start, {p}created_at) <= ?")
        params.append(until)
    return (" AND ".join(clauses), params)


async def search(query: str, top_k: int = 5, mode: str = "hybrid",
                 project: Optional[str] = None, since: Optional[str] = None,
                 until: Optional[str] = None, recency_weight: float = 0.3,
                 rerank: Optional[bool] = None) -> List[dict]:
    """
    Hybrid recall: dense cosine (vector index) + BM25 (FTS5), fused with
    reciprocal rank fusion, filtered by project/date, tilted toward recent
    memories, optionally reranked by a cross-encoder.
    """
    import json
    if mode not in ("hybrid", "dense", "keyword"):
        raise ValueError(f"Unknown mode: {mode}")

    pool = max(top_k * 5, 50)
    where, params = _filter_sql(project, since, until)
    allowed = None
    if where:
        rows = await db.execute_fetchall(f"SELECT id FROM chunks WHERE {where}", params)
        allowed = {r[0] for r in rows}
        if not allowed:
            return []

    dense: List[tuple] = []
    if mode in ("hybrid", "dense") and vec_index.size:
        qemb = await aembed_text(query)
        dense = vec_index.search(qemb, top_k=pool, allowed_ids=allowed)

    keyword: List[tuple] = []
    if mode in ("hybrid", "keyword"):
        match = fts_query(query)
        if match:
            fwhere, fparams = _filter_sql(project, since, until, alias="c")
            sql = ("SELECT f.rowid, bm25(chunks_fts) AS r FROM chunks_fts f "
                   "JOIN chunks c ON c.id = f.rowid WHERE chunks_fts MATCH ?")
            if fwhere:
                sql += " AND " + fwhere
            sql += " ORDER BY r LIMIT ?"
            rows = await db.execute_fetchall(sql, [match, *fparams, pool])
            keyword = [(r[0], float(r[1])) for r in rows]

    # Reciprocal rank fusion
    fused: dict = {}
    dense_sim = {cid: s for cid, s in dense}
    kw_rank = {}
    for rank, (cid, _) in enumerate(dense):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (cid, _) in enumerate(keyword):
        kw_rank[cid] = rank + 1
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    if not fused:
        return []

    cand_ids = sorted(fused, key=fused.get, reverse=True)[:max(top_k * 4, 20)]
    placeholders = ",".join("?" * len(cand_ids))
    rows = await db.execute_fetchall(
        f"SELECT id, content, source, project, session_id, ts_start, ts_end, "
        f"metadata, created_at FROM chunks WHERE id IN ({placeholders})", cand_ids)
    by_id = {r[0]: r for r in rows}

    now = datetime.now(timezone.utc)
    results = []
    for cid in cand_ids:
        r = by_id.get(cid)
        if not r:
            continue
        rec = _recency(r["ts_end"] or r["created_at"], now)
        w = max(0.0, min(1.0, recency_weight))
        score = fused[cid] * ((1 - w) + w * rec)
        results.append({
            "id": cid,
            "content": r["content"],
            "source": r["source"],
            "project": r["project"],
            "session_id": r["session_id"],
            "ts_start": r["ts_start"],
            "ts_end": r["ts_end"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
            "score": round(score, 6),
            "similarity": round(dense_sim[cid], 4) if cid in dense_sim else None,
            "keyword_rank": kw_rank.get(cid),
            "recency": round(rec, 3),
        })
    results.sort(key=lambda x: x["score"], reverse=True)

    use_rerank = config.RERANK if rerank is None else rerank
    if use_rerank and results:
        head = results[:20]
        scores = await asyncio.to_thread(
            lambda: list(_get_reranker().rerank(query, [h["content"] for h in head])))
        for h, s in zip(head, scores):
            h["rerank_score"] = round(float(s), 4)
        head.sort(key=lambda x: x["rerank_score"], reverse=True)
        results = head + results[20:]

    return results[:top_k]


# =============================================================================
# STATS
# =============================================================================

async def stats() -> dict:
    total = (await db.execute_fetchall("SELECT COUNT(*) FROM chunks"))[0][0]
    with_emb = (await db.execute_fetchall(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"))[0][0]
    sources = await db.execute_fetchall(
        "SELECT CASE WHEN instr(source, ':') > 0 THEN substr(source, 1, instr(source, ':') - 1) "
        "ELSE source END AS src, COUNT(*) FROM chunks GROUP BY src")
    fts = (await db.execute_fetchall("SELECT COUNT(*) FROM chunks_fts"))[0][0]
    return {
        "total_memories": total,
        "with_embeddings": with_emb,
        "vector_index_size": vec_index.size,
        "keyword_index_size": fts,
        "sources": {s[0] or "other": s[1] for s in sources},
        "embed_model": config.EMBED_MODEL,
        "db_path": config.DB_PATH,
        "vector_engine": "numpy_matrix",
        "keyword_engine": "sqlite_fts5",
    }


async def projects() -> List[dict]:
    rows = await db.execute_fetchall(
        "SELECT project, COUNT(*) AS chunks, COUNT(DISTINCT session_id) AS sessions, "
        "MIN(ts_start) AS first_seen, MAX(ts_end) AS last_seen "
        "FROM chunks WHERE project IS NOT NULL GROUP BY project ORDER BY last_seen DESC")
    return [dict(r) for r in rows]


async def count_where(where: str, params: tuple) -> int:
    return (await db.execute_fetchall(f"SELECT COUNT(*) FROM chunks WHERE {where}", params))[0][0]
