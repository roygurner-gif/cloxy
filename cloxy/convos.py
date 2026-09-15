"""
Claude Code conversation ingest — incremental, metadata-aware, self-maintaining.

Session files (`~/.claude/projects/<encoded-cwd>/<session>.jsonl`) are
append-only. We track a byte offset per file and only re-read what's new:

  - Messages are packed whole into ~CHUNK_SIZE groups. Every group but the
    last is *final* and committed; the last ("tail") is stored too so it is
    searchable immediately, and replaced on the next pass when the file grows.
  - Each chunk carries project (the session's cwd), session id, first/last
    timestamps, git branch, and the session's AI title — so recall can filter
    by project/date and results come back dated.
  - A Watcher polls the session directories every few seconds. Polling a few
    thousand stat() calls is cheaper than an fsevents dependency and can't
    miss events.
"""
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from . import config, memory
from .memory import ChunkIn

logger = logging.getLogger("cloxy.convos")

# Harness plumbing that shows up inside message text and adds nothing to memory.
_STRIP_BLOCKS = re.compile(
    r"<(system-reminder|local-command-caveat|local-command-stdout|command-message|command-args)>"
    r".*?</\1>", re.DOTALL)
_STRIP_TAGS = re.compile(r"</?command-name>")


@dataclass
class Msg:
    role: str
    text: str
    ts: Optional[str]
    end_offset: int          # byte offset just past this message's line


@dataclass
class SessionMeta:
    session_id: str
    project: Optional[str] = None
    git_branch: Optional[str] = None
    title: Optional[str] = None


# =============================================================================
# PARSING
# =============================================================================

def iter_complete_lines(path: str, start: int):
    """Yield (end_offset, line_bytes) for every *complete* line from `start`."""
    with open(path, "rb") as f:
        f.seek(start)
        pos = start
        for line in f:
            if not line.endswith(b"\n"):
                break            # partial last line — still being written
            pos += len(line)
            yield pos, line


def message_text(content) -> str:
    """Flatten a message's content to text (text blocks only; no tool payloads)."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "").strip()
                if t:
                    parts.append(t)
        text = "\n".join(parts)
    else:
        return ""
    text = _STRIP_BLOCKS.sub("", text)
    text = _STRIP_TAGS.sub("", text)
    return text.strip()


def parse_messages(path: str, start: int, meta: SessionMeta) -> Tuple[List[Msg], SessionMeta, int]:
    """
    Parse complete lines from `start`. Returns (messages, updated meta,
    end offset of the last complete line). `meta` is updated in place from
    whatever session fields appear (cwd, gitBranch, aiTitle).
    """
    msgs: List[Msg] = []
    end = start
    for end, line in iter_complete_lines(path, start):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue

        if d.get("cwd"):
            meta.project = d["cwd"]
        if d.get("gitBranch"):
            meta.git_branch = d["gitBranch"]
        if d.get("sessionId"):
            meta.session_id = d["sessionId"]
        t = d.get("type")
        if t == "ai-title" and d.get("aiTitle"):
            meta.title = d["aiTitle"]
            continue
        if t not in ("user", "assistant"):
            continue
        if d.get("isMeta") or d.get("isSidechain"):
            continue
        msg = d.get("message") or {}
        if msg.get("role") not in ("user", "assistant"):
            continue
        text = message_text(msg.get("content", ""))
        if not text:
            continue
        msgs.append(Msg(role=msg["role"], text=text, ts=d.get("timestamp"), end_offset=end))
    return msgs, meta, end


# =============================================================================
# GROUPING + RENDERING
# =============================================================================

def group_messages(msgs: List[Msg], size: Optional[int] = None) -> List[List[Msg]]:
    """Pack whole messages into groups of about `size` characters."""
    size = size or config.CHUNK_SIZE
    groups: List[List[Msg]] = []
    cur: List[Msg] = []
    cur_len = 0
    for m in msgs:
        mlen = len(m.text) + 12
        if mlen > size * 1.5:
            # One oversized message: flush, then split it into its own groups.
            if cur:
                groups.append(cur)
                cur, cur_len = [], 0
            for piece in memory.chunk_text(m.text, size=size, overlap=0):
                groups.append([Msg(m.role, piece, m.ts, m.end_offset)])
            continue
        if cur and cur_len + mlen > size:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(m)
        cur_len += mlen
    if cur:
        groups.append(cur)
    return groups


def _short_date(ts: Optional[str]) -> str:
    return ts[:16].replace("T", " ") if ts else ""


def render_group(group: List[Msg], meta: SessionMeta) -> str:
    """Header line with date · project · title, then the dialogue."""
    bits = [_short_date(group[0].ts)]
    if meta.project:
        bits.append(os.path.basename(meta.project.rstrip("/")) or meta.project)
    bits.append(meta.title or meta.session_id[:8])
    header = "[" + " · ".join(b for b in bits if b) + "]"
    body = "\n\n".join(
        f"{'USER' if m.role == 'user' else 'ASSISTANT'}: {m.text}" for m in group)
    return f"{header}\n{body}"


# =============================================================================
# INCREMENTAL INGEST
# =============================================================================

@dataclass
class IngestSummary:
    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    chunks_stored: int = 0
    chunks_duplicate: int = 0
    errors: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {**self.__dict__, "errors": self.errors[:10]}


class ConvoIngester:
    def __init__(self):
        self._lock = asyncio.Lock()   # one file at a time keeps offsets honest

    async def _state(self, path: str):
        rows = await memory.db.execute_fetchall(
            "SELECT * FROM ingest_state WHERE path = ?", (path,))
        return rows[0] if rows else None

    async def _save_state(self, path: str, meta: SessionMeta, committed: int, next_index: int,
                          tail_id: Optional[int], tail_hash: Optional[str], st: os.stat_result,
                          chunks: int):
        await memory.db.execute(
            "INSERT INTO ingest_state (path, session_id, project, title, git_branch, "
            "committed_offset, next_index, tail_chunk_id, tail_hash, size, mtime, chunks, "
            "legacy_cleared, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?) "
            "ON CONFLICT(path) DO UPDATE SET session_id=excluded.session_id, "
            "project=excluded.project, title=excluded.title, git_branch=excluded.git_branch, "
            "committed_offset=excluded.committed_offset, next_index=excluded.next_index, "
            "tail_chunk_id=excluded.tail_chunk_id, tail_hash=excluded.tail_hash, "
            "size=excluded.size, mtime=excluded.mtime, chunks=excluded.chunks, "
            "legacy_cleared=1, updated_at=excluded.updated_at",
            (path, meta.session_id, meta.project, meta.title, meta.git_branch, committed,
             next_index, tail_id, tail_hash, st.st_size, st.st_mtime, chunks,
             datetime.now(timezone.utc).isoformat()))
        await memory.db.commit()

    async def ingest_file(self, path: str, force: bool = False) -> dict:
        """Ingest whatever is new in one session file. Returns a small report."""
        async with self._lock:
            return await self._ingest_file_locked(path, force)

    async def _ingest_file_locked(self, path: str, force: bool) -> dict:
        path = os.path.abspath(path)
        st = os.stat(path)
        row = await self._state(path)
        name_sid = os.path.basename(path)[:-len(".jsonl")] if path.endswith(".jsonl") \
            else os.path.basename(path)

        if row and not force and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
            return {"path": path, "skipped": True, "stored": 0, "duplicates": 0}

        if force and row:
            # Start over for this file: drop its chunks, forget its offsets.
            await memory.delete_where("source LIKE ? ESCAPE '\\'",
                                      (f"convo:{row['session_id']}:%".replace("_", r"\_"),))
            row = None

        meta = SessionMeta(
            session_id=(row["session_id"] if row else None) or name_sid,
            project=row["project"] if row else None,
            git_branch=row["git_branch"] if row else None,
            title=row["title"] if row else None,
        )
        committed = row["committed_offset"] if row else 0
        next_index = row["next_index"] if row else 0
        chunks_so_far = row["chunks"] if row else 0
        old_tail_id = row["tail_chunk_id"] if row else None
        old_tail_hash = row["tail_hash"] if row else None

        msgs, meta, _ = parse_messages(path, committed, meta)

        if not row or not row["legacy_cleared"]:
            # Replace pre-v5 chunks of this session (no metadata, old chunker).
            await memory.delete_where(
                "source LIKE ? ESCAPE '\\' AND project IS NULL AND session_id IS NULL",
                (f"convo:{meta.session_id}:%".replace("_", r"\_"),))

        groups = group_messages(msgs)
        if not groups:
            await self._save_state(path, meta, committed, next_index, old_tail_id,
                                   old_tail_hash, st, chunks_so_far)
            return {"path": path, "skipped": False, "stored": 0, "duplicates": 0}

        final, tail = groups[:-1], groups[-1]
        tail_text = render_group(tail, meta)
        tail_hash = memory.content_hash(tail_text)

        if not final and old_tail_id and tail_hash == old_tail_hash:
            # File changed (e.g. non-message lines) but the dialogue tail didn't.
            await self._save_state(path, meta, committed, next_index, old_tail_id,
                                   old_tail_hash, st, chunks_so_far)
            return {"path": path, "skipped": False, "stored": 0, "duplicates": 0}

        stored = duplicates = 0
        if old_tail_id:
            await memory.delete_ids([old_tail_id])
            chunks_so_far -= 1

        def _chunk(g: List[Msg], text: str, idx: int) -> ChunkIn:
            return ChunkIn(
                text=text, source=f"convo:{meta.session_id}:{idx}",
                project=meta.project, session_id=meta.session_id,
                ts_start=g[0].ts, ts_end=g[-1].ts,
                metadata={k: v for k, v in {
                    "title": meta.title, "git_branch": meta.git_branch, "path": path,
                }.items() if v},
            )

        if final:
            res = await memory.store_chunks(
                [_chunk(g, render_group(g, meta), next_index + i) for i, g in enumerate(final)])
            stored += res.stored
            duplicates += res.duplicates
            next_index += len(final)
            committed = final[-1][-1].end_offset

        tail_res = await memory.store_chunks([_chunk(tail, tail_text, next_index)])
        stored += tail_res.stored
        duplicates += tail_res.duplicates
        new_tail_id = tail_res.ids[0] if tail_res.ids else None

        chunks_so_far += stored
        await self._save_state(path, meta, committed, next_index, new_tail_id, tail_hash,
                               st, chunks_so_far)
        return {"path": path, "skipped": False, "stored": stored, "duplicates": duplicates,
                "replaced_tail": old_tail_id is not None}

    async def scan(self, dirs: Optional[List[str]] = None, force: bool = False,
                   settle: Optional[float] = None, on_file=None) -> IngestSummary:
        """
        Ingest every *.jsonl under `dirs` that changed since last time.
        `on_file(path, report)` is called after each file (progress reporting).
        """
        dirs = [os.path.expanduser(d) for d in (dirs or config.WATCH_DIRS)]
        settle = config.WATCH_SETTLE if settle is None else settle
        summary = IngestSummary()
        files: List[Path] = []
        for d in dirs:
            if os.path.isdir(d):
                files.extend(Path(d).rglob("*.jsonl"))
        files.sort(key=lambda p: p.stat().st_mtime)
        summary.files_found = len(files)
        now = time.time()
        for p in files:
            try:
                if not force and now - p.stat().st_mtime < settle:
                    summary.files_skipped += 1      # still being written; next pass
                    continue
                rep = await self.ingest_file(str(p), force=force)
                if rep.get("skipped"):
                    summary.files_skipped += 1
                else:
                    summary.files_processed += 1
                    summary.chunks_stored += rep["stored"]
                    summary.chunks_duplicate += rep["duplicates"]
                if on_file:
                    on_file(str(p), rep)
            except Exception as e:  # one bad file must not stop the scan
                logger.warning(f"ingest failed for {p}: {e}")
                summary.errors.append({"file": p.name, "error": str(e)[:200]})
        return summary


ingester = ConvoIngester()


# =============================================================================
# WATCHER
# =============================================================================

class Watcher:
    """Background loop: scan the session dirs every WATCH_INTERVAL seconds."""

    def __init__(self, ingester: ConvoIngester):
        self.ingester = ingester
        self.task: Optional[asyncio.Task] = None
        self.status = {
            "running": False, "dirs": [], "interval": config.WATCH_INTERVAL,
            "scans": 0, "scanning": False, "current_file": None,
            "files_this_scan": 0, "chunks_this_scan": 0,
            "last_scan_at": None, "last_scan_seconds": None,
            "last_result": None, "chunks_stored_total": 0, "last_error": None,
        }

    def start(self):
        dirs = [d for d in config.WATCH_DIRS if os.path.isdir(d)]
        self.status["dirs"] = dirs
        if not dirs:
            logger.info("Watcher: no session directories found; not starting "
                        f"(looked in {config.WATCH_DIRS})")
            return
        self.status["running"] = True
        self.task = asyncio.create_task(self._loop(dirs), name="cloxy-watcher")
        logger.info(f"Watcher: following {dirs} every {config.WATCH_INTERVAL:g}s")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        self.status["running"] = False

    def _progress(self, path: str, rep: dict):
        # Called per file so `cloxy status` is honest during a long first backfill.
        self.status["current_file"] = os.path.basename(path)
        self.status["files_this_scan"] += 1
        self.status["chunks_this_scan"] += rep.get("stored", 0)
        self.status["chunks_stored_total"] += rep.get("stored", 0)

    async def scan_once(self, dirs: List[str]):
        started = time.monotonic()
        self.status.update(scanning=True, current_file=None, files_this_scan=0, chunks_this_scan=0)
        try:
            summary = await self.ingester.scan(dirs, on_file=self._progress)
            self.status["last_result"] = summary.as_dict()
            self.status["last_error"] = None
            if summary.chunks_stored:
                logger.info(f"Watcher: +{summary.chunks_stored} chunks from "
                            f"{summary.files_processed} files")
        except Exception as e:
            self.status["last_error"] = str(e)[:300]
            logger.warning(f"Watcher scan failed: {e}")
        self.status["scans"] += 1
        self.status["scanning"] = False
        self.status["current_file"] = None
        self.status["last_scan_at"] = datetime.now(timezone.utc).isoformat()
        self.status["last_scan_seconds"] = round(time.monotonic() - started, 3)

    async def _loop(self, dirs: List[str]):
        while True:
            await self.scan_once(dirs)
            await asyncio.sleep(config.WATCH_INTERVAL)


watcher = Watcher(ingester)
