"""
Mirror agent session files from this machine to a Cloxy server.

Client (`cloxy push`): walk the local session roots, and for every *.jsonl
that grew, send the new *complete* lines to `POST /push` with the byte offset
they start at. The last offset the server confirmed is kept per file in
~/.cloxy/push_state.json, so each pass only sends what is new.

Server (`apply_push`): append the bytes to
    <DATA_DIR>/inbox/<host>/<agent>/<relative path>
only if the offset matches the mirror's current size. A mismatch returns the
real size (409) and the client resumes from there — that covers a lost state
file, a retried request, and two clients racing. The server's watcher already
follows the inbox, so pushed sessions are ingested like local ones.

This replaces the "rsync on a timer" recipe: no ssh, same API key as the
rest of Cloxy, byte-exact, and resumable.
"""
from __future__ import annotations

import base64
import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import config

AGENTS = ("claude", "codex")
_SAFE_PART = re.compile(r"^[\w.\- ]{1,200}$")
_SAFE_HOST = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# =============================================================================
# SERVER SIDE
# =============================================================================

class PushError(Exception):
    def __init__(self, status: int, message: str, size: Optional[int] = None):
        super().__init__(message)
        self.status, self.message, self.size = status, message, size


def mirror_path(host: str, agent: str, rel: str) -> Path:
    """Validated destination inside the inbox. Raises PushError(400) on anything odd."""
    if not _SAFE_HOST.match(host or ""):
        raise PushError(400, "bad host name")
    if agent not in AGENTS:
        raise PushError(400, f"agent must be one of {', '.join(AGENTS)}")
    parts = (rel or "").replace("\\", "/").split("/")
    if not rel.endswith(".jsonl") or rel.startswith("/") or any(
            p in ("", ".", "..") or not _SAFE_PART.match(p) for p in parts):
        raise PushError(400, "path must be a relative .jsonl path with plain components")
    root = Path(config.inbox_dir()).resolve()
    dest = (root / host / agent).joinpath(*parts).resolve()
    if root not in dest.parents:
        raise PushError(400, "path escapes the inbox")
    return dest


def apply_push(host: str, agent: str, rel: str, offset: int, data_b64: str,
               replace: bool = False) -> Tuple[Path, int, bool]:
    """
    Append (or, with `replace`, rewrite from 0) one slice of a session file.
    Returns (mirror path, new size, rewritten). The slice must be whole lines.
    """
    dest = mirror_path(host, agent, rel)
    try:
        data = base64.b64decode(data_b64, validate=True)
    except ValueError:
        raise PushError(400, "data is not valid base64")
    if data and not data.endswith(b"\n"):
        raise PushError(400, "data must end on a complete line")

    current = dest.stat().st_size if dest.exists() else 0
    if replace:
        if offset != 0:
            raise PushError(400, "replace must start at offset 0")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest, len(data), current > 0
    if offset != current:
        raise PushError(409, f"offset {offset} != mirror size {current}", size=current)
    if data:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "ab") as f:
            f.write(data)
    return dest, current + len(data), False


# =============================================================================
# CLIENT SIDE
# =============================================================================

def default_host() -> str:
    name = socket.gethostname().split(".")[0]
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)[:64]
    return name or "host"


def _state_path() -> Path:
    return Path(config.DATA_DIR) / "push_state.json"


def load_state() -> dict:
    try:
        return json.loads(_state_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=0, sort_keys=True))
    tmp.replace(p)


def complete_slice(path: Path, start: int, max_bytes: int) -> bytes:
    """Bytes from `start`, at most `max_bytes`, cut after the last newline."""
    with open(path, "rb") as f:
        f.seek(start)
        buf = f.read(max_bytes)
    cut = buf.rfind(b"\n")
    if cut >= 0:
        return buf[:cut + 1]
    if len(buf) < max_bytes:
        return b""                       # partial line still being written
    # One line longer than a whole request: read on until it ends.
    with open(path, "rb") as f:
        f.seek(start)
        line = f.readline()
    return line if line.endswith(b"\n") else b""


@dataclass
class PushSummary:
    files_seen: int = 0
    files_pushed: int = 0
    bytes_sent: int = 0
    resyncs: int = 0
    errors: List[str] = field(default_factory=list)


def push_file(post: Callable[[dict], Tuple[int, dict]], host: str, agent: str, root: Path,
              path: Path, state: dict, summary: PushSummary,
              max_bytes: Optional[int] = None) -> None:
    """Send whatever is new in one file. `post(payload)` → (status, json body)."""
    max_bytes = max_bytes or config.PUSH_MAX_BYTES
    key = str(path)
    rel = path.relative_to(root).as_posix()
    size = path.stat().st_size
    sent = int(state.get(key, 0))
    replace = size < sent                # file was truncated/rewritten locally
    if replace:
        sent = 0
    pushed = False
    for _ in range(10_000):              # bounded: each pass moves forward or resyncs
        if sent >= size and not replace:
            break
        data = complete_slice(path, sent, max_bytes)
        if not data and not replace:
            break
        status, body = post({"host": host, "agent": agent, "path": rel, "offset": sent,
                             "data": base64.b64encode(data).decode(), "replace": replace})
        if status == 409 and isinstance(body.get("size"), int):
            summary.resyncs += 1
            server_size = body["size"]
            if server_size <= size:
                sent = server_size       # trust the mirror; send only what it lacks
            else:
                sent, replace = 0, True  # mirror is longer than our file: rewrite it
            continue
        if status >= 400:
            summary.errors.append(f"{rel}: {status} {body.get('error') or body.get('detail')}")
            return
        sent = int(body["size"])
        replace = False
        summary.bytes_sent += len(data)
        pushed = True
        state[key] = sent
    if pushed:
        summary.files_pushed += 1


def push_once(post: Callable[[dict], Tuple[int, dict]], host: Optional[str] = None,
              sources: Optional[List[Tuple[str, str]]] = None,
              max_bytes: Optional[int] = None) -> PushSummary:
    host = host or default_host()
    state = load_state()
    summary = PushSummary()
    for agent, root in (sources or config.PUSH_SOURCES):
        root_p = Path(os.path.expanduser(root))
        if not root_p.is_dir():
            continue
        for path in sorted(root_p.rglob("*.jsonl")):
            summary.files_seen += 1
            try:
                push_file(post, host, agent, root_p, path, state, summary, max_bytes)
            except OSError as e:
                summary.errors.append(f"{path.name}: {e}")
            save_state(state)
    return summary


def is_loopback_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


def run_loop(post, host: Optional[str], interval: float, report: Callable[[PushSummary], None]):
    while True:
        report(push_once(post, host))
        time.sleep(interval)
