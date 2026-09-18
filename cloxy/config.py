"""
All runtime configuration, read once from environment variables.

Other modules access these as `config.NAME` (not `from config import NAME`)
so tests can monkeypatch a value and have every reader see it.
"""
import os
from pathlib import Path


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False", "no")


def _load_env_file(path: str) -> None:
    """
    Read KEY=VALUE lines into the environment without overriding anything
    already set. Lets a client machine keep CLOXY_URL / CLOXY_API_KEY in
    ~/.cloxy/client.env so `cloxy status`, `cloxy recall` and `cloxy mcp`
    just work.
    """
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key.startswith("CLOXY_") or key in ("FASTEMBED_CACHE_PATH", "HF_TOKEN"):
                    os.environ.setdefault(key, value)
    except OSError:
        pass


# --- Storage (first: the env file lives here) ---
DATA_DIR = os.environ.get("CLOXY_DATA_DIR", os.path.expanduser("~/.cloxy"))
_load_env_file(os.path.join(DATA_DIR, "client.env"))
DB_PATH = os.path.join(DATA_DIR, "memory.db")
CONFIG_PATH = Path(os.environ.get("CLOXY_CONFIG", os.path.join(DATA_DIR, "config.json")))

# --- Server ---
PORT = int(os.environ.get("CLOXY_PORT", 9055))
# Bind loopback-only by default. Set CLOXY_HOST=0.0.0.0 to expose on the network
# (do that only behind CLOXY_API_KEY — see the Security section of the README).
HOST = os.environ.get("CLOXY_HOST", "127.0.0.1")
API_KEY = os.environ.get("CLOXY_API_KEY", "")  # empty = no auth
URL = os.environ.get("CLOXY_URL", f"http://127.0.0.1:{PORT}")  # what clients (CLI, MCP) talk to

# --- Web proxy ---
USER_AGENT = os.environ.get(
    "CLOXY_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)
FETCH_TIMEOUT = float(os.environ.get("CLOXY_FETCH_TIMEOUT", 30))
MAX_CONTENT_LENGTH = 500_000      # bytes read from a fetched page (streamed, then cut)
MAX_REDIRECTS = 5                 # each hop is re-checked by the SSRF guard
# Allow the proxy to reach private/loopback/link-local addresses. Off by default
# so an exposed instance can't be used to pivot into internal services / cloud
# metadata (169.254.169.254). Set CLOXY_ALLOW_PRIVATE_URLS=1 for local scraping.
ALLOW_PRIVATE_URLS = _flag("CLOXY_ALLOW_PRIVATE_URLS")

# --- Memory ---
EMBED_MODEL = os.environ.get("CLOXY_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("CLOXY_EMBED_DIM", 384))


def default_embed_prefixes(model: str) -> tuple:
    """
    (query prefix, passage prefix) the model was trained with. E5 models
    expect "query: " / "passage: " and measurably lose recall without them
    (fastembed does not add them); BGE and most others take raw text.
    """
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("e5-") or "-e5-" in name or name.startswith("multilingual-e5"):
        return ("query: ", "passage: ")
    return ("", "")


_QUERY_PREFIX, _PASSAGE_PREFIX = default_embed_prefixes(EMBED_MODEL)
EMBED_QUERY_PREFIX = os.environ.get("CLOXY_EMBED_QUERY_PREFIX", _QUERY_PREFIX)
# Recorded in the DB like the model; changing it means `cloxy reembed`.
EMBED_PASSAGE_PREFIX = os.environ.get("CLOXY_EMBED_PASSAGE_PREFIX", _PASSAGE_PREFIX)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_INGEST_CHARS = 2_000_000      # cap on a single /ingest_text payload
# Optional cross-encoder rerank of the top candidates (extra model download).
RERANK = _flag("CLOXY_RERANK")
RERANK_MODEL = os.environ.get("CLOXY_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# --- Conversation watcher (live ingest of Claude Code sessions) ---
WATCH = _flag("CLOXY_WATCH", "1")
WATCH_DIRS = [
    os.path.expanduser(p) for p in
    os.environ.get("CLOXY_WATCH_DIRS", "~/.claude/projects").split(os.pathsep) if p
]
WATCH_INTERVAL = float(os.environ.get("CLOXY_WATCH_INTERVAL", 5))
WATCH_SETTLE = 2.0                # seconds a file must be quiet before we read it

# --- Local LLM ---
EAGER_LLM = _flag("CLOXY_EAGER_LLM")
# Prepend relevant memory to every /v1/chat/completions request by default.
# Per-request override: {"memory": true|false} in the body.
CHAT_MEMORY = _flag("CLOXY_CHAT_MEMORY")
CHAT_MEMORY_TOP_K = int(os.environ.get("CLOXY_CHAT_MEMORY_TOP_K", 5))
