"""
Shared fixtures. The app runs under Starlette's TestClient with:
  - a deterministic bag-of-words fake embedder (no model download, but
    texts that share words really are closer — so dense recall is testable),
  - a temp SQLite DB,
  - httpx.MockTransport standing in for the internet,
  - DNS patched so the SSRF guard sees whatever IP the test wants,
  - the conversation watcher disabled (tests trigger ingest explicitly).
No MLX, no network.
"""
import hashlib
import re

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cloxy import config, memory, proxy, server


class FakeEmbedder:
    """Feature-hashed bag of words → fixed-dim vector. Deterministic, cheap."""
    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(config.EMBED_DIM, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % config.EMBED_DIM
            v[idx] += 1.0 if h[4] & 1 else -1.0
        if not v.any():
            v[0] = 1.0
        return v

    def embed(self, texts):
        for t in texts:
            yield self._vec(t)


# Everything resolves public unless listed here.
DNS = {"metadata.internal": "169.254.169.254", "169.254.169.254": "169.254.169.254"}
PUBLIC_IP = "93.184.216.34"


def fake_getaddrinfo(host, *a, **k):
    return [(2, 1, 6, "", (DNS.get(host, PUBLIC_IP), 0))]


PAGES = {
    "https://site.example/page": httpx.Response(
        200, headers={"content-type": "text/html"},
        content=b"<html><body><h1>Title One</h1><h2>Sub Two</h2>"
                b"<p>" + b"Body text about quarterly revenue. " * 40 + b"</p></body></html>",
    ),
    "https://site.example/bounce": httpx.Response(
        302, headers={"location": "http://metadata.internal/latest/meta-data/"}),
    "https://site.example/hop": httpx.Response(301, headers={"location": "/page"}),
    "https://site.example/big": httpx.Response(
        200, headers={"content-type": "text/plain"},
        content=b"a" * (config.MAX_CONTENT_LENGTH * 2)),
    "https://site.example/file.pdf": httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4"),
}


def page_handler(request: httpx.Request) -> httpx.Response:
    return PAGES.get(str(request.url), httpx.Response(404))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    data_dir = tmp_path_factory.mktemp("cloxy")
    mp.setattr(config, "DATA_DIR", str(data_dir))
    mp.setattr(config, "DB_PATH", str(data_dir / "memory.db"))
    mp.setattr(config, "CONFIG_PATH", data_dir / "no-llm.json")
    mp.setattr(config, "WATCH", False)
    mp.setattr(config, "ALLOW_PRIVATE_URLS", False)
    mp.setattr(config, "RERANK", False)
    mp.setattr(config, "CHAT_MEMORY", False)
    mp.setattr(memory, "init_embedder", lambda: setattr(memory, "embedder", FakeEmbedder()))
    mp.setattr(proxy.socket, "getaddrinfo", fake_getaddrinfo)
    mp.setattr(proxy, "_make_client",
               lambda: httpx.AsyncClient(transport=httpx.MockTransport(page_handler)))
    with TestClient(server.app) as c:
        yield c
    mp.undo()


@pytest.fixture(autouse=True)
def clear_cache():
    proxy.cache.clear()
