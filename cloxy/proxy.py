"""
Web proxy: SSRF guard, redirect-aware safe fetch, and content extraction.
"""
import hashlib
import ipaddress
import json
import socket
from typing import Optional
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from cachetools import TTLCache
from markdownify import markdownify as md

from . import config

FETCH_MODES = ("clean", "raw", "markdown", "extract")

# --- Cache (TTL 15 min, max 200 entries) ---
cache: TTLCache = TTLCache(maxsize=200, ttl=900)


def cache_key(url: str, mode: str, selector: Optional[str] = None,
              headers: Optional[dict] = None) -> str:
    # Selector and caller headers change the result, so they're part of the key.
    hdrs = json.dumps(sorted(headers.items())) if headers else ""
    return hashlib.sha256(f"{url}:{mode}:{selector or ''}:{hdrs}".encode()).hexdigest()


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

    Rejects non-http(s) schemes and (unless CLOXY_ALLOW_PRIVATE_URLS is set)
    any host that resolves to a private / loopback / link-local address —
    e.g. 169.254.169.254 cloud metadata, localhost, or RFC1918 services.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "URL must be http or https"

    host = parsed.hostname
    if not host:
        return "URL has no host"

    if config.ALLOW_PRIVATE_URLS:
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
# SAFE FETCH — redirect-aware SSRF guard, streamed + capped body
# =============================================================================

class FetchError(Exception):
    """A fetch failed in a way we want to surface as a specific HTTP status."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# Binary payloads that trafilatura/BeautifulSoup would only turn into noise.
_BLOCKED_CONTENT_PREFIXES = ("image/", "video/", "audio/", "font/")
_BLOCKED_CONTENT_TYPES = {
    "application/pdf", "application/zip", "application/gzip",
    "application/x-tar", "application/octet-stream",
}


def _make_client() -> httpx.AsyncClient:
    # Redirects are walked by hand in safe_get so every hop passes the SSRF
    # guard; with follow_redirects=True a public URL could 302 to 169.254.169.254.
    return httpx.AsyncClient(follow_redirects=False, timeout=config.FETCH_TIMEOUT)


def default_headers() -> dict:
    return {"User-Agent": config.USER_AGENT}


async def safe_get(url: str, headers: Optional[dict] = None) -> tuple:
    """
    GET `url` and return (text, status_code, final_url).

    - Validates the URL and every redirect target against the SSRF guard.
    - Streams the body and stops reading at MAX_CONTENT_LENGTH bytes.
    - Refuses obvious binary content types.
    Raises FetchError with the status to return to the caller.
    """
    headers = headers or default_headers()
    current = url
    async with _make_client() as client:
        for _ in range(config.MAX_REDIRECTS + 1):
            blocked = validate_fetch_url(current)
            if blocked:
                raise FetchError(400, blocked)

            async with client.stream("GET", current, headers=headers) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError(502, "Redirect without Location header")
                    current = str(resp.url.join(location))
                    continue
                if resp.status_code >= 400:
                    raise FetchError(502, f"Upstream returned HTTP {resp.status_code}")

                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype.startswith(_BLOCKED_CONTENT_PREFIXES) or ctype in _BLOCKED_CONTENT_TYPES:
                    raise FetchError(415, f"Unsupported content type: {ctype}")

                buf = bytearray()
                async for part in resp.aiter_bytes():
                    buf.extend(part)
                    if len(buf) >= config.MAX_CONTENT_LENGTH:
                        break
                encoding = resp.charset_encoding or "utf-8"
                text = bytes(buf[:config.MAX_CONTENT_LENGTH]).decode(encoding, errors="replace")
                return text, resp.status_code, str(resp.url)

    raise FetchError(502, f"Too many redirects (>{config.MAX_REDIRECTS})")


# =============================================================================
# EXTRACTION
# =============================================================================

def clean_text(html: str, include_links: bool = True) -> str:
    """Main-content extraction via trafilatura, with a BeautifulSoup fallback."""
    cleaned = trafilatura.extract(html, include_links=include_links, include_tables=True)
    if not cleaned:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        cleaned = soup.get_text(separator="\n", strip=True)
    return cleaned or ""


def extract(html: str, mode: str, selector: Optional[str] = None) -> dict:
    """Apply a fetch mode to raw HTML. Returns the result fields for that mode."""
    if mode == "raw":
        return {"content": html}

    if mode == "clean":
        cleaned = clean_text(html)
        return {"content": cleaned, "length": len(cleaned)}

    if mode == "markdown":
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        markdown = md(str(soup), heading_style="ATX", strip=["img"]).strip()
        return {"content": markdown, "length": len(markdown)}

    if mode == "extract":
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector or "")
        content = "\n---\n".join(el.get_text(strip=True) for el in elements)
        return {"content": content, "matches": len(elements), "length": len(content)}

    raise ValueError(f"Unknown mode: {mode}")
