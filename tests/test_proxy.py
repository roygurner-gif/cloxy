"""SSRF guard and cache-key tests (no network)."""
from cloxy import config, proxy


def _patch_resolve(monkeypatch, ip):
    monkeypatch.setattr(proxy.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", (ip, 0))])


def test_ssrf_rejects_non_http_scheme():
    assert proxy.validate_fetch_url("file:///etc/passwd") is not None
    assert proxy.validate_fetch_url("ftp://example.com") is not None


def test_ssrf_blocks_cloud_metadata(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_URLS", False)
    _patch_resolve(monkeypatch, "169.254.169.254")
    assert proxy.validate_fetch_url("http://metadata.internal/latest") is not None


def test_ssrf_blocks_loopback_and_private(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_URLS", False)
    for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.10"):
        _patch_resolve(monkeypatch, ip)
        assert proxy.validate_fetch_url("http://internal.example/") is not None


def test_ssrf_allows_public_host(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_URLS", False)
    _patch_resolve(monkeypatch, "93.184.216.34")
    assert proxy.validate_fetch_url("https://example.com") is None


def test_ssrf_opt_in_allows_private(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_URLS", True)
    assert proxy.validate_fetch_url("http://localhost:8080") is None


def test_cache_key_includes_selector_and_headers():
    base = proxy.cache_key("https://a.example", "extract")
    assert proxy.cache_key("https://a.example", "extract", "h1") != base
    assert proxy.cache_key("https://a.example", "extract", "h1") != \
        proxy.cache_key("https://a.example", "extract", "h2")
    assert proxy.cache_key("https://a.example", "clean", None, {"Cookie": "a"}) != \
        proxy.cache_key("https://a.example", "clean")
    assert proxy.cache_key("https://a.example", "clean", None, {"A": "1", "B": "2"}) == \
        proxy.cache_key("https://a.example", "clean", None, {"B": "2", "A": "1"})


def test_extract_modes():
    html = "<html><body><h1>T</h1><p>Hello <b>there</b></p><script>x()</script></body></html>"
    assert proxy.extract(html, "raw")["content"] == html
    assert "Hello" in proxy.extract(html, "clean")["content"]
    assert "# T" in proxy.extract(html, "markdown")["content"]
    ex = proxy.extract(html, "extract", "h1")
    assert ex == {"content": "T", "matches": 1, "length": 1}
