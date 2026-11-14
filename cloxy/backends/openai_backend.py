"""
Upstream backend: forward chat to an OpenAI-compatible server (CLOXY_LLM_URL).

The main use is an MLX distributed cluster — `cloxy cluster serve` runs
mlx_lm.server under mlx.launch, so the model is sharded across several Macs
(JACCL over Thunderbolt 5, or ring over TCP) and rank 0 answers HTTP. Anything
else that speaks /v1/chat/completions works too (Ollama, LM Studio, vLLM).

Same interface as mlx_backend: load_model / is_loaded / current_model /
generate_chat / stream_chat (yields StreamPiece).
"""
from __future__ import annotations

import json
import time
from typing import AsyncGenerator, List, Optional

import httpx

from .. import config
from .mlx_backend import StreamPiece

# Tests swap this for an httpx.MockTransport.
_transport: Optional[httpx.AsyncBaseTransport] = None
_model_id: Optional[str] = None


class UpstreamError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status


def _client() -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}
    return httpx.AsyncClient(base_url=config.LLM_URL, headers=headers,
                             timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=10),
                             transport=_transport)


def is_loaded() -> bool:
    return _model_id is not None


def current_model() -> Optional[str]:
    return _model_id


async def load_model(hf_id: Optional[str] = None) -> None:
    """
    Nothing to load locally: check the upstream answers and remember which model
    to name. The configured id wins; otherwise ask /v1/models. mlx_lm.server
    lists every cached model there, so the configured id is the reliable one.
    """
    global _model_id
    try:
        async with _client() as c:
            r = await c.get("/v1/models")
    except httpx.HTTPError as e:
        raise UpstreamError(503, f"LLM upstream {config.LLM_URL} unreachable: {e}") from e
    if r.status_code != 200:
        raise UpstreamError(503, f"LLM upstream {config.LLM_URL} returned HTTP {r.status_code}")
    if hf_id:
        _model_id = hf_id
        return
    data = (r.json() or {}).get("data") or []
    _model_id = data[0]["id"] if data else "upstream"


def _body(messages, max_tokens, temperature, top_p, stream: bool) -> dict:
    # No "model" field: mlx_lm.server would try to load whatever we name, which
    # a sharded cluster can't do mid-flight. Omitted = the model it was started with.
    body = {"messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


async def generate_chat(messages: List[dict], max_tokens: int = 512,
                        temperature: float = 0.7, top_p: float = 0.95) -> dict:
    started = time.perf_counter()
    try:
        async with _client() as c:
            r = await c.post("/v1/chat/completions",
                             json=_body(messages, max_tokens, temperature, top_p, False))
    except httpx.HTTPError as e:
        raise UpstreamError(502, f"LLM upstream error: {e}") from e
    if r.status_code != 200:
        raise UpstreamError(502, f"LLM upstream returned HTTP {r.status_code}: {r.text[:300]}")
    out = r.json()
    elapsed = time.perf_counter() - started
    out["model"] = _model_id or out.get("model")
    toks = (out.get("usage") or {}).get("completion_tokens")
    out["cloxy_meta"] = {
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(toks / elapsed, 2) if toks and elapsed > 0 else None,
        "upstream": config.LLM_URL,
    }
    return out


async def stream_chat(messages: List[dict], max_tokens: int = 512,
                      temperature: float = 0.7, top_p: float = 0.95,
                      ) -> AsyncGenerator[StreamPiece, None]:
    """Relay the upstream SSE stream as StreamPiece fragments."""
    finish, usage = None, None
    try:
        async with _client() as c:
            async with c.stream("POST", "/v1/chat/completions",
                                json=_body(messages, max_tokens, temperature, top_p, True)) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:300]
                    raise UpstreamError(502, f"LLM upstream returned HTTP {r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield StreamPiece(text=text)
                        finish = choice.get("finish_reason") or finish
    except httpx.HTTPError as e:
        raise UpstreamError(502, f"LLM upstream error: {e}") from e
    yield StreamPiece(text="", finish_reason=finish or "stop", usage=usage)
