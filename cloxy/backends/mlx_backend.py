"""
MLX backend for Cloxy.

Wraps mlx-lm to provide a small async-friendly interface that the FastAPI
layer can call. Exposes:
  - load_model(hf_id) — pulls the model into unified memory
  - generate_chat(messages, max_tokens, ...) — non-streaming
  - stream_chat(messages, max_tokens, ...) — async generator of StreamPiece
  - is_loaded() / current_model() — introspection

Generation is serialized through one lock: MLX runs a single model on a single
Metal command queue, so concurrent generate() calls contend on the GPU and
balloon memory instead of running in parallel.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

# mlx-lm imports are deferred until load_model() so that
# `import backends.mlx_backend` is cheap and doesn't pull MLX
# into memory at FastAPI startup unless an LLM is configured.

_model = None
_tokenizer = None
_current_hf_id: Optional[str] = None
_load_lock = asyncio.Lock()
_gen_lock = asyncio.Lock()   # one generation at a time on the GPU


@dataclass
class StreamPiece:
    """One streamed fragment. finish_reason/usage are set only on the last piece."""
    text: str
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None


def is_loaded() -> bool:
    return _model is not None and _tokenizer is not None


def current_model() -> Optional[str]:
    return _current_hf_id


async def load_model(hf_id: str) -> None:
    """
    Pull `hf_id` from Hugging Face (or cache) and load into unified memory.
    Idempotent — calling again with the same id is a no-op.
    """
    global _model, _tokenizer, _current_hf_id

    if _current_hf_id == hf_id and is_loaded():
        return

    async with _load_lock:
        if _current_hf_id == hf_id and is_loaded():
            return

        # Defer import so module import is cheap.
        from mlx_lm import load

        # mlx_lm.load is blocking; run it off the event loop.
        loop = asyncio.get_running_loop()
        model, tokenizer = await loop.run_in_executor(None, load, hf_id)

        _model = model
        _tokenizer = tokenizer
        _current_hf_id = hf_id


def _apply_chat_template(messages: List[dict]) -> str:
    """
    Render a list of {role, content} messages into a model-specific prompt
    string using the tokenizer's chat template.
    """
    if _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def generate_chat(
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict:
    """
    Non-streaming chat completion. Returns an OpenAI-compatible dict.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded.")

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _apply_chat_template(messages)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    loop = asyncio.get_running_loop()
    async with _gen_lock:
        started = time.perf_counter()
        text = await loop.run_in_executor(
            None,
            lambda: generate(
                _model,
                _tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            ),
        )
        elapsed = time.perf_counter() - started

    # mlx-lm returns the generated text only (not the prompt).
    completion_tokens = len(_tokenizer.encode(text))
    prompt_tokens = len(_tokenizer.encode(prompt))
    # generate() doesn't report why it stopped; hitting the budget is the tell.
    finish_reason = "length" if completion_tokens >= max_tokens else "stop"

    return {
        "id": f"cloxy-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _current_hf_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": _usage(prompt_tokens, completion_tokens),
        "cloxy_meta": {
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(completion_tokens / elapsed, 2) if elapsed > 0 else None,
        },
    }


async def stream_chat(
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> AsyncGenerator[StreamPiece, None]:
    """
    Streaming chat completion. Yields StreamPiece fragments as they are
    produced; the last piece carries finish_reason ("stop" / "length") and
    usage. The FastAPI route is responsible for SSE-framing them.

    Closing the generator early (client disconnect) sets a stop flag that the
    producer thread checks every token, so generation doesn't run on to
    max_tokens for nobody.
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded.")

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _apply_chat_template(messages)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    # stream_generate is a sync generator; bridge it to async via a thread.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    _SENTINEL = object()

    def _producer():
        try:
            last = None
            for response in stream_generate(
                _model,
                _tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if stop.is_set():
                    break
                last = response
                # Newer mlx-lm yields a GenerationResponse with .text;
                # older versions yield a raw string.
                text = getattr(response, "text", response)
                loop.call_soon_threadsafe(queue.put_nowait, StreamPiece(text=text))
            if last is not None and not stop.is_set():
                finish = getattr(last, "finish_reason", None) or "stop"
                p_tok = getattr(last, "prompt_tokens", None)
                g_tok = getattr(last, "generation_tokens", None)
                usage = _usage(p_tok, g_tok) if p_tok is not None and g_tok is not None else None
                loop.call_soon_threadsafe(
                    queue.put_nowait, StreamPiece(text="", finish_reason=finish, usage=usage)
                )
        except Exception as e:  # surface generation errors to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    async with _gen_lock:
        fut = loop.run_in_executor(None, _producer)
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            await fut
