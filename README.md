# CLOXY

**Local AI with eyes and memory — native to your Mac.**

Cloxy is a lightweight stack that gives you a local LLM plus unrestricted web access plus persistent conversation memory — all running natively on Apple Silicon. One install, one command, your hardware.

## The Problem

Local LLMs are smart but blind and amnesiac:
- **No web access** — they can't browse the internet, or hit content restrictions when they try
- **No memory** — every conversation starts from scratch
- **Multiple tools to set up** — pick a model runtime, pick a model, wire it to your AI tools, hope nothing breaks

## The Solution

Cloxy runs on your Mac and provides:
1. **LLM bootstrap** — `cloxy init` detects your hardware, recommends MLX-Community models that fit in unified memory, downloads your pick, and wires it in. No separate Ollama, no separate LM Studio.
2. **OpenAI-compatible chat endpoint** — `/v1/chat/completions` works with any tool that speaks the OpenAI API (Claude Code, Continue.dev, Cursor, custom scripts).
3. **Web Proxy** — fetch any URL, get back clean text, markdown, or raw HTML. No content filtering, no restrictions.
4. **Conversation Memory** — ingest past conversations into a local RAG database. Your AI can recall what you actually discussed instead of hallucinating.

One install. Your hardware. Your AI.

## What's New in v4.0 — Apple Silicon

- **`cloxy init` wizard** — detects M-series chip + unified memory, recommends MLX models that fit, downloads your pick, persists the config.
- **MLX backend** — native Apple Silicon inference via `mlx-lm`. Significantly faster than llama.cpp on M-series hardware because it uses the unified-memory architecture directly.
- **`/v1/chat/completions`** — OpenAI-compatible streaming + non-streaming chat endpoint. Drop-in compatible with any tool that speaks the OpenAI API.
- **`/v1/models`** — lists the currently-loaded model.
- **Memory-aware recommendations** — Cloxy won't suggest a model that won't fit. Try anyway with the "Custom" option if you know your hardware; you'll get a warning before the download.

## v3.1

- **`/verify` endpoint** — fetch a URL and rank passages by semantic match to a claim. Cloxy returns the top-K most relevant chunks with cosine similarity scores; the calling agent reads them and decides support/contradiction. Reuses the `/fetch` clean-mode cache so follow-up fetches are free.

## v3.0

- **Numpy matrix vector index** — pre-normalized embeddings, cosine similarity via single matrix multiply. No Python loops, no full table scans.
- **aiosqlite** — fully async database access, no thread-safety hacks.
- **SHA256** hashing for content dedup and cache keys.
- **Optional API key auth** — set `CLOXY_API_KEY` to lock it down.
- **TTL cache** via cachetools — proper eviction, no hand-rolled LRU.
- **FastAPI lifespan** — modern lifecycle management, no deprecated decorators.

## Requirements

- **Apple Silicon Mac** (M1, M2, M3, M4 or later) running macOS 13+
- **Python 3.10+**
- Enough unified memory for the model you want (see the catalog below)

Cloxy is currently Apple Silicon only. Cross-platform support (Linux / Windows via `llama-cpp-python`) is on the roadmap as a separate release.

## Quick Start

```bash
pip install -r requirements.txt
python cli.py init      # pick a model, download it
python cloxy.py         # start the server
```

Cloxy starts on `http://localhost:9055`. The LLM is loaded on first chat request, or eagerly at boot if you set `CLOXY_EAGER_LLM=1`.

### Model catalog

```bash
python cli.py list
```

Shipping with curated MLX-Community models from Phi-3 mini (2.5 GB) up through Llama 3.1 405B (220 GB, included as a tongue-in-cheek option for Studio Ultra users).

### Chat with the LLM (OpenAI-compatible)

```bash
curl -X POST http://localhost:9055/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hi."}],
    "max_tokens": 64
  }'
```

### Docker (proxy + RAG only — no LLM)

> **Note:** Docker mode runs Cloxy's web proxy, memory, and verify endpoints, but **does not include the local LLM**. MLX needs direct access to Apple Silicon's unified memory and Neural Engine, which standard Docker virtualization doesn't expose. For the full v4 experience (LLM + proxy + memory), run Cloxy natively with `python cloxy.py`. Use Docker only if you're running cloxy as a pure backend for an external LLM.

```bash
docker build -t cloxy .
docker run -p 9055:9055 -v cloxy-data:/data cloxy
# or:
docker compose up -d
```

## Why MLX (not Ollama or llama.cpp)?

MLX is Apple's machine-learning framework, designed specifically for the unified-memory architecture of M-series chips. On Apple Silicon it's materially faster than llama.cpp-based runtimes (Ollama, LM Studio) because there's no abstraction layer between the model weights and the GPU — the model lives in the same memory the GPU reads from.

For Cloxy specifically, MLX also fits cleanly into the FastAPI Python stack with no separate daemon, no HTTP roundtrip between proxy and LLM, no extra process to manage. Cloxy + LLM is **one Python process, one port, one install**.

Cross-platform support (Linux via vLLM or `llama-cpp-python`, Windows via `llama-cpp-python`) is planned as a separate release. v4 is intentionally Apple Silicon native.

## Integrations

Cloxy's `/v1/chat/completions` is OpenAI-compatible, so anything that speaks the OpenAI API can use it.

### Claude Code

Set the model endpoint in your settings:

```jsonc
{
  "model": "cloxy",
  "modelEndpoint": "http://localhost:9055/v1",
  "apiKey": "not-required"  // unless you set CLOXY_API_KEY
}
```

### Continue.dev (VS Code / JetBrains)

In your `~/.continue/config.json`:

```jsonc
{
  "models": [{
    "title": "Cloxy (local)",
    "provider": "openai",
    "model": "cloxy",
    "apiBase": "http://localhost:9055/v1",
    "apiKey": "not-required"
  }]
}
```

### Cursor

Settings → Models → Add custom OpenAI-compatible model:
- Name: `cloxy`
- Base URL: `http://localhost:9055/v1`
- API key: anything (or your `CLOXY_API_KEY` if set)

### Plain OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9055/v1",
    api_key="not-required",
)

resp = client.chat.completions.create(
    model="cloxy",
    messages=[{"role": "user", "content": "Summarize the v4 release."}],
)
print(resp.choices[0].message.content)
```

## Performance

Approximate throughput on common Apple Silicon configs (4-bit MLX models, single user, no streaming overhead). Real numbers vary with context length, prompt complexity, and what else your Mac is doing.

| Hardware | Model | Tokens/sec (approx) |
|---|---|---|
| M2 16 GB | Qwen 2.5 7B | ~35-50 |
| M3 Pro 32 GB | Qwen 2.5 14B | ~25-35 |
| M4 32 GB | Qwen 2.5 14B | ~30-40 |
| M4 Pro 64 GB | Qwen 2.5 32B | ~18-25 |
| M3 Ultra 192 GB | Llama 3.1 70B | ~12-18 |

MLX is generally 1.5-2.5x faster than llama.cpp / Ollama on the same Apple Silicon hardware, and uses unified memory more efficiently.

## Usage

### Fetch a webpage

```bash
# Clean text (default — strips nav, ads, scripts)
curl -X POST http://localhost:9055/fetch \
  -H "Content-Type: application/json" \
  -d '{"url": "https://breakingdefense.com", "mode": "clean"}'

# Markdown
curl -X POST http://localhost:9055/fetch \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "mode": "markdown"}'

# Raw HTML
curl -X POST http://localhost:9055/fetch \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "mode": "raw"}'

# Extract specific CSS selector
curl -X POST http://localhost:9055/fetch \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "mode": "extract", "selector": "h2.title"}'
```

### Search a webpage for a pattern

```bash
curl -X POST http://localhost:9055/search \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "pattern": "quarterly revenue"}'
```

### Verify a claim against a webpage

```bash
curl -X POST http://localhost:9055/verify \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://justice.gov/opa/pr/...",
    "claim": "Castro was indicted for the 1996 Brothers to the Rescue shootdown",
    "top_k": 3
  }'
```

Returns the top-K cleaned passages from the page ranked by cosine similarity against the claim. The calling agent reads the passages and decides whether they support, contradict, or fail to address the claim. Cloxy stays a tool, not a judge.

### Ingest Claude Code conversations

```bash
# Ingest all conversations from Claude Code
curl -X POST http://localhost:9055/ingest_convos \
  -H "Content-Type: application/json" \
  -d '{"convo_dir": "~/.claude/projects"}'
```

### Ingest any text

```bash
curl -X POST http://localhost:9055/ingest_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Important context to remember...", "source": "meeting-notes"}'
```

### Recall from memory

```bash
curl -X POST http://localhost:9055/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "what architecture did we decide on", "top_k": 5}'
```

### With API key auth

```bash
# Set the key
export CLOXY_API_KEY="your-secret-key"
python cloxy.py

# Include in requests
curl -X POST http://localhost:9055/fetch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"url": "https://example.com", "mode": "clean"}'
```

### Check status

```bash
curl http://localhost:9055/health
curl http://localhost:9055/memory_stats
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming + non-streaming) |
| `GET` | `/v1/models` | List currently-loaded model (OpenAI shape) |
| `POST` | `/fetch` | Fetch and clean a URL |
| `POST` | `/search` | Fetch URL, extract lines matching a pattern (keyword/substring) |
| `POST` | `/verify` | Fetch URL, rank passages by semantic match to a claim |
| `POST` | `/ingest_convos` | Parse Claude Code conversations into memory |
| `POST` | `/ingest_text` | Store any text into memory |
| `POST` | `/recall` | Semantic search over memory (numpy vector index) |
| `GET` | `/memory_stats` | Memory database stats |
| `GET` | `/health` | Health check |
| `GET` | `/` | Service info |

## Fetch Modes

| Mode | Description |
|------|-------------|
| `clean` | Main content extracted via trafilatura (default) |
| `raw` | Full HTML response |
| `markdown` | HTML converted to markdown |
| `extract` | Content from specific CSS selector |

## Configuration

All config via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOXY_PORT` | `9055` | Server port |
| `CLOXY_DATA_DIR` | `~/.cloxy` | Database and data directory |
| `CLOXY_API_KEY` | *(none)* | API key for auth (empty = open) |
| `CLOXY_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for memory |
| `CLOXY_USER_AGENT` | Chrome UA | User agent for web requests |
| `CLOXY_FETCH_TIMEOUT` | `30` | Web fetch timeout in seconds |
| `CLOXY_CONFIG` | `~/.cloxy/config.json` | LLM config file written by `cli.py init` |
| `CLOXY_EAGER_LLM` | *(unset)* | If `1`, load the LLM at server startup instead of on first request |

## How It Works

**Web Proxy**: Cloxy fetches URLs using httpx with a real browser user agent, then extracts clean content using trafilatura (the same library used by academic web scraping projects). Results are cached for 15 minutes.

**Memory**: Conversations are parsed from Claude Code's JSONL format, chunked into ~1500 character segments with overlap, embedded using a local embedding model (BAAI/bge-small-en-v1.5 via fastembed), and stored in SQLite. Recall uses an in-memory numpy matrix — cosine similarity via single matrix multiply, no Python loops or table scans.

## Architecture

```
[Your AI tool] --HTTP--> [Cloxy :9055]
                            ├── /v1/chat/completions --> MLX --> Qwen/Llama in unified memory
                            ├── /fetch               --> httpx --> any website
                            ├── /recall              --> numpy vector index --> semantic search
                            └── /ingest              --> chunk + embed --> SQLite + vec
```

Everything runs locally. No external APIs. No telemetry. No cloud. The LLM, the web proxy, the memory index — all in one Python process on your Mac.

## Stack

- Python 3.12+ / FastAPI / uvicorn
- **MLX / mlx-lm** for LLM inference (Apple Silicon native)
- httpx / trafilatura / BeautifulSoup / markdownify
- fastembed (BAAI/bge-small-en-v1.5) / numpy / aiosqlite
- cachetools (TTL cache)

## FAQ

**Q: Can I use my own model that isn't in the catalog?**
Yes. `cloxy init` has a "Custom" option — enter any Hugging Face MLX model id (typically from the `mlx-community` org). The wizard will warn if it looks unreasonably large for your hardware but will let you proceed.

**Q: Does Cloxy work offline?**
Yes, once a model is downloaded. The LLM, RAG, and memory all run locally. The web proxy obviously needs network access for `/fetch`, but the LLM and `/recall` don't.

**Q: Can I run multiple models?**
Currently Cloxy loads one model at a time. Switching means running `cloxy init` again and restarting. Multi-model support is a possible future feature.

**Q: Why no Linux / Windows support?**
v4 is intentionally Apple Silicon native to take full advantage of MLX. A separate release with `llama-cpp-python` as a cross-platform fallback is on the roadmap. The Apple Silicon focus is deliberate — it's where unified memory makes commodity-hardware AI viable in ways x86 + discrete GPU can't match without dedicated server-class hardware.

**Q: Is my data sent anywhere?**
No. No telemetry, no analytics, no model usage reporting. Cloxy is local-only. The only outbound traffic is `/fetch` calls to URLs you explicitly request, and one-time model downloads from Hugging Face.

**Q: How does Cloxy compare to Ollama?**
Ollama is great. It's cross-platform and has a huge model registry. Cloxy is narrower (Apple Silicon only, one model at a time) but adds two things Ollama doesn't have built-in: persistent RAG memory and an integrated web proxy. Think of Cloxy as "Ollama + eyes + memory, native to Apple Silicon."

**Q: How does Cloxy compare to LM Studio?**
LM Studio has a GUI and broader model support. Cloxy is CLI-only and Apple Silicon focused, with web access and memory as first-class features instead of add-ons.

## License

MIT

## Author

Roy Gurner — [roygurner.com](https://roygurner.com)
