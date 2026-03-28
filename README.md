# CLOXY

**Give your local AI eyes and memory.**

Cloxy is a lightweight proxy that gives AI tools (Claude Code, local LLMs, coding assistants) unrestricted web access and persistent conversation memory — running entirely on your own hardware.

## The Problem

Local LLMs and AI coding tools are smart but blind and amnesiac:
- **No web access** — they can't browse the internet, or hit content restrictions when they try
- **No memory** — every conversation starts from scratch

## The Solution

Cloxy runs on your machine and provides two things:
1. **Web Proxy** — fetch any URL, get back clean text, markdown, or raw HTML. No content filtering, no restrictions.
2. **Conversation Memory** — ingest past conversations into a local RAG database. Your AI can recall what you actually discussed instead of hallucinating.

One file. One command. Your hardware, your rules.

## Quick Start

```bash
pip install -r requirements.txt
python cloxy.py
```

Cloxy starts on `http://localhost:9055`.

### Docker

```bash
docker build -t cloxy .
docker run -p 9055:9055 -v cloxy-data:/data cloxy
```

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

### Check status

```bash
curl http://localhost:9055/health
curl http://localhost:9055/memory_stats
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/fetch` | Fetch and clean a URL |
| `POST` | `/search` | Fetch URL, extract lines matching a pattern |
| `POST` | `/ingest_convos` | Parse Claude Code conversations into memory |
| `POST` | `/ingest_text` | Store any text into memory |
| `POST` | `/recall` | Semantic search over memory |
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
| `CLOXY_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for memory |
| `CLOXY_USER_AGENT` | Chrome UA | User agent for web requests |
| `CLOXY_FETCH_TIMEOUT` | `30` | Web fetch timeout in seconds |

## How It Works

**Web Proxy**: Cloxy fetches URLs using httpx with a real browser user agent, then extracts clean content using trafilatura (the same library used by academic web scraping projects). Results are cached for 15 minutes.

**Memory**: Conversations are parsed from Claude Code's JSONL format, chunked into ~1500 character segments with overlap, embedded using a local embedding model (BAAI/bge-small-en-v1.5 via fastembed), and stored in SQLite. Recall uses cosine similarity search over the embeddings.

## Use Cases

- **Claude Code users** who hit WebFetch content restrictions
- **Local LLM users** (MLX, Ollama, llama.cpp) who want their model to browse the web
- **AI coding assistants** that need persistent memory across sessions
- **Privacy-focused developers** who want AI capabilities without cloud dependencies
- **Self-hosters** building autonomous AI agents on their own hardware

## Architecture

```
[Your AI tool] --HTTP--> [Cloxy :9055]
                            ├── /fetch    --> httpx --> any website
                            ├── /recall   --> SQLite + fastembed --> semantic search
                            └── /ingest   --> chunk + embed --> SQLite
```

Everything runs locally. No external APIs. No telemetry. No cloud.

## License

MIT

## Author

Roy Gurner — [roygurner.com](https://roygurner.com)
