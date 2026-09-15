# CLOXY

**Give your local AI eyes and memory — native to your Mac.**

Cloxy is one process that gives any AI tool three things it doesn't have on its own:

- **Memory that keeps itself.** Cloxy watches your Claude Code sessions and ingests them as they happen. Ask "what did we decide about the auth flow last week" and get the actual conversation back — dated, tagged with the project, ranked by a hybrid semantic + keyword search.
- **Eyes.** A web proxy that turns any URL into clean text, markdown, or a CSS-selected extract, and a `/verify` endpoint that ranks a page's passages against a claim.
- **A local LLM** (optional, Apple Silicon). `cloxy init` picks an MLX model that fits your unified memory; `/v1/chat/completions` serves it OpenAI-style — with your memory injected if you want.

All of it is exposed as an **MCP server**, so Claude Code, Cursor, Continue, and Zed pick it up with one line. Nothing leaves your machine.

## Install

Requires Python 3.11+. macOS (Apple Silicon) for the local LLM; the proxy, memory, and MCP server run anywhere.

```bash
# from a clone
pip install .            # proxy + memory + MCP
pip install ".[mlx]"     # + Apple Silicon LLM

# or straight from GitHub
pipx install "git+https://github.com/roygurner-gif/cloxy"
```

## Quick start

```bash
cloxy start              # server on http://127.0.0.1:9055 — the watcher starts ingesting ~/.claude/projects
cloxy status             # memories, watcher, index sizes
cloxy recall "what port does the staging cluster use"
```

Keep it running across logins (macOS):

```bash
cloxy install-service    # launchd agent; logs in ~/.cloxy/logs/cloxy.log
```

### Give Claude Code the tools

```bash
claude mcp add cloxy -- cloxy mcp
```

or in `.mcp.json` (Claude Code, Cursor, Continue, Zed all read this shape):

```json
{ "mcpServers": { "cloxy": { "command": "cloxy", "args": ["mcp"] } } }
```

Tools exposed: `recall`, `remember`, `forget`, `fetch`, `search_page`, `verify`, `projects`, `memory_status`. The MCP server is a thin client of the running Cloxy server (`CLOXY_URL`), so every editor shares one index and one embedder.

## How memory works

```
~/.claude/projects/**/*.jsonl  ──watcher (5s)──▶  parse new lines from last byte offset
                                                       │
                                            pack whole messages into ~1500-char chunks
                                            header: [2026-09-12 14:40 · rmbr · Board colors]
                                                       │
                                     embed (bge-small) ─┼─ SQLite: content + project + session
                                                        │           + ts_start/ts_end + metadata
                                        numpy vector index   +   FTS5 keyword index
                                                       │
                              /recall = dense ⊕ BM25 (reciprocal rank fusion) × recency
```

- **Incremental.** Each session file is tracked by byte offset. Only new lines are read. The last, still-growing chunk is stored so it's searchable immediately and replaced on the next pass.
- **Dated and scoped.** Every memory carries the session's working directory (project), timestamps, git branch, and title. Filter with `project`, `since`, `until`.
- **Hybrid.** Dense vectors catch meaning; FTS5 catches the exact port number, hostname, or flag that embeddings blur. Results are fused and gently tilted toward recent memories (30-day half-life; `recency_weight` 0–1).
- **Optional reranker.** `CLOXY_RERANK=1` runs a small cross-encoder over the top 20.
- **Self-cleaning.** Delete one memory, a whole source, or force a re-ingest; the vector and keyword indexes stay in sync.

Existing v3/v4 databases migrate in place on first start.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/recall` | Hybrid search. `{query, top_k, mode: hybrid\|dense\|keyword, project, since, until, recency_weight, rerank}` |
| `POST` | `/ingest_text` | Store any text. `{text, source, project?, metadata?}` |
| `POST` | `/ingest_convos` | Run an ingest pass now. `{convo_dir?, force?}` |
| `GET` | `/projects` | Projects present in memory with counts and date ranges |
| `GET` | `/ingest_status` | Watcher state |
| `GET` | `/memory_stats` | Counts, sources, index sizes |
| `DELETE` | `/memory/{id}` | Delete one memory |
| `POST` | `/forget` | Delete memories by source prefix (`convo:`, a session id, `manual`…) |
| `POST` | `/reindex` | Rebuild the vector + keyword indexes from the DB |
| `POST` | `/fetch` | Fetch a URL. `{url, mode: clean\|raw\|markdown\|extract, selector?, headers?}` |
| `POST` | `/search` | Fetch a URL, return lines containing a pattern |
| `POST` | `/verify` | Fetch a URL, rank passages by semantic match to a claim |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming or not). Extra: `memory`, `memory_top_k`, `memory_project` |
| `GET` | `/v1/models` | The currently loaded model |
| `GET` | `/health` | Health + watcher summary |

### Examples

```bash
# recall, scoped to one project since a date
curl -s localhost:9055/recall -H 'content-type: application/json' \
  -d '{"query":"why did we switch to WAL mode","project":"cloxy","since":"2026-09-01","top_k":3}'

# remember something
curl -s localhost:9055/ingest_text -H 'content-type: application/json' \
  -d '{"text":"Staging DB is read-only on Fridays.","source":"decision","project":"/w/infra"}'

# read a page as clean text
curl -s localhost:9055/fetch -H 'content-type: application/json' \
  -d '{"url":"https://example.com","mode":"clean"}'

# check a claim against a page
curl -s localhost:9055/verify -H 'content-type: application/json' \
  -d '{"url":"https://example.com/press","claim":"Revenue grew 12% year over year","top_k":3}'
```

`/verify` returns the top-K passages with cosine scores. The caller decides support/contradiction — Cloxy stays a tool, not a judge.

## Local LLM (Apple Silicon)

```bash
pip install ".[mlx]"
cloxy init               # detects chip + memory, recommends MLX models that fit, downloads your pick
cloxy start
```

The model loads on first request (or at startup with `CLOXY_EAGER_LLM=1`). Any OpenAI-compatible client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9055/v1", api_key="not-required")
resp = client.chat.completions.create(
    model="cloxy",
    messages=[{"role": "user", "content": "What did we decide about the auth flow?"}],
    extra_body={"memory": True},          # prepend relevant recall to the prompt
)
print(resp.choices[0].message.content)
```

Continue.dev / Cursor: add an OpenAI-compatible model with base URL `http://localhost:9055/v1` and any API key (or your `CLOXY_API_KEY`).

> Claude Code speaks the Anthropic Messages API, not the OpenAI one, so it can't use Cloxy as its *model* — but it uses Cloxy's memory and eyes through MCP (above).

Why MLX: it's Apple's framework for the unified-memory architecture, it runs in-process (no daemon, no HTTP hop between proxy and model), and it's fast on M-series parts. Cross-platform inference via `llama-cpp-python` is planned as a separate extra.

## CLI

```
cloxy start [--host H] [--port P]   run the server
cloxy mcp                           MCP stdio server (for editors)
cloxy recall QUERY [-k N] [--project P] [--since D] [--until D] [--mode M] [--full] [--json]
cloxy ingest [DIR] [--force]        run an ingest pass now
cloxy status                        health, memory, watcher
cloxy install-service | uninstall-service   launchd (macOS)
cloxy init | show | list            local LLM setup
```

## Configuration

Everything is an environment variable.

| Variable | Default | Description |
|---|---|---|
| `CLOXY_PORT` | `9055` | Server port |
| `CLOXY_HOST` | `127.0.0.1` | Bind address. `0.0.0.0` exposes it on the network — set an API key |
| `CLOXY_URL` | `http://127.0.0.1:9055` | Where the CLI and MCP server find the server |
| `CLOXY_API_KEY` | *(none)* | API key (`X-API-Key` header). Empty = open |
| `CLOXY_DATA_DIR` | `~/.cloxy` | Database, LLM config, logs |
| `CLOXY_WATCH` | `1` | Run the conversation watcher |
| `CLOXY_WATCH_DIRS` | `~/.claude/projects` | Directories to watch (`:`-separated) |
| `CLOXY_WATCH_INTERVAL` | `5` | Seconds between scans |
| `CLOXY_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (recorded in the DB; changing it needs a fresh data dir) |
| `CLOXY_EMBED_DIM` | `384` | Must match the model |
| `CLOXY_RERANK` | *(unset)* | `1` to rerank the top 20 with a cross-encoder |
| `CLOXY_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Reranker |
| `CLOXY_CHAT_MEMORY` | *(unset)* | `1` to inject memory into every chat completion by default |
| `CLOXY_CHAT_MEMORY_TOP_K` | `5` | How many memories to inject |
| `CLOXY_ALLOW_PRIVATE_URLS` | *(unset)* | `1` lets the proxy fetch private/loopback addresses (SSRF risk) |
| `CLOXY_USER_AGENT` | Chrome UA | User agent for web requests |
| `CLOXY_FETCH_TIMEOUT` | `30` | Web fetch timeout, seconds |
| `CLOXY_CONFIG` | `~/.cloxy/config.json` | LLM config written by `cloxy init` |
| `CLOXY_EAGER_LLM` | *(unset)* | `1` loads the LLM at startup |

## Security

- **Loopback by default.** Nothing is reachable off your machine unless you set `CLOXY_HOST=0.0.0.0`.
- **If you expose it, set an API key.** Otherwise anyone on the network can read and write your memory. The key is compared in constant time.
- **SSRF guard.** `/fetch`, `/search`, `/verify` resolve the target and refuse private, loopback, link-local, and cloud-metadata addresses — and re-check every redirect hop (max 5). Bodies are streamed and cut at 500 KB; binary content types are refused.
- **Your transcripts stay local.** The watcher reads `~/.claude/projects` on this machine and writes to `~/.cloxy/memory.db`. No telemetry. The only outbound traffic is `/fetch` requests you make and one-time model downloads.

## Docker (proxy + memory + MCP; no LLM, no watcher)

```bash
docker build -t cloxy .
docker run -p 9055:9055 -v cloxy-data:/data -e CLOXY_API_KEY=change-me cloxy
# or: docker compose up -d
```

The image binds `0.0.0.0` (containers need that) — set `CLOXY_API_KEY`. Feed it with `/ingest_text` or `/ingest_convos` against a mounted directory.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The suite (no network, no models) covers the SSRF guard and redirect walking, cache keys, hybrid recall and filters, incremental ingest, migration, the MCP tools, and the OpenAI request shapes.

## FAQ

**Does it work offline?** Yes. Memory, recall, and the local LLM are offline once models are downloaded. `/fetch` needs the network.

**What if I change the embedding model?** Cloxy refuses to start against a database built with a different model or dimension. Point `CLOXY_DATA_DIR` at a fresh directory and let the watcher rebuild.

**Can I use a model not in the catalog?** `cloxy init` → Custom → any Hugging Face MLX id (usually `mlx-community/...`).

**How is this different from Ollama / LM Studio?** They serve models. Cloxy is the memory and eyes around a model — with a small model of its own on Apple Silicon. Point Cloxy at Ollama's OpenAI endpoint if you prefer it as the brain; the memory works the same.

**Is my data sent anywhere?** No.

## License

MIT
