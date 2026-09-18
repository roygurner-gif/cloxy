# Changelog

## v5.1 — E5 prefixes and `cloxy reembed`

- **Query/passage prefixes.** E5-family embedding models are trained with
  `query: ` / `passage: ` prefixes and lose a lot of recall without them;
  fastembed does not add them. Cloxy now picks the prefixes from the model
  name (`CLOXY_EMBED_QUERY_PREFIX` / `CLOXY_EMBED_PASSAGE_PREFIX` override),
  embeds queries and stored text accordingly, and records the passage prefix
  in the DB next to the model and dim.
- **`cloxy reembed`.** Re-embeds every memory in place with the configured
  model and prefix, then records that setup. This is now the way to change
  embedding models, and the one-time fix for an E5 database built before
  this release (the server refuses to start until it runs).

## v5.0 — memory that keeps itself

- **Live ingest.** A watcher follows `~/.claude/projects` and ingests new
  conversation lines seconds after they're written. Incremental by byte
  offset; whole messages are packed into chunks; the trailing partial chunk
  is searchable immediately and replaced as the session grows. Pre-v5 chunks
  are replaced in place on first run.
- **Metadata on every memory.** Project (the session's working directory),
  session id, first/last timestamps, git branch, and the session's AI title.
  Every chunk starts with a `[date · project · title]` header. `/recall`
  filters by `project`, `since`, `until` and returns dated results.
- **Hybrid recall.** SQLite FTS5 keyword search fused with dense cosine via
  reciprocal rank fusion, mild recency weighting, optional cross-encoder
  rerank (`CLOXY_RERANK=1`). `mode` = `hybrid` | `dense` | `keyword`.
- **MCP server.** `cloxy mcp` exposes `recall`, `remember`, `forget`, `fetch`,
  `search_page`, `verify`, `projects`, `memory_status` over stdio. One line
  to add to Claude Code, Cursor, Continue, Zed.
- **Memory in local chat.** `/v1/chat/completions` accepts `"memory": true`
  (or `CLOXY_CHAT_MEMORY=1`) to prepend relevant recall to the prompt.
- **Real package.** `pip install .` → `cloxy` command: `start`, `mcp`,
  `recall`, `ingest`, `status`, `init`, `install-service` (launchd).
  Modules live under `cloxy/` (`server`, `memory`, `proxy`, `convos`,
  `mcp_server`, `cli`, `hardware`, `catalog`, `backends`).
- New endpoints: `GET /projects`, `GET /ingest_status`. `/ingest_convos`
  takes `{convo_dir?, force?}` and runs an incremental pass.
- Existing databases migrate in place (new columns, FTS index built from
  existing rows). Embedding layout unchanged.

## v4.1 — correctness

- Redirect-safe SSRF guard: every hop re-checked, max 5; body streamed and
  cut at 500 KB; binary content types refused with 415.
- Fetch cache keyed by selector and caller headers; cached entries no longer
  mutated on hit.
- One MLX generation at a time; a disconnected streaming client stops
  generation; producer errors surface.
- OpenAI compatibility: content-part lists, `max_completion_tokens`,
  `finish_reason: "length"`, `usage` on stream end.
- `/recall` single query + bounded `top_k`; `/ingest_text` capped; embeddings
  via numpy (byte-identical layout).
- Python floor is 3.11 (numpy ≥ 2.4).

## v4.0 — Apple Silicon

- `cloxy init` wizard: detects M-series chip + unified memory, recommends
  MLX models that fit, downloads, persists config.
- MLX backend via `mlx-lm`; OpenAI-compatible `/v1/chat/completions`
  (streaming + non-streaming) and `/v1/models`.
- Hardening: loopback-only bind by default, SSRF guard, non-blocking
  embeddings, batched dedupe + upsert-on-conflict ingest, O(1) index
  appends, `DELETE /memory/{id}`, `/forget`, `/reindex`, embed-model/dim
  recorded in the DB and enforced on startup, pytest + CI.

## v3.1

- `/verify`: fetch a URL and rank passages by semantic match to a claim.

## v3.0

- numpy matrix vector index, aiosqlite, SHA256 hashing, optional API key
  auth, TTL cache, FastAPI lifespan.

## v2.0

- Initial release: web proxy + conversation RAG.
