FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY cloxy ./cloxy
RUN pip install --no-cache-dir .

# Proxy + memory + MCP only: MLX needs Apple Silicon. Watcher off — there are
# no Claude Code session files inside a container; ingest over HTTP instead.
ENV CLOXY_DATA_DIR=/data \
    CLOXY_HOST=0.0.0.0 \
    CLOXY_WATCH=0
VOLUME /data
EXPOSE 9055

CMD ["cloxy", "start"]
