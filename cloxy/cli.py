"""
Cloxy CLI.

  cloxy init               detect hardware, pick + download an MLX model
  cloxy start              run the server (proxy + memory + watcher + LLM)
  cloxy mcp                run the MCP stdio server (for Claude Code, Cursor, ...)
  cloxy recall "query"     search memory from the shell
  cloxy ingest [DIR]       run an ingest pass now (the watcher does this live)
  cloxy status             health + memory + watcher summary
  cloxy install-service    keep the server running via launchd (macOS)
  cloxy uninstall-service
  cloxy show / list        current LLM config / model catalog
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from . import __version__, config

PLIST_LABEL = "com.cloxy.server"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"


# =============================================================================
# LLM setup (init / show / list)
# =============================================================================

def _write_config(model) -> None:
    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_PATH.write_text(json.dumps({
        "model_short_name": model.short_name,
        "model_hf_id": model.hf_id,
        "model_load_gb": model.load_gb,
    }, indent=2))
    print(f"\n  Config saved to {config.CONFIG_PATH}")


def _read_config() -> dict | None:
    if not config.CONFIG_PATH.exists():
        return None
    try:
        return json.loads(config.CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _download_model(hf_id: str) -> None:
    print(f"\n  Downloading {hf_id}")
    print("  (this can take a while on first run; subsequent loads are instant)")
    try:
        from mlx_lm import load
    except ImportError:
        print("\n  ERROR: mlx-lm is not installed. Run: pip install 'cloxy[mlx]'")
        sys.exit(1)
    load(hf_id)   # pulls weights into the HF cache; we discard the model object
    print("  Download complete.")


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix} ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def cmd_init(args: argparse.Namespace) -> int:
    from .hardware import NotAppleSiliconError, detect
    from .catalog import Model, fits, recommend

    print("Cloxy init — detecting your hardware...\n")
    try:
        hw = detect()
    except NotAppleSiliconError as e:
        print(f"  {e}")
        print("  You can still run `cloxy start` for the proxy + memory features.")
        return 1

    print(hw.pretty())
    print()

    recs = recommend(hw.ai_available_gb)
    if not recs:
        print("  Your machine is too small for any catalog model (need ≥3 GB).")
        print("  You can still run Cloxy without an LLM for the proxy/memory features.")
        return 1

    print("Recommended models for your hardware (largest fitting first):\n")
    for i, m in enumerate(recs, 1):
        marker = "   <- recommended" if i == 1 else ""
        print(f"  [{i}] {m.line()}{marker}")
    custom_index = len(recs) + 1
    print(f"  [{custom_index}] Custom (enter a Hugging Face MLX model id)")
    print()

    raw = input("Choose [1]: ").strip() or "1"
    try:
        choice = int(raw)
    except ValueError:
        print(f"  '{raw}' is not a number.")
        return 1

    if choice == custom_index:
        hf_id = input("  Hugging Face id (e.g. mlx-community/Some-Model-4bit): ").strip()
        if not hf_id:
            print("  No id given. Aborting.")
            return 1
        if "405b" in hf_id.lower() and not _confirm(
                "  That looks like a 405B model (~220 GB). Almost no Mac can run that. Continue?",
                default=False):
            return 1
        chosen = Model(short_name=hf_id.split("/")[-1], hf_id=hf_id, load_gb=0.0,
                       tier="custom", blurb="user-specified")
    elif 1 <= choice <= len(recs):
        chosen = recs[choice - 1]
    else:
        print(f"  {choice} is not a valid option.")
        return 1

    print(f"\nSelected: {chosen.short_name}")
    if chosen.load_gb and not fits(chosen, hw.ai_available_gb) and not _confirm(
            f"  {chosen.short_name} needs ~{chosen.load_gb} GB but you only have "
            f"~{hw.ai_available_gb} GB available for AI. Continue anyway?", default=False):
        return 1

    _download_model(chosen.hf_id)
    _write_config(chosen)
    print("\nDone. Start Cloxy with:  cloxy start")
    print("The LLM loads on first request, or at startup if CLOXY_EAGER_LLM=1.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    cfg = _read_config()
    if not cfg:
        print("No LLM configured yet. Run `cloxy init` to set one up.")
        return 0
    print(f"Configured model: {cfg.get('model_short_name')} ({cfg.get('model_hf_id')})")
    print(f"Config file: {config.CONFIG_PATH}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from .catalog import CATALOG
    print("Cloxy MLX model catalog:\n")
    for m in CATALOG:
        print("  " + m.line())
    return 0


# =============================================================================
# Run things
# =============================================================================

def cmd_start(args: argparse.Namespace) -> int:
    if args.host:
        config.HOST = args.host
    if args.port:
        config.PORT = args.port
    from .server import serve
    serve()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import run
    run()
    return 0


# =============================================================================
# Talk to a running server
# =============================================================================

def _http():
    import httpx
    headers = {"X-API-Key": config.API_KEY} if config.API_KEY else {}
    return httpx.Client(base_url=config.URL, timeout=600, headers=headers)


def _not_running() -> int:
    print(f"Cloxy server is not reachable at {config.URL}.")
    print("Start it with `cloxy start`, or `cloxy install-service` to keep it running.")
    return 2


def cmd_status(args: argparse.Namespace) -> int:
    import httpx
    try:
        with _http() as c:
            h = c.get("/health").json()
            s = c.get("/memory_stats").json()
            w = c.get("/ingest_status").json()
    except httpx.ConnectError:
        return _not_running()
    print(f"cloxy v{h['version']} at {config.URL} — up {h['uptime']}s — auth {h['auth']}")
    print(f"memories: {s['total_memories']} (vector {s['vector_index_size']}, "
          f"keyword {s.get('keyword_index_size')}) — embed {s['embed_model']}")
    for src, n in sorted(s["sources"].items(), key=lambda kv: -kv[1]):
        print(f"  {src:<12} {n}")
    state = "running" if w["running"] else "off"
    print(f"watcher: {state}, {w.get('files_tracked', 0)} files tracked, {w['scans']} scans, "
          f"last {w['last_scan_at'] or 'never'}")
    if w.get("scanning"):
        print(f"  scanning now: {w['files_this_scan']} files, +{w['chunks_this_scan']} chunks "
              f"so far (current: {w.get('current_file')})")
    if w.get("last_error"):
        print(f"  last error: {w['last_error']}")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    import httpx
    payload = {"query": " ".join(args.query), "top_k": args.top_k, "mode": args.mode,
               "project": args.project, "since": args.since, "until": args.until}
    try:
        with _http() as c:
            r = c.post("/recall", json=payload)
    except httpx.ConnectError:
        return _not_running()
    body = r.json()
    if r.status_code >= 400:
        print(body.get("error") or body)
        return 1
    if args.json:
        print(json.dumps(body, indent=2))
        return 0
    hits = body["results"]
    if not hits:
        print("No matches.")
        return 0
    for i, h in enumerate(hits, 1):
        when = (h.get("ts_start") or "")[:16].replace("T", " ")
        proj = os.path.basename((h.get("project") or "").rstrip("/"))
        title = (h.get("metadata") or {}).get("title")
        tag = " · ".join(b for b in (when, proj, title) if b) or h.get("source")
        print(f"\n#{i} [{tag}]  score {h['score']}  id {h['id']}")
        text = h["content"]
        print(text if args.full or len(text) <= 600 else text[:600] + " …")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    import httpx
    payload = {"convo_dir": args.dir, "force": args.force}
    try:
        with _http() as c:
            r = c.post("/ingest_convos", json=payload)
    except httpx.ConnectError:
        return _not_running()
    body = r.json()
    if r.status_code >= 400:
        print(body.get("error") or body)
        return 1
    print(f"files: {body['files_found']} found, {body['files_processed']} processed, "
          f"{body['files_skipped']} unchanged")
    print(f"chunks: {body['chunks_stored']} stored, {body['chunks_duplicate']} duplicate")
    for e in body.get("errors", []):
        print(f"  error {e['file']}: {e['error']}")
    return 0


# =============================================================================
# launchd service (macOS)
# =============================================================================

def _launchctl(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def cmd_install_service(args: argparse.Namespace) -> int:
    if sys.platform != "darwin":
        print("install-service uses launchd and is macOS only. On Linux, run `cloxy start` "
              "under systemd — see the README.")
        return 1
    log_dir = Path(config.DATA_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k.startswith("CLOXY_")}
    env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")
    plist = {
        "Label": PLIST_LABEL,
        "ProgramArguments": [sys.executable, "-m", "cloxy", "start"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "WorkingDirectory": str(Path.home()),
        "StandardOutPath": str(log_dir / "cloxy.log"),
        "StandardErrorPath": str(log_dir / "cloxy.log"),
        "EnvironmentVariables": env,
    }
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    if PLIST_PATH.exists():
        _launchctl("bootout", domain, str(PLIST_PATH))
    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)
    res = _launchctl("bootstrap", domain, str(PLIST_PATH))
    if res.returncode != 0:
        print(f"launchctl bootstrap failed: {res.stderr.strip() or res.stdout.strip()}")
        return 1
    print(f"Installed {PLIST_PATH}")
    print(f"Cloxy will start at login and restart if it exits. Logs: {log_dir / 'cloxy.log'}")
    print(f"Server: {config.URL}   Check with: cloxy status")
    return 0


def cmd_uninstall_service(args: argparse.Namespace) -> int:
    if not PLIST_PATH.exists():
        print("No launchd service installed.")
        return 0
    _launchctl("bootout", f"gui/{os.getuid()}", str(PLIST_PATH))
    PLIST_PATH.unlink()
    print(f"Removed {PLIST_PATH}")
    return 0


# =============================================================================
# Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloxy", description="Local AI with eyes and memory.")
    parser.add_argument("--version", action="version", version=f"cloxy {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Detect hardware, pick + download an LLM").set_defaults(func=cmd_init)
    sub.add_parser("show", help="Show current LLM config").set_defaults(func=cmd_show)
    sub.add_parser("list", help="List the MLX model catalog").set_defaults(func=cmd_list)

    p = sub.add_parser("start", help="Run the server")
    p.add_argument("--host", help=f"bind address (default {config.HOST})")
    p.add_argument("--port", type=int, help=f"port (default {config.PORT})")
    p.set_defaults(func=cmd_start)

    sub.add_parser("mcp", help="Run the MCP stdio server").set_defaults(func=cmd_mcp)
    sub.add_parser("status", help="Show server, memory and watcher status").set_defaults(func=cmd_status)

    p = sub.add_parser("recall", help="Search memory")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    p.add_argument("--project", help="substring of the project directory")
    p.add_argument("--since", help="ISO date lower bound, e.g. 2026-09-01")
    p.add_argument("--until", help="ISO date upper bound")
    p.add_argument("--full", action="store_true", help="print full chunks")
    p.add_argument("--json", action="store_true", help="raw JSON output")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("ingest", help="Run an ingest pass now")
    p.add_argument("dir", nargs="?", help="directory of .jsonl sessions (default: watched dirs)")
    p.add_argument("--force", action="store_true", help="re-ingest from scratch")
    p.set_defaults(func=cmd_ingest)

    sub.add_parser("install-service", help="Run cloxy at login via launchd (macOS)") \
        .set_defaults(func=cmd_install_service)
    sub.add_parser("uninstall-service", help="Remove the launchd service") \
        .set_defaults(func=cmd_uninstall_service)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
