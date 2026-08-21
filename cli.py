"""
Cloxy CLI — `cloxy init` wizard.

Detects Apple Silicon hardware, recommends MLX-Community models that fit
in unified memory, downloads the user's choice, and persists the selection
so `cloxy start` can launch the FastAPI app with the LLM already wired in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hardware import detect, NotAppleSiliconError
from models import CATALOG, Model, fits, recommend


CONFIG_PATH = Path(os.environ.get("CLOXY_CONFIG", Path.home() / ".cloxy" / "config.json"))


def _write_config(model: Model) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "model_short_name": model.short_name,
        "model_hf_id": model.hf_id,
        "model_load_gb": model.load_gb,
    }, indent=2))
    print(f"\n  Config saved to {CONFIG_PATH}")


def _read_config() -> dict | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _download_model(hf_id: str) -> None:
    """Stream the model files from Hugging Face into MLX's local cache."""
    print(f"\n  Downloading {hf_id}")
    print("  (this can take a while on first run; subsequent loads are instant)")
    try:
        from mlx_lm import load
    except ImportError:
        print("\n  ERROR: mlx-lm is not installed.")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)

    # load() pulls weights and caches them; we discard the returned model
    # because we just want it on disk.
    load(hf_id)
    print("  Download complete.")


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix} ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def cmd_init(args: argparse.Namespace) -> int:
    print("Cloxy init — detecting your hardware...\n")
    try:
        hw = detect()
    except NotAppleSiliconError as e:
        print(f"  {e}")
        return 1

    print(hw.pretty())
    print()

    recs = recommend(hw.ai_available_gb)
    if not recs:
        print("  Your machine is too small for any catalog model (need ≥3 GB).")
        print("  You can still run Cloxy without an LLM for the proxy/RAG features.")
        return 1

    print("Recommended models for your hardware (largest fitting first):\n")
    for i, m in enumerate(recs, 1):
        marker = "   <- recommended" if i == 1 else ""
        print(f"  [{i}] {m.line()}{marker}")
    custom_index = len(recs) + 1
    print(f"  [{custom_index}] Custom (enter a Hugging Face MLX model id)")
    print()

    raw = input(f"Choose [1]: ").strip() or "1"
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
        # Build an ad-hoc Model entry. We can't know load_gb up front; assume
        # the user knows their hardware (with a polite warning).
        if "405b" in hf_id.lower():
            if not _confirm(
                "  That looks like a 405B model (~220 GB). "
                "Almost no Mac can actually run that. Continue anyway?",
                default=False,
            ):
                return 1
        chosen = Model(
            short_name=hf_id.split("/")[-1],
            hf_id=hf_id,
            load_gb=0.0,
            tier="custom",
            blurb="user-specified",
        )
    elif 1 <= choice <= len(recs):
        chosen = recs[choice - 1]
    else:
        print(f"  {choice} is not a valid option.")
        return 1

    print(f"\nSelected: {chosen.short_name}")
    if chosen.load_gb and not fits(chosen, hw.ai_available_gb):
        if not _confirm(
            f"  {chosen.short_name} needs ~{chosen.load_gb} GB but you only have "
            f"~{hw.ai_available_gb} GB available for AI. Continue anyway?",
            default=False,
        ):
            return 1

    _download_model(chosen.hf_id)
    _write_config(chosen)

    print("\nDone. Start Cloxy with:  python cloxy.py")
    print("The LLM will be loaded on first request, or eagerly if CLOXY_EAGER_LLM=1.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    cfg = _read_config()
    if not cfg:
        print("No Cloxy config yet. Run `cloxy init` to set one up.")
        return 0
    print(f"Configured model: {cfg.get('model_short_name')} ({cfg.get('model_hf_id')})")
    print(f"Config file: {CONFIG_PATH}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("Cloxy MLX model catalog:\n")
    for m in CATALOG:
        print("  " + m.line())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cloxy", description="Cloxy for Apple Silicon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Detect hardware, pick + download an LLM")
    p_init.set_defaults(func=cmd_init)

    p_show = sub.add_parser("show", help="Show current Cloxy config")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List the MLX model catalog")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
