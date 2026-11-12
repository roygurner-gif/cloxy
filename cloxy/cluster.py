"""
MLX distributed cluster helpers for `cloxy cluster ...`.

Cloxy doesn't shard models itself. It drives Apple's own tooling:

  setup  → mlx.distributed_config   discovers the Thunderbolt (or Ethernet)
                                      links between the Macs and writes a
                                      hostfile (RDMA devices for JACCL)
  serve  → mlx.launch -- -m mlx_lm server
                                      one process per Mac; pipeline parallel on
                                      ring, tensor parallel on jaccl, rank 0
                                      (the first host) answers HTTP

Then run cloxy with CLOXY_LLM_URL pointing at rank 0, and memory-augmented
/v1/chat/completions is served by the whole cluster.

Backends (mlx.launch --backend):
  jaccl       RDMA over Thunderbolt 5, every pair of Macs cabled (full mesh);
              needs macOS 26.2+ with RDMA enabled. Lowest latency.
  jaccl-ring  RDMA over Thunderbolt 5 in a ring (each Mac cabled to two
              neighbours) — for more Macs than ports.
  ring        TCP over any network. Always works, slowest.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

BACKENDS = ("jaccl", "jaccl-ring", "ring")


def _tool(name: str) -> List[str]:
    """Console script next to this interpreter, else on PATH, else `python -m`."""
    local = Path(sys.executable).parent / name
    if local.is_file():
        return [str(local)]
    found = shutil.which(name)
    if found:
        return [found]
    module = {"mlx.launch": "mlx._distributed_utils.launch",
              "mlx.distributed_config": "mlx._distributed_utils.config"}[name]
    return [sys.executable, "-m", module]


def setup_argv(hosts: str, hostfile: str, over: str = "thunderbolt",
               backend: Optional[str] = None, auto_setup: bool = False,
               verbose: bool = False) -> List[str]:
    argv = _tool("mlx.distributed_config") + [
        "--hosts", hosts, "--over", over, "--output-hostfile", hostfile]
    if backend and over == "thunderbolt":
        argv += ["--backend", backend]
    if auto_setup:
        argv.append("--auto-setup")
    if verbose:
        argv.append("--verbose")
    return argv


def serve_argv(model: str, hostfile: str, port: int, backend: Optional[str] = None,
               host: str = "0.0.0.0", pipeline: bool = False, python: Optional[str] = None,
               extra: Optional[List[str]] = None, verbose: bool = False) -> List[str]:
    """
    mlx.launch --python <py> … -- -m mlx_lm server …  mlx.launch runs the
    command with that interpreter on every Mac, so it must not repeat it. The
    remote Macs need the same python path with mlx-lm installed (pass
    `python` if theirs differs).
    """
    py = python or sys.executable
    argv = _tool("mlx.launch") + ["--hostfile", hostfile, "--python", py]
    if backend:
        argv += ["--backend", backend]
    if verbose:
        argv.append("--verbose")
    argv += ["--", "-m", "mlx_lm", "server", "--model", model,
             "--host", host, "--port", str(port)]
    if pipeline:
        argv.append("--pipeline")
    extra = list(extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    return argv + extra


def use_pipeline(backend: Optional[str], pipeline: bool = False, tensor: bool = False) -> bool:
    """
    Tensor parallel syncs every layer of every token, so it only pays off on
    JACCL's microsecond RDMA. Over TCP (ring) each sync is a network round trip:
    measured on two M4 minis over Wi-Fi, tensor ran 0.85 tok/s and pipeline
    27-36 tok/s on the same model. Default: pipeline unless the backend is jaccl*.
    """
    if pipeline or tensor:
        return pipeline
    return not (backend or "").startswith("jaccl")


def read_hostfile(path: str) -> dict:
    """{"backend": str, "hosts": [{"ssh", "ips", "rdma"}]}; a bare list is accepted too."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = {"backend": "", "hosts": data}
    return data


def describe(hostfile: dict) -> List[str]:
    """Human-readable lines: one per rank, plus whether RDMA is wired for JACCL."""
    hosts = hostfile.get("hosts") or []
    lines = [f"backend: {hostfile.get('backend') or '(set at launch)'}   ranks: {len(hosts)}"]
    for rank, h in enumerate(hosts):
        rdma = [d for d in (h.get("rdma") or []) if d]
        role = "rank 0 — serves HTTP" if rank == 0 else f"rank {rank}"
        lines.append(f"  {h.get('ssh', '?'):<24} {', '.join(h.get('ips') or []) or '-':<18} "
                     f"rdma {len(rdma)}  ({role})")
    return lines


def rank0_url(hostfile: dict, port: int) -> Optional[str]:
    """
    Where mlx_lm.server listens, as seen from the LAN: rank 0's ssh name. (The
    hostfile IPs are often Thunderbolt-bridge addresses only the cluster sees.)
    If cloxy runs on rank 0 itself, http://127.0.0.1:<port> works too.
    """
    hosts = hostfile.get("hosts") or []
    if not hosts:
        return None
    addr = hosts[0].get("ssh") or (hosts[0].get("ips") or [None])[0]
    if addr and "@" in addr:
        addr = addr.split("@", 1)[1]
    return f"http://{addr}:{port}" if addr else None
