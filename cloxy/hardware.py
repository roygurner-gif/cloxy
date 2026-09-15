"""
Hardware detection for Cloxy — Apple Silicon only.

Detects M-series chip, unified memory, and estimates how much RAM
is realistically available for an LLM (reserves headroom for OS + user apps).
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


class NotAppleSiliconError(RuntimeError):
    """Raised when Cloxy is run on a non-Apple-Silicon machine."""


@dataclass(frozen=True)
class Hardware:
    chip: str
    total_ram_gb: float
    ai_available_gb: float
    macos_version: str

    def pretty(self) -> str:
        return (
            f"Chip: {self.chip}\n"
            f"Total RAM: {self.total_ram_gb:.0f} GB unified memory\n"
            f"Available for AI: ~{self.ai_available_gb:.0f} GB (after OS reserve)\n"
            f"macOS: {self.macos_version}"
        )


def _sysctl(key: str) -> str:
    return subprocess.check_output(["sysctl", "-n", key], text=True).strip()


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _reserve_headroom(total_gb: float) -> float:
    """
    Reserve a chunk of unified memory for the OS, user apps, and the LLM's
    KV cache / context window. Bigger machines can afford a smaller % reserve
    because the absolute number of GB still leaves plenty for the OS.
    """
    if total_gb <= 16:
        return total_gb * 0.55       # tight — leave ~45% for OS+apps
    if total_gb <= 32:
        return total_gb * 0.65
    if total_gb <= 64:
        return total_gb * 0.72
    return total_gb * 0.78            # 96GB / 128GB / 192GB Studios


def detect() -> Hardware:
    if not _is_apple_silicon():
        raise NotAppleSiliconError(
            "Cloxy is currently Apple Silicon only. "
            "Detected platform: "
            f"{platform.system()} {platform.machine()}"
        )

    chip = _sysctl("machdep.cpu.brand_string")
    total_bytes = int(_sysctl("hw.memsize"))
    total_gb = total_bytes / (1024 ** 3)
    macos_version = platform.mac_ver()[0] or "unknown"

    return Hardware(
        chip=chip,
        total_ram_gb=round(total_gb, 1),
        ai_available_gb=round(_reserve_headroom(total_gb), 1),
        macos_version=macos_version,
    )


if __name__ == "__main__":
    print(detect().pretty())
