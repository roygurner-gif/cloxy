"""
Model catalog for Cloxy — MLX-Community quantized models.

Each entry is a real model on Hugging Face that mlx-lm can load directly.
`load_gb` is the conservative memory footprint after load (weights + KV cache
for a reasonable context window). Recommendations leave headroom inside that
already-conservative number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Model:
    short_name: str
    hf_id: str
    load_gb: float
    tier: str
    blurb: str

    def line(self) -> str:
        return f"{self.short_name:<22} {self.load_gb:>5.1f} GB   {self.tier:<8}  {self.blurb}"


# Curated MLX-Community models. Keep this list short and trusted.
# Add more later as they prove out.
CATALOG: List[Model] = [
    Model("Phi-3 mini 4B",
          "mlx-community/Phi-3-mini-4k-instruct-4bit",
          2.5, "tiny",
          "Fastest, lowest RAM. Good for chat, weak at reasoning."),
    Model("Qwen 2.5 7B",
          "mlx-community/Qwen2.5-7B-Instruct-4bit",
          5.0, "small",
          "Strong all-rounder for its size."),
    Model("Llama 3.1 8B",
          "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
          5.5, "small",
          "Meta's workhorse. Broad knowledge."),
    Model("Qwen 2.5 14B",
          "mlx-community/Qwen2.5-14B-Instruct-4bit",
          9.0, "medium",
          "Sweet spot for 32 GB Macs."),
    Model("Qwen 2.5 32B",
          "mlx-community/Qwen2.5-32B-Instruct-4bit",
          19.0, "large",
          "Excellent reasoning. SWARM/OMNISCIENT default."),
    Model("Llama 3.1 70B",
          "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
          40.0, "xlarge",
          "Top-tier capability. Needs a Studio or M4 Max."),
    Model("Qwen 2.5 72B",
          "mlx-community/Qwen2.5-72B-Instruct-4bit",
          41.0, "xlarge",
          "Very strong, especially on code + math."),
    Model("Llama 3.1 405B",
          "mlx-community/Meta-Llama-3.1-405B-Instruct-4bit",
          220.0, "absurd",
          "Yes, this exists. Yes, your Mac probably can't run it."),
]


def fits(model: Model, available_gb: float, context_headroom: float = 1.2) -> bool:
    """
    A model 'fits' if its load_gb plus a context-window buffer is under the
    AI-available memory budget. The 1.2x default leaves room for long contexts
    and the system to breathe.
    """
    return model.load_gb * context_headroom <= available_gb


def recommend(available_gb: float, max_results: int = 5) -> List[Model]:
    """Return the largest-fitting models first, capped at max_results."""
    fitting = [m for m in CATALOG if fits(m, available_gb)]
    fitting.sort(key=lambda m: m.load_gb, reverse=True)
    return fitting[:max_results]


def by_short_name(name: str) -> Model:
    name_l = name.strip().lower()
    for m in CATALOG:
        if m.short_name.lower() == name_l or m.hf_id.lower() == name_l:
            return m
    raise KeyError(f"No model matching '{name}'. Run `cloxy init` to see options.")


if __name__ == "__main__":
    # Quick demo: pretend we have 64 GB
    print("Demo: 64 GB unified memory, ~46 GB available for AI\n")
    for m in recommend(available_gb=46.0):
        print("  " + m.line())
