"""flashforge — MoE expert-caching and prefetch research.

Stage 0: instrument a small fine-grained MoE, collect routing traces, and
measure the four properties that decide whether the later stages are worth
building — skew, locality, cross-layer predictability, and domain clustering —
plus a cache simulation with Belady as the ceiling.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "analysis",
    "cachesim",
    "models",
    "plots",
    "prompts",
    "tracing",
    "viz",
]
