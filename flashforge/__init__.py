"""flashforge — MoE expert-caching and prefetch research.

Stage 0: instrument a small fine-grained MoE, collect routing traces, and
measure the properties that decide whether the later stages are worth
building — skew, locality, cross-layer predictability, domain clustering and
expert-set expansion — plus a cache simulation with Belady as the ceiling.

Alongside those, `hardware` measures the machine rather than the model: the
CPU/GPU/PCIe cost model the placement policy resolves against, and the storage
read curve that says how much lead time a disk tier would need.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "analysis",
    "cachesim",
    "hardware",
    "models",
    "plots",
    "prompts",
    "tracing",
    "viz",
]
