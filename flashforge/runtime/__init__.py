"""Stage 1 — the offloaded MoE runtime.

Stage 0 measured the routing behaviour and the machine. This package spends
those measurements: expert weights live in CPU RAM, a fixed pool of GPU slots
caches the hot ones, and the MoE block pulls what it needs through that cache.

The Stage 0 constants this is built against, all measured on an RTX 2060 6GB
with OLMoE-1B-7B (see the README's "Measured results"):

- LRU beats LFU by 12 points at 25% capacity, so the cache is LRU (Q5).
- Belady reaches 77.9% against LRU's 54.5%, so roughly a quarter of the misses
  are addressable by a better policy — but the larger prize is hiding the
  remaining misses behind compute rather than eliminating them (Q5).
- One expert is 12.58 MB and crosses PCIe in 1.21 ms pinned (Q7).
- A layer of decode compute is 2.07 ms (Q7/Q8 calibration).

WHAT THE MEASUREMENT ACTUALLY SAID
---------------------------------
Those constants predict a bandwidth-bound system. A token touches 16 layers x
top-8 = 128 experts; at LRU's 54.5% hit rate that is ~58 misses, ~70 ms of PCIe
traffic against ~33 ms of compute to hide it behind. An earlier version of this
docstring said so, confidently, before `ff-serve` existed.

It is wrong, and the sweep says so plainly. Going from 128 to 384 slots cuts
transfer volume by **9x** (1.080 -> 0.120 GB/token) and does not reliably move
decode speed at all: the medians over five passes are 3.84, 4.41 and 3.53
tok/s, while a single capacity varies by 26% between its own passes. At 384
slots the remaining 0.120 GB/token is, at the measured 10.4 GB/s, ~11 ms of a
~230 ms token. A profiler agrees: `aten::copy_` is 4.95% of decode.

The other 95% is dispatch. Decode issues **465 separate GEMMs per token** —
16 layers x (8 experts x 3 projections + gate) + attention — each one
multiplying a *single* token's activations. Self CPU time and self CUDA time
come out within 7% of each other, which is the signature of a launch-bound
workload: neither side is doing arithmetic, both are doing bookkeeping.

So the ordering in the roadmap was backwards. Prefetching perfectly would buy
at most 5% here, because transfers are already nearly free at a workable cache
size. The lever is batching the per-expert GEMMs into one grouped call, which
was filed as a Stage 2 kernel concern and is really the Stage 1 bottleneck.

STAGE 1B — SPENDING THAT FINDING
--------------------------------
`CachedMoEBlock` therefore grew a second execution path. It sorts the (token,
expert) pairs by expert, pads each expert's group to a common width, gathers
the routed experts' weights into one batched tensor and runs three `bmm` calls
for the whole layer instead of 3E separate GEMMs.

Timed against the loop path on the same loaded model, the same slot pool and
the same warm cache — one variable — over nine passes at 256 slots:

    phase     loop    grouped   change
    prefill   86.1       98.7    +15%   inside the 25% within-path spread
    decode     4.61       5.93   +29%   real

Decode is the win, and it reproduced at +39%, +35% and +29% across three runs.
Prefill is not a result in either direction: an early run appeared to show a
36% *regression* and the next reversed it, which is what a 68.9-101.6 tok/s
spread does to a median reported without one.

The grouped path is **not** bit-exact — one `index_add_` over every expert at
once has no defined accumulation order — so it is on by default only because
the divergence was measured to sit below the argmax: fp16, the real 7B model,
65 greedy tokens, identical ids. The loop path stays as the oracle the parity
tests hold to a difference of exactly zero.

Against accelerate's `device_map="auto"` on the same prompt, timed by the same
harness, in the **same `ff-serve --baseline --path both` invocation** — the
baseline re-measured there at 0.40 tok/s (0.39-0.40 over five passes) rather
than being carried over from the Stage 1 run — decode goes 0.40 -> 4.69 on the
loop path and 0.40 -> 5.98 grouped. That is **15.0x**, on a model 2.3x larger
than the card.
"""

from __future__ import annotations

from flashforge.runtime.cache import CacheStats, ExpertCache
from flashforge.runtime.store import ExpertStore
from flashforge.runtime.block import CachedMoEBlock
from flashforge.runtime.patch import PatchReport, install_expert_cache

__all__ = [
    "CacheStats",
    "CachedMoEBlock",
    "ExpertCache",
    "ExpertStore",
    "PatchReport",
    "install_expert_cache",
]
