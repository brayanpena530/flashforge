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

STAGE 1C — THE BUDGET THAT WAS NEVER 5%
---------------------------------------
The paragraphs above deprioritised prefetch on the strength of one number:
`aten::copy_` at 4.95% of decode. That number was profiled at **384 slots**,
and the runtime ships at 256. Measured at the operating point instead:

    slots   fill ms/token   % of decode   perfect-prefetch ceiling
      128          216.7         77.0%        +335%
      256          126.3         70.9%        +244%
      320          123.0         22.4%         +29%

So transfers were never nearly free; they are most of decode. And the 384-slot
row the original profile came from was not a valid configuration at all: 5.39 GB
resident on a 6 GB card, which on Windows does not OOM — WDDM backs the overflow
with host memory. It measured 0.83 tok/s at a 92.7% hit rate against 5.53 tok/s
at 46.9% with 256 slots. Hit rate doubled, throughput fell 6.7x. `ff-serve` now
prints VRAM headroom per row so that cannot be mistaken for a cache result
again.

WHAT THE PREFETCHER DID, AND WHAT IT DID NOT
--------------------------------------------
`CachedMoEBlock` runs the *next* layer's router on *this* layer's hidden state
(Q3's `stale_router`, 0.835 recall at one layer) and issues the predicted fills
on a side CUDA stream, so they overlap this layer's GEMMs. Measured at 256
slots with 6.75 GB of the store pinned:

    path        decode t/s   blocking fill    decode hit   GB/token
    loop              4.98   130.4 ms (50%)        46.8%      0.858
    grouped           5.85   100.5 ms (65%)        46.7%      0.859
    prefetch          6.08    32.7 ms (20%)        81.8%      1.069

The mechanism works and reproduces: hit rate 46.8 -> 81.8%, blocking fill down
by two thirds, predictor precision 78.2% *in the runtime* (Q3's 0.835 was
recall, offline, on a trace).

The throughput win is **not established**. Measured against grouped three
times, prefetch came in at **-15%, +4% and +29%** — and every one of those sat
inside a within-path spread of 18-53%. Three runs that disagree on the sign are
not three noisy estimates of a real effect; they are the harness saying it
cannot resolve this. Sixty-eight ms per token left the critical path and did
not come back as tokens.

(The -15% is the unpinned run, and belongs to a different configuration: with a
pageable store `copy_(non_blocking=True)` is synchronous, so the side stream
could not overlap at all and the extra bytes were pure loss. Pinning is a
prerequisite for prefetch, not an optimisation alongside it.)

The live hypothesis is bandwidth, not latency. Prefetch moves 24% more bytes
(0.859 -> 1.069 GB/token) because 22% of its guesses are wrong, and it spends
them on a link that was already the constraint. Removing a *stall* does not help
when the *link* is what is saturated. That predicts a specific fix — spend the
speculation budget only on the highest-weighted predictions rather than all
top_k — and that is where Stage 1c resumes.

One thing the prefetcher is not allowed to be is approximate. It changes when
weights arrive, never which ones, so `ff-serve` treats any greedy divergence as
a cross-stream race rather than rounding. It has been identical for 33 tokens
in every run.

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
