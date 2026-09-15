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

The throughput is **worse**, and at nine passes per path that is a measurement
rather than a shrug:

    path      decode t/s   GB/token   change   permutation test
    grouped         6.86      0.846        -   -
    pf-all          5.92      1.045     -14%   real, p=0.010
    pf-k4           6.36      0.872      -7%   not distinguishable, p=0.09

An earlier five-pass run showed the opposite — k=4 peaking at 6.50 against 5.97
— and it was the *baseline* that was noisy, not the treatment. Nine passes put
grouped at 6.86 with a 5.77-6.90 spread. Prefetch never beat it at any budget.

WHY, AND IT IS THE USEFUL PART
------------------------------
Multiply each path's throughput by its bytes per token: 5.80, 6.19, 5.55 GB/s.
Throughput varies 8% across those paths and bytes vary 11%, but the product is
flat. That is a saturated link, and on a saturated link

    decode tok/s = bandwidth / bytes-per-token

Prefetch only ever raises the denominator. It did everything it promised — 80
ms/token of blocking transfer off the critical path (116.1 -> 36.0 ms), decode
hit rate 47.5% -> 82.4% — and lost anyway, because the side stream contends for
the same PCIe link and that link was already the constraint.

So the conclusion is not "prefetch is slow". It is that **no reordering of
transfers can help at this operating point**; only moving fewer bytes can.
`ff-serve` prints a sustained-GB/s column so the next person sees this without
doing the arithmetic. The levers that remain are pinning the rest of the store
(5.8 GB/s sustained against Q7's 10.4 GB/s pinned, with 9 of 16 layers done),
Q7's CPU-side path, Q5's 23.4 points of Belady headroom, and quantisation.

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
