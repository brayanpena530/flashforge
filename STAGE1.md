# Stage 1 — what was built, what it measured

A recap of Stage 1, 1b and 1c. Every number here names the run it came from,
because several of them contradict numbers from other runs and the difference
is usually the machine, not the code.

---

## The problem

OLMoE-1B-7B is **13.8 GB** in fp16. The card is an **RTX 2060, 6 GB**.

```
model   ████████████████████████████████████████████  13.8 GB
card    ███████████████████                            6.0 GB
```

But 6.44B of the model's 6.9B parameters are **experts**, and a decode token
only touches `top_k = 8` of the 64 in each layer. So almost all of that 13.8 GB
is idle at any instant.

```
BEFORE                          AFTER (Stage 1)
┌──────────────────────┐        ┌──────────────────────┐
│ GPU 6 GB             │        │ GPU 6 GB             │
│  ✗ doesn't fit       │        │  ├─ model    0.9 GB  │  non-expert weights
└──────────────────────┘        │  ├─ cache    3.0 GB  │  256 expert slots, LRU
                                │  └─ free     0.5 GB  │  activations
   accelerate spills            └──────────────────────┘
   to disk: 0.40 tok/s          ┌──────────────────────┐
                                │ HOST RAM  12.0 GB    │  all 1,024 experts
                                └──────────────────────┘
```

---

## Stage 1 — the offloaded runtime

**Built:** `ExpertStore` (experts in host RAM, one contiguous row each),
`ExpertCache` (preallocated GPU slot pool, LRU eviction), `CachedMoEBlock`
(drop-in for `OlmoeSparseMoeBlock`), `install_expert_cache` (model surgery).

**Result:** the model runs on a card 2.3x too small for it.

**The finding that reordered the roadmap.** The design predicted a
bandwidth-bound system. Profiling said otherwise:

```
decode time, 384 slots     transfer  ▎ 4.95%
                           dispatch  ████████████████████ 95%
```

465 separate GEMMs per token, each multiplying a *single* token's activations.
Self CPU within 7% of self CUDA — the signature of a launch-bound workload. So
grouped GEMM, filed as Stage 2, was actually the Stage 1 bottleneck.

---

## Stage 1b — grouped GEMM

**Built:** a second execution path. Sort the (token, expert) pairs by expert,
pad each expert's group to a common width, gather the routed weights into one
batched tensor, run **three `bmm` calls per layer** instead of 3×E separate
GEMMs.

```
loop path      expert 0: [gate][up][down]   ← 3 kernel launches
               expert 1: [gate][up][down]      × 8 experts
               ...                             × 16 layers = 465/token
               expert 7: [gate][up][down]

grouped path   all 8:    [═══bmm═══][═══bmm═══][═══bmm═══]   3/layer
```

**Result** — one `ff-serve --baseline --path both` invocation, 5 passes:

```
decode tok/s
baseline   0.40  ██▌
loop       4.69  ██████████████████████████████
grouped    5.98  ██████████████████████████████████████    +27%  ← 15.0x baseline
```

Prefill: **no finding** (+8% inside a 34% spread).

**Trade-off, measured not assumed.** The grouped path is *not* bit-exact — one
`index_add_` over all experts has no defined accumulation order. A `2e-19`
tolerance in fp32 on a toy block says nothing about fp16 on a 7B model, where
one rounding difference flips an argmax. So `ff-serve` decodes greedily on both
paths and compares token ids: **identical for 65 tokens**. That is why it ships
on by default, and why the loop path stays as the bit-exact oracle.

---

## Stage 1c — the prefetch budget

The roadmap said prefetch was "worth at most ~5%", on the strength of that
4.95% figure. **Both were wrong, for the same reason: the profile was taken at
384 slots and the runtime ships at 256.**

```
fill time as a share of decode, measured per capacity

128 slots  ████████████████████████████████████████  77.0%   (216.7 ms/token)
256 slots  ████████████████████████████████████      70.9%   (126.3 ms/token)
320 slots  ███████████                               22.4%   (123.0 ms/token)
384 slots  ▎                                          4.95%  ← INVALID, see below
```

Transfers were never nearly free. They are most of decode.

### The 384-slot row was never a valid measurement

It puts **5.39 GB resident on a 6 GB card**. On Windows that does not raise —
WDDM silently backs the overflow with host memory, so every read of a
"resident" expert crosses PCIe *again*.

```
              hit rate        decode
256 slots     46.9%  ███      5.53 tok/s  ████████████████████
384 slots     92.7%  ██████   0.83 tok/s  ███
              ▲ up 2x                     ▲ down 6.7x
```

Hit rate up, throughput down. That is the signature of a resource bug, one tier
below the host-RAM version already in `troubleshoot.md`. `ff-serve` now prints
VRAM headroom per row.

### The prefetcher

Run the **next** layer's router on **this** layer's hidden state (Q3's
`stale_router`, 0.835 recall), and issue the predicted fills on a side CUDA
stream so they overlap this layer's GEMMs.

```
before   layer N   [fill ██████][GEMM ███]
         layer N+1              [fill ██████][GEMM ███]

after    layer N   [GEMM ███]
                   [prefetch N+1 ██████]        ← side stream, overlapped
         layer N+1 [GEMM ███]
```

Two write hazards, two mechanisms — and missing either produces a model that is
correct on most tokens and quietly wrong on the rest:

| hazard | covered by |
|---|---|
| side stream overwrites slots *earlier* layers hold | an event recorded on the compute stream at issue time |
| side stream overwrites slots *this* layer holds | a protected set — this layer's GEMMs aren't enqueued yet, so no event covers them |

**The mechanism works.** Blocking fill 101.6 → 37.3 ms/token. Decode hit rate
47.4% → 82.3%. Predictor precision 78.7% measured in the runtime.

**The throughput didn't follow.** And that turned out to be the interesting part.

### Why: it's bandwidth, not latency

Prefetch buys overlap by moving **more bytes** — 22% of its guesses are wrong
and it pays full freight for them. Overlapping a transfer helps when *ordering*
is the problem. It does not help when the *link* is saturated; it makes the link
worse.

Testable prediction: truncate the speculation to the router's most confident
predictions. Bandwidth should fall, precision should *rise* (the dropped guesses
are the wrong ones), and throughput should peak somewhere in the middle.

**5 passes** showed an inverted-U: k=4 peaked at 6.50 against 5.97 for no
prefetch. That looked like the prediction confirming itself.

**9 passes killed it.** The no-prefetch baseline had been the noisy one — it
read 5.97 on five passes and 6.86 on nine, with a 5.77–6.90 spread:

| path | precision | GB/token | dec hit | blocking fill | decode tok/s | verdict |
|---|---|---|---|---|---|---|
| grouped (no prefetch) | — | 0.846 | 47.5% | 116.1 ms | **6.86** | — |
| `pf-all` (k=8) | 78.7% | 1.045 | 82.4% | 36.0 ms | 5.92 | **−14%, real (p=0.010)** |
| `pf-k4` | 92.8% | 0.872 | 69.8% | 69.6 ms | 6.36 | −7%, not distinguishable (p=0.09) |

Prefetch never beat no-prefetch at any budget. Fetching all `top_k` is a
*measured regression*.

### The thing that did survive, and it is bigger

Multiply each path's throughput by its bytes per token:

```
                 tok/s   ×   GB/token   =   GB/s sustained
grouped           6.86       0.846          5.80  ████████████████████
pf-all            5.92       1.045          6.19  █████████████████████
pf-k4             6.36       0.872          5.55  ███████████████████

throughput varies  ±8%
bytes     vary    ±11%
the product is FLAT
```

That is a saturated link. **`decode tok/s = bandwidth ÷ bytes-per-token`**, and
prefetch only ever raises the denominator. Which means no reordering of
transfers can help at this operating point — overlap is not the lever, *bytes*
are.

So Stage 1c's premise was wrong in a more interesting way than "prefetch is
slow". Prefetch did exactly what it promised: it took 80 ms/token of blocking
transfer off the critical path (116.1 → 36.0 ms) and lifted the hit rate to
82%. It was still slower, because the side stream contends for the same PCIe
link and the link was already the constraint.

`ff-serve` now prints a sustained-GB/s column and says so out loud when that
column is flatter than the throughput column it derives from.

---

## What got corrected along the way

Stage 1c spent more time fixing measurement than writing kernels. Each of these
changed a number that was already published:

| # | The mistake | The correction |
|---|---|---|
| 1 | Prefetch budget quoted from a 384-slot profile | Measure the breakdown at the operating point: 5% → 50–71% |
| 2 | 384-slot row treated as a data point | It was a VRAM overcommit; `ff-serve` now prints VRAM headroom |
| 3 | VRAM check read 0.00 GB free *everywhere* | `empty_cache()` before `mem_get_info()` — the allocator counts as used |
| 4 | Fill timer wrapped the demand path only | Prefetch *relocated* work; the column showed a win that hadn't happened. Two columns now |
| 5 | Sweep rows compared across one invocation | Free RAM fell 12.4 → 5.8 GB; 256 slots read 2.77 tok/s vs 5.53 alone. Warn on drift |
| 6 | Pinned memory never switched on | Pageable `non_blocking` copies are synchronous — prefetch *cannot* overlap without it |
| 7 | Three device→host syncs per layer | `bincount` folds routed set + group sizes + prediction into one copy |
| 8 | "Real if bigger than the min–max spread" | Min–max **grows** with sample count, so more data raised the bar. Permutation test instead |

**#8 is the one worth remembering.** Three separate runs of prefetch-vs-grouped
came back "inside the spread — no finding" from a rule that structurally could
never say anything else as passes accumulated. It could not distinguish *the
effect is absent* from *the test cannot see it*.

**#7 is the one that was hiding in plain sight.** `.tolist()` and `.item()` are
the expensive operations on a hot path regardless of how little data they move.
Removing two of three per layer took the loop path's blocking fill from 130.4 to
98.4 ms/token — a bigger change than most of what was being benchmarked around
it, from code that was never the subject of a benchmark.

---

## Where it stands

```
0.40 ─────────► 4.69 ─────────► 5.98 ─────────► (6.86 on a later run)
baseline        Stage 1         Stage 1b         same grouped path,
(accelerate)    offload+LRU     grouped GEMM     quieter machine
```

⚠️ **The first three are one measurement.** 0.40 / 4.69 / 5.98 come from a
single `ff-serve --baseline --path both` invocation and can be compared: that
is the **15.0x**. The 6.86 is the same grouped path re-measured later at 96
gen-tokens on an idle box — it is not a fourth improvement, and there is no
baseline beside it. Stage 1c added **no throughput at all**.

**Shipping defaults:** grouped **on** (measured bit-identical output at fp16 on
the real model). Prefetch **off** — not "unproven" but *measured worse*: −14% at
full budget, p=0.010.

**Test suite:** 36 checks, CPU-only, no model download. The loop path is held to
a difference of **exactly 0.0** against the stock block; everything else is
measured against it.

**Open — and now sharply pointed.** The system delivers ~5.8 GB/s sustained
against Q7's 10.4 GB/s for a *pinned* transfer, and only 9 of 16 layers are
pinned. Every remaining lever reduces bytes or raises that constant; none of
them reorder transfers:

1. **Finish pinning** — the largest single lever, and a memory-budget problem
   rather than a kernel one. The ceiling is 11.8 GB against a 12.0 GB store.
2. **Q7's CPU path** — an expert computed in place is a transfer not made, and
   at decode every expert has 1 token routed to it against a break-even of 10.8.
3. **Eviction policy** — Q5 measured 23.4 points of Belady headroom over LRU.
   Fewer misses is fewer bytes.
4. **Quantised experts** — halves the bytes per miss directly.

Prefetch is closed, not parked. It works, and it cannot help here.
