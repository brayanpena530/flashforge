# flashforge

MoE expert-caching and prefetch research, aimed eventually at running
DeepSeek-V4-Flash on constrained hardware — but built the other way round, on
models small enough that a full experiment takes seconds.

## Why it's structured this way

The optimizations worth exploring — expert caching, predictive prefetch,
memory-hierarchy scheduling — are properties of **MoE routing behaviour**, not
of any one model. Almost nothing about them is V4-specific. So the dev ladder
runs small→large, and iteration speed is treated as the primary constraint:

| Stage | Model | Topology | Purpose |
|---|---|---|---|
| dev | `allenai/OLMoE-1B-7B-0924-Instruct` | 64 experts, top-8, 16 MoE layers | fits a 6GB card; instant iteration |
| realistic | `Qwen/Qwen3-30B-A3B` | 128 experts, top-8, 48 layers | forces genuine RAM→VRAM streaming |
| target | `deepseek-ai/DeepSeek-V4-Flash` | 256 experts, top-6 + 1 shared | the endgame |

Qwen3-30B-A3B is structurally close to V4-Flash (fine-grained, high expert
count, low top-k), so policies tuned there should port with the constants
changed rather than the design.

## Measured results

Stage 0 has been run end to end. Everything below is measured, not estimated.

**Setup.** OLMoE-1B-7B-0924-Instruct (16 MoE layers, 64 experts, top-8, 12.6 MB
per expert at fp16) on an RTX 2060 6GB with the checkpoint split across GPU and
system RAM, 8 CPU threads, 31.8 GB RAM, Samsung NVMe. Corpus of 48 sequences ×
512 tokens = 24,576 tokens, 8 sequences in each of six domains.

| | Result | What it means |
|---|---|---|
| **Q1** | hottest 10% serve **21.6%**, Gini 0.258, **zero** unused experts | mild skew — a small pinned cache helps, but every expert earns its keep |
| **Q2** | **40.1%** consecutive overlap vs 12.5% random | strong temporal locality |
| **Q3** | `stale_router` **0.835** at k=1; `probe` **0.726** at k=8 | prefetch works, and the two tiers want different predictors |
| **Q4** | between-domain JS **24.8×** the within-domain floor | domains separate cleanly — cache warming has real signal |
| **Q5** | Belady 77.9%, **LRU 54.5%**, static 42.2%, LFU 42.1% | 23.4% headroom, and LRU is the *best* online policy |
| **Q6** | union(4) = **2.48×** top-k → 1.61× fewer bytes/token | moderate; speculation needs high draft acceptance |
| **Q7** | `t_c = 0.0367·m + 1.073` ms, PCIe 1.212 ms, **m\* = 10.8 tokens** | experts under ~11 tokens are cheaper on the CPU |
| **Q8** | 2.39 GB/s peak, 8.27 ms/expert, **4 layers** of lookahead needed | a disk tier is reachable, at 2× prefetch budget |

### The four findings that changed the design

**Predictor choice is tier-dependent.** `stale_router` — running layer *N+k*'s
real router on layer *N*'s hidden state, no training, no parameters — wins at
one layer ahead. But it loses **41%** of its recall between k=1 and k=8, while
a ridge probe loses **11%**. From k=2 onward the probe is simply better.

```
1x budget      k=1     k=2     k=4     k=8
stale_router   0.835   0.775   0.669   0.495
probe          0.814   0.792   0.762   0.726
```

RAM→GPU needs one layer, so use the free predictor. Disk→RAM needs four, where
a flat decay curve is worth more than peak accuracy. Judging both by the same
predictor hides the second case entirely.

**LRU is the right default here — and an earlier version of this README said
the opposite, because the cache simulator was reading a biased sample.**

The sim budgets itself with `--max-accesses`, which used to truncate the
flattened access stream to its first N entries. That stream is ordered by
`(seq_id, pos, layer)`, so a prefix is the first few *documents* — and this
corpus is grouped by domain, so the default 300k budget showed the simulator
**five `code` documents and nothing else.** Inside a single-domain monoculture
expert usage is maximally concentrated, frequency policies look excellent, and
LFU (72.4%) beat LRU (65.2%).

Across all 48 documents the ranking reverses:

```
                belady     lru     lfu  static
5 code docs      83.9%   65.2%   72.4%   72.8%    <- prefix, domain monoculture
all 48 docs      77.9%   54.5%   42.1%   42.2%    <- LRU wins by 12 points
```

`subsample_sequences` now budgets by dropping whole sequences at random, so
each document's internal access pattern stays intact while corpus diversity is
preserved. The cyclic-sweep cliff described below is still real *below one
token's working set* — but above it, what decides the ranking is how many
distinct kinds of document the cache sees.

**Prefill and decode behave almost identically.** Captured 768 real greedy
decode steps to check whether serving differs from prefill. At matched shape
(48 sequences × 16 tokens) the two are within about one point on every policy:

```
                   belady     lru     lfu  static
prefill[0:16]       75.2%   52.3%   45.6%   46.6%
decode              76.2%   53.1%   41.1%   42.2%
prefill[248:264]    76.2%   51.4%   41.8%   43.0%
```

Q2 agrees: 40.1% consecutive-expert overlap on prefill, 40.9% on decode. So
prefill traces are a sound proxy for serving behaviour on this model, which is
useful because they are roughly 9× cheaper to collect — decode runs one forward
pass per token and cost 2.25 s/token here against 0.03 s/token for prefill.

**The storage tier wants batched reads, not a deeper queue of single ones.**

```
request size   peak GB/s   knee QD   depth-1 penalty
  0.0625 MiB       1.42        16        6.52x
     1 MiB         1.97        16        1.77x
    12 MiB         2.39         2        1.59x
```

PCIe saturates at queue depth 1, which is why serial I/O cost models work for
it. NVMe does not: small reads need **16 concurrent requests** to reach peak,
and issuing them one at a time costs 6.5×. Store experts contiguously, read
them whole, and batch the prefetch — a scheduler fetching one expert at a time
cannot use the drive however good its predictions are.

### Caveats

Single model, single machine. `m*`, the PCIe figures and the whole storage
curve are properties of this hardware and have to be re-measured elsewhere —
which is what `ff-bench` is for. The corpus is 48 sequences; large enough to
make the probe well-determined (≈14,700 training samples against 2,048
dimensions) but not a substitute for a real workload trace.

Q1–Q6 are measured on prefill, with 768 real decode steps used to confirm the
two phases agree rather than to carry the headline numbers. Any policy tuned
here should be re-checked on a decode trace at serving scale before it ships,
even though the two matched closely at this one.

**Q7 and Q8 have to come from the same `ff-bench` run.** They were briefly
mixed here: an earlier version of this table quoted Q7 from a run whose scratch
file sat on a spinning disk (`m* = 18.0`) alongside Q8 from the later NVMe run.
The two differ because the second run's CPU was busier, not because the machine
changed — `m*` moved from 18.0 to 10.8 on a measurement of the *CPU*, which is
a useful reminder that these constants carry the load of the box at the moment
they were taken. `hardware.json` records the whole set together for exactly
this reason; read it rather than copying individual figures across.

That mistake, and every other one this project has made in the course of
producing a number, is written up in [`troubleshoot.md`](troubleshoot.md).
Read it before publishing a measurement.

## Stage 1 — the offload runtime

Experts live in host RAM; a fixed pool of GPU slots caches the hot ones; the
MoE block pulls what it needs through that cache. LRU, because Q5 measured it
12 points ahead of the frequency policies.

```bash
uv run ff-serve --baseline --capacity 128,256,384 --gen-tokens 32 --repeats 5
```

```
  slots  VRAM GB  prefill t/s  pf hit  decode t/s        spread  dec hit  GB/tok
    128     2.39         78.6   1.2%        3.84     2.87-3.89   33.0%   1.080
    256     3.89         82.4   8.3%        4.41     3.94-4.47   46.8%   0.858
    384     5.39         86.2  26.5%        3.53     3.21-3.63   92.6%   0.120

baseline                 45.3               0.36     0.34-0.38
```

**4.41 tok/s against a measured accelerate-offload baseline of 0.36 — 12.1x**,
on a model 2.3x larger than the card it runs on. Decode figures are the median
of five timed passes; the baseline runs the same prompt through the same
harness. The runtime is bit-exact against the stock block —
`tests/runtime_check.py` asserts a max absolute difference of zero.

### The finding that redirected Stage 1

**Transfer is not the bottleneck, and the roadmap had the order wrong.**

Read the `spread` column before the `decode t/s` column. Across capacities the
median moves 20%; *within* a single capacity, repeated passes move 26%. The
sweep cannot resolve a difference between these three configurations at all —
and that is the result, not a failed measurement. Transfer volume falls **9x**
from 128 slots to 384 and decode speed does not reliably respond.

The arithmetic agrees. At 384 slots the remaining 0.120 GB/token is, at the
measured 10.4 GB/s PCIe, ~11 ms of a ~230 ms token. A profiler puts
`aten::copy_` at **4.95%** of decode.

The other 95% is dispatch. A decode token issues **465 separate GEMMs** —
16 layers x (8 experts x 3 projections + gate) plus attention — each one
multiplying a *single* token's activations against a 2048x1024 matrix. Self CPU
time lands within 7% of self CUDA time, which is what a launch-bound workload
looks like: both sides doing bookkeeping rather than arithmetic.

So the planned next step was wrong. **Prefetch would buy at most 5% here**,
because at a workable cache size the transfers are already nearly free. The
lever is collapsing the per-expert GEMMs into one grouped call — filed under
Stage 2 as a kernel concern, actually the Stage 1 bottleneck. Stage 1b below
does that, and gets 27%.

Note also that 384 slots is not the best configuration despite the best hit
rate. 5.39 GB resident on a 6 GB card leaves little room for activations, and
the median comes out *below* 256 slots. Whether that is real or more of the
same variance, this sweep cannot say — but "give the cache everything spare"
is not supported.

This is the second time a confident prediction from Stage 0's constants stood
until something measured it. The constants were right; the inference was not.

### Two things worth knowing before reading the table

**Prefill and decode hit rates are not comparable.** A prefill batch touches
the *union* of its tokens' experts, which past a few dozen tokens is every
expert in the layer — those misses are compulsory and no capacity removes them.
A decode step touches exactly top-k per layer. Only the decode column belongs
next to Q5's simulated 54.5%, and at 256 slots it measures 46.8%, close enough
that the simulator was doing its job.

**Time more than one pass, and measure the baseline yourself.** The first
version of this section reported single passes and quoted 9.8x against a figure
lifted from Stage 0's decode *tracing* run — which had forward hooks writing
router logits and hidden states for 16 layers to disk on every token, at a
different prompt length. That is not a baseline, it is a different experiment.
`--baseline` now runs `device_map="auto"` through the same harness on the same
prompt, in a subprocess because accelerate's offload does not reliably release
its CPU-side weights (measured as a segfault when the next model load ran into
what it had kept). Single-pass timing also produced a tidy monotonic table,
3.94 -> 4.18 -> 4.33, that five passes show was noise.

**Sweep capacities in one process only if the RAM allows it.** The expert store
is 12 GB against ~15 GB free. An early version of the sweep held the previous
iteration's store alive while loading the next model, which never raised — it
just swapped, and produced a table where a *better* hit rate came with *worse*
throughput (46.8% at 2.33 tok/s against 33.0% at 3.67). That reads exactly like
a cache-policy finding and is not one. `ff-serve` now frees the old store
first, and prints free host RAM at each step so the failure is visible.

## Stage 1b — the grouped GEMM path

Stage 1's profile said dispatch, not transfer. So `CachedMoEBlock` grew a
second path: sort the (token, expert) pairs by expert, pad each expert's group
to a common width, gather the routed experts' weights into one batched tensor,
and run three `bmm` calls for the whole layer instead of 3E separate GEMMs.

```bash
uv run ff-serve --baseline --capacity 256 --path both --repeats 5 --gen-tokens 32
```

```
  slots     path  VRAM GB  prefill t/s        spread  decode t/s        spread  dec hit
    256     loop     3.89         92.3 63.3-94.9            4.69 4.17-4.69       46.8%
    256  grouped     3.89         99.7 80.0-104.3           5.98 4.61-6.06       46.7%

  slots    phase   loop t/s  grouped t/s    change  verdict
    256  prefill      92.30        99.72      +8%  inside the 34% within-path spread — no finding
    256   decode       4.69         5.98     +27%  real

baseline                 48.7                0.40  0.39-0.40
```

**5.98 tok/s against a baseline of 0.40 measured in the same invocation —
15.0x.** Decode gains 27% over the loop path, and reproduced at +39%, +35%,
+29% and +27% across four runs.

Both paths are timed on the *same* loaded model, the same slot pool and the
same warm cache — the flag flips on the already-installed blocks — so the only
variable is the path. That matters more than it sounds: the identical hit rates
and identical GB/token in the table are the evidence that nothing else moved.

**Prefill is not a result.** The first Stage 1b run appeared to show a 36%
prefill *regression*, with a plausible story attached — a prefill batch routes
to all 64 experts, so the gather copies 805 MB per layer to save launches the
loop path was not wasting. The next run reversed the sign. Prefill is one short
timed region per pass and its spread is 63.3–94.9 tok/s on the loop path alone,
which swallows the difference whole. `ff-serve` now prints a prefill spread
column for exactly this reason, and says "no finding" rather than "+8%".

### The grouped path is not bit-exact, and that was measured too

One `index_add_` over every expert at once has no defined accumulation order,
so the grouped path cannot be held to the zero-difference standard the loop
path meets. `tests/runtime_check.py` holds it to a tolerance instead (`2e-19`
in fp32 on the toy block) and keeps the loop path as the oracle.

A tolerance on a toy block in fp32 says nothing about fp16 on a 7B model, where
one rounding difference can flip an argmax and the two paths then walk away
from each other. So `ff-serve` decodes greedily on both paths and reports the
first token id where they differ: **identical for all 65 tokens**. That is why
`grouped=True` is the default and `grouped=False` is still there.

## Stage 1c — the prefetch budget, and a profile taken in the wrong place

Stage 1's profile put `aten::copy_` at 4.95% of decode, and the roadmap below
used to say prefetch was "worth at most ~5%". Both were wrong, for the same
reason: **that profile was taken at 384 slots and the runtime ships at 256.**

```
ff-serve --capacity 256 --path all --repeats 5 --pin-gb 7
```

| slots | fill ms/token | % of decode | ceiling if prefetch were perfect |
|------:|--------------:|------------:|---------------------------------:|
| 128   | 216.7         | 77.0%       | +335% |
| 256   | 126.3         | 70.9%       | +244% |
| 320   | 123.0         | 22.4%       | +29%  |

Transfers were never nearly free. They are most of decode.

**And the 384-slot row was never a valid measurement.** It puts 5.39 GB resident
on a 6 GB card. On Windows that does not raise — WDDM backs the overflow with
host memory, so every read of a "resident" expert crosses PCIe again. It
measured **0.83 tok/s at a 92.7% hit rate**, against 5.53 tok/s at 46.9% with
256 slots: hit rate doubled, throughput fell 6.7x. That is the signature of a
resource bug, one tier down from the host-RAM version already in
[troubleshoot.md](troubleshoot.md). `ff-serve` now prints VRAM headroom per row.

### The prefetcher works. The speedup does not reproduce.

`CachedMoEBlock` runs the next layer's router on this layer's hidden state
(Q3's `stale_router`) and issues the predicted fills on a side CUDA stream.
At 256 slots, 6.75 GB of the store pinned, one invocation:

| path | decode t/s | blocking fill | decode hit | GB/token |
|------|-----------:|--------------:|-----------:|---------:|
| loop     | 4.98 | 130.4 ms (50%) | 46.8% | 0.858 |
| grouped  | 5.85 | 100.5 ms (65%) | 46.7% | 0.859 |
| prefetch | 6.08 | **32.7 ms (20%)** | **81.8%** | 1.069 |

Every intermediate metric says it worked: hit rate 46.8 → 81.8%, blocking fill
down by two thirds, predictor precision 78.2% measured in the runtime rather
than quoted from Q3's offline recall.

Decode throughput is **worse**. Nine passes per path, one invocation, with a
two-sided permutation test on the difference of medians:

| path | decode t/s | GB/token | change | verdict |
|------|-----------:|---------:|-------:|---------|
| grouped | **6.86** | 0.846 | — | — |
| `pf-all` (all top_k) | 5.92 | 1.045 | −14% | **real, p=0.010** |
| `pf-k4` (top 4 only) | 6.36 | 0.872 | −7% | not distinguishable, p=0.09 |

An earlier five-pass run showed the opposite — `pf-k4` peaking at 6.50 against
5.97 — and the *baseline* was the noisy one. Prefetch never beat it at any
speculation budget. It is **off by default**: not unproven, measured worse.

### Why — and this is the part worth keeping

Multiply each path's throughput by its bytes per token:

```
                tok/s   ×   GB/token   =   GB/s sustained
grouped          6.86       0.846          5.80
pf-all           5.92       1.045          6.19
pf-k4            6.36       0.872          5.55

throughput varies  ±8%   |   bytes vary ±11%   |   the product is flat
```

That is a saturated link. On one, `decode tok/s = bandwidth ÷ bytes-per-token`,
and prefetch only ever raises the denominator. It did everything it promised —
80 ms/token of blocking transfer removed from the critical path, hit rate 47.5%
→ 82.4% — and lost anyway, because the side stream contends for the same PCIe
link that was already the constraint.

**No reordering of transfers can help at this operating point. Only moving
fewer bytes can.** `ff-serve` now prints a sustained-GB/s column and says so
when that column is flatter than the throughput it derives from.

### Two things that fell out of it

**Pinning was never on.** Every measurement in this project before now ran with
a fully pageable store, where `copy_(non_blocking=True)` is synchronous and a
side stream cannot overlap at all. Pinning is a prerequisite for prefetch, not
an optimisation beside it. The ceiling is lower than free RAM suggests: 11.8 GB
pinnable on this 32 GB machine against a 12.0 GB store, reported as `CUDA error:
out of memory` from a *host* allocation. `ExpertStore` now degrades to pageable
for the remaining layers instead of discarding an eleven-minute load.

**The hot path was doing three device→host syncs per layer** — `unique`,
`counts`, and the prediction — 16 times per token. `torch.bincount` returns the
routed set and the group sizes in one vector, so the prediction concatenates
onto it and a layer costs one copy. That alone took the loop path's blocking
fill from 130.4 to 98.4 ms/token, and it preserves the ascending expert order
that keeps the loop path bit-exact at *exactly zero* difference.

## Stage 0 — instrumentation

Answers eight questions, each of which gates a later design decision. Q1–Q6
measure the **model** and need a trace; Q7–Q8 measure the **machine** and need
neither a trace nor a checkpoint.

| | Question | Decides |
|---|---|---|
| **Q1** | How skewed is expert usage? | whether a small pinned cache is viable |
| **Q2** | Does token *t+1* reuse token *t*'s experts? | whether recency-based eviction makes sense |
| **Q3** | Can layer *N* predict layer *N+k*'s routing? | **whether prefetch works at all** |
| **Q4** | Does usage cluster by domain? | whether per-conversation cache warming pays |
| **Q5** | Hit rate vs capacity, incl. Belady | how much headroom any online policy has |
| **Q6** | How fast does the expert set grow with block size? | whether batching/speculation amortises loads |
| **Q7** | What does one expert cost on CPU, GPU and PCIe? | where the CPU/GPU placement line falls |
| **Q8** | How fast is a read off disk, and at what queue depth? | whether a disk tier can ever be hidden |

Q1 is reported **per layer and per band**, not just pooled. Expert usage does
not behave the same way at every depth, and two different things vary with it:

- **Hot-expert concentration** — what share of accesses the hottest 10% serve.
  Decides whether a small pinned cache pays, and where.
- **Routing-weight dominance** — how much of the gate's output mass goes to the
  top-ranked expert. This reads the `weight` column the tracer has always
  written and nothing else touches.

The second one matters because it separates two regimes that look identical if
you only count accesses. When one expert's weight dominates, the block's output
is close to that single expert's, so its routing decision propagates strongly
to the next layer — which is what makes *cross-layer routing correlation* high.
When weights are balanced the output is a genuine blend, the hidden state barely
moves between layers, and it is *hidden-state similarity* that is high instead.

Those two regimes want different prefetch features — previous-layer expert IDs
in the first case, the hidden state itself in the second. So Q1's depth profile
is what tells you which feature to feed Q3's predictor at which depth, and the
two bands tables are meant to be read side by side.

**Q3 is the one that decides the project.** Prefetch only pays if you have lead
time: predicting layer *N+k* at layer *N* buys *k* layers of compute to hide the
transfer behind. If accuracy collapses at *k*=1, the design has to change — so
measure it before building anything.

Four predictors are compared, cheapest first:

- `prior` — static global top-k for the target layer. The floor; beat it or
  prefetch is pointless.
- `identity` — reuse layer *N*'s own expert indices.
- `stale_router` — run layer *N+k*'s **actual** router on layer *N*'s hidden
  state. No training, no extra parameters, free at inference time. This is the
  one to beat, and often the one to ship.
- `probe` — ridge regression from hidden state to routing probabilities. An
  approximate ceiling on a learned predictor of this size.

Q3 reports recall **per layer band**, not as one stack-wide average.
Predictability is not uniform with depth: near-input and near-output layers
have skewed routing weights and hidden states that shift sharply between
layers, while middle layers hold a nearly stable hidden state and lean on a
small set of hot experts. A single averaged number blends those regimes, and a
predictor that is strong through the middle half with weak ends looks merely
mediocre — which is the wrong conclusion, and points at the wrong fix. Band
boundaries are a parameter (`--edge-fraction`), because no published rule fixes
them.

The offset sweep runs to 8 by default rather than 4. Lookahead depth is not a
free choice; it is set by what you are hiding behind it. One layer of compute
covers a PCIe transfer, which is why a one-layer horizon is enough today. A
disk read is several times slower and needs a proportionally deeper horizon.
**Q8 measures that ratio and Q3 says whether prediction survives out that far** —
see the disk-tier section below.

Q5 includes **Belady's optimal** because it bounds every online policy. If LRU
already sits near Belady, eviction is a dead end and the effort belongs in
prefetch; if the gap is wide, eviction policy is worth real work.

Q6 asks what happens when you stop thinking one token at a time. Process *B*
tokens in one forward pass — speculative-decoding verification, or a block-ahead
prefetch — and you pay for the **union** of experts those tokens touch, not *B*
times one token's worth. If `union(4) ≈ 1.3 × top_k`, one load serves four
tokens and speculative decoding becomes a *bandwidth* optimisation rather than a
decode-speed trick, which is the larger effect on memory-bound hardware. If it
grows near-linearly ("expert scattering"), batching buys little until the
draft's expert footprint is constrained. The random-routing line is the null
model; landing on it means there is no block-level reuse to exploit.

Q3 and Q6 together set the shape of Stage 1:

| | expansion slow | expansion fast |
|---|---|---|
| **predictable** | block-ahead prefetch, large lookahead | per-token prefetch, small lookahead |
| **unpredictable** | batch anyway — reuse carries it | eviction and pinning only |

## Q7 and Q8 — measuring the machine

Q1–Q6 answer "does the model's routing have exploitable structure?". Neither
answers "what does exploiting it actually cost here?", and every scheduling
decision in Stage 1 turns on constants that are properties of *this* box.

**Q7 — the cost model.** A hybrid CPU/GPU runtime chooses, per expert, between
shipping it across PCIe and running it in place on the CPU. GPU cost is roughly
flat in token count (the transfer dominates); CPU cost is linear,
`t_c = β·m + C`. Those lines cross at some **m\***, and m\* is the number the
whole placement policy resolves against. Below it, an expert is cheaper
computed in place — and every expert you keep on the CPU frees a PCIe slot for
a prefetch, which is what makes prefetching a budgeted decision rather than a
heuristic.

Q7 also measures the transfer three ways — pageable, pinned, and split across
three CUDA streams — because a scheduler that plans against `t_io` needs
`t_io` to be a number it can trust.

**Q8 — the storage curve.** Read bandwidth and latency against request size and
queue depth. The headline is not peak bandwidth, it is **the queue depth at
which you reach peak**. PCIe saturates at depth 1, which is why serial-I/O cost
models work for it. NVMe does not: a single outstanding read leaves the device
mostly idle. If the knee is at 8, a prefetcher that issues one expert at a time
cannot use the drive no matter how good its predictions are, and the disk tier
needs batched requests — a different scheduling shape, not the same one scaled
down.

The two combine into one number: `t_disk / t_layer`, rounded up, is how many
layers of lookahead a disk read needs to hide behind. `ff-bench` writes it to
`hardware.json`, `ff-analyze` picks it up automatically and draws it on the Q3
chart. Where a band's curve has already collapsed by that line, per-layer
routing prediction cannot cover a disk tier there, and the disk→RAM decision
needs a longer-horizon signal instead — block-level speculation, or Q4's domain
warming.

Two ways to be misled, both of which the tooling shouts about:

- **The page cache.** A scratch file smaller than your RAM is read from DRAM,
  and the resulting figure lands comfortably inside the plausible range for a
  good NVMe drive. `ff-bench` compares the file size against physical RAM and
  marks the run untrusted in both the console output and `hardware.json`. Pass
  `--file-size-gb` above your RAM for numbers you intend to design against.
- **Bands that ran out of layers.** You cannot look 8 layers past the end of
  the stack, so the output band has no deep-offset pairs. That is missing data,
  not a prediction failure; the reach table reports it as `limited_by=coverage`
  and the disk verdict excludes those bands rather than manufacturing a ceiling
  out of arithmetic.

β is measured in fp32, since CPU fp16 GEMM is emulated in most builds and would
time the emulation. A production CPU path would use a quantised kernel and beat
it, so the fitted β is an **upper bound** — which makes any "CPU execution is
worth it" conclusion drawn from it conservative.

## Setup

```bash
uv sync
```

`torch` comes from the PyTorch cu124 index (configured in `pyproject.toml`) —
the default PyPI wheels for Windows are CPU-only. The torch pin is `>=2.6,<2.7`
**on purpose**: torch 2.6 pairs with Triton 3.2, the last Triton release that
officially supports Turing (sm_75 / RTX 2060). Triton dropped Turing in 3.3.
Lift the pin once the GPU is sm_80+.

### Put the model cache on a roomy drive first

A 7B fp16 checkpoint is ~14GB and the HF cache defaults to `C:`.

```powershell
$env:HF_HOME = "D:\ai\hf-cache"
```

`ff-collect` checks free space and warns before downloading.

## Running it

```bash
uv run ff-collect --out traces/olmoe
```

```bash
uv run ff-bench --traces traces/olmoe --file-size-gb 48
```

```bash
uv run ff-analyze --traces traces/olmoe --bytes-per-expert 12.6e6
```

Run `ff-bench` before `ff-analyze`: it drops `hardware.json` in the report
directory, and `ff-analyze` reads the required lookahead depth from it so Q3
and Q8 can be read against each other without copying numbers by hand. It needs
no trace of its own — `--traces` is only there to pick up the expert's shape,
and `--hidden-size` / `--intermediate-size` work instead.

`ff-analyze` writes CSVs and charts to `traces/olmoe/report/` and prints a
verdict line for each question. `--bytes-per-expert` turns hit rates into
**MB fetched per token**, which divided by your storage bandwidth is seconds
per token — the number that actually matters.

Useful flags:

| Flag | Command | Effect |
|---|---|---|
| `--gen-tokens N` | collect | also capture real greedy decode steps (true serving access pattern) |
| `--load-4bit` | collect | ~4GB instead of ~14GB, much faster iteration, **perturbs routing** |
| `--no-hidden` | collect | skip hidden states, disables Q3 |
| `--prompts file.jsonl` | collect | your own corpus: `{"id":…, "domain":…, "text":…}` per line |
| `--q3-offsets 1,2,4,8` | analyze | lookahead depths to sweep |
| `--edge-fraction 0.25` | analyze | share of layers in each of the input/output bands |
| `--required-lookahead N` | analyze | draw this depth on the Q3 chart (default: from `hardware.json`) |
| `--file-size-gb N` | bench | scratch file size. **Must exceed your RAM** or Q8 measures the page cache |
| `--layer-time-ms X` | bench | measured per-layer decode time; without it, a conservative floor is used |
| `--skip-q7` / `--skip-q8` | bench | run one half only |
| `--remove-scratch` | bench | delete the scratch file afterwards (recreated next run) |
| `--capacity 128,256` | serve | expert slots to cache; comma-separated to sweep |
| `--baseline` | serve | also time accelerate's `device_map="auto"` offload, same prompt and harness |
| `--repeats N` | serve | timed passes per capacity, reported as median (min–max). Default 3 |
| `--pin-gb N` | serve | host RAM to page-lock for async DMA. Pinned pages cannot be swapped |

The built-in prompt set is a starting point sized for a few thousand tokens.
For numbers you intend to trust, point `--prompts` at a real corpus — a couple
of hundred sequences of 512+ tokens.

## Validating changes without a model

```bash
uv run python tests/synthetic_check.py
```

Builds a synthetic trace with known planted structure and asserts each analysis
recovers it — no download, no GPU, ~2 minutes. Run it after touching
`analysis.py`, `hardware.py`, `cachesim.py`, or `plots.py`.

```bash
uv run python tests/runtime_check.py
```

Does the same for Stage 1, in a few seconds, against a randomly initialised
OLMoE block on CPU. The load-bearing check is **parity**: the cached block must
match the stock `OlmoeSparseMoeBlock` exactly, including at a capacity small
enough to force eviction on every layer. It asserts a max absolute difference
of *zero*, not a tolerance, because the block reproduces the original's op
order deliberately — a caching bug that served the wrong expert would otherwise
show up as slightly worse generated text and nothing else.

The controls are the point. Q4 runs a positive **and** a negative control, so
the metric has to discriminate rather than always answer "yes". Q3's band split
has a planted U-shaped drift — edge layers wander further from the shared base
than middle ones — so the split has to recover a known ordering rather than
bucket noise. Q6 has an i.i.d. control that must land *on* the random-routing
null rather than below it.

Q7's fitting is asserted against a curve with known coefficients rather than a
live measurement: a microbenchmark on a busy desktop is genuinely noisy, and
asserting goodness-of-fit on it tests whether the machine happened to be quiet.
The live curve is still exercised, but only for direction.

### One thing it already found

MoE decode is a **cyclic** access pattern: each token sweeps every layer,
touching about `top_k * n_layers` distinct experts before returning to layer 0.
That is LRU's textbook worst case. Below a capacity of one token's working set,
LRU evicts every entry just before its next use — in the synthetic check its hit
rate is **exactly zero** while LFU and static are at 25–40% on the same trace.

So a single global LRU is the wrong default *at that capacity*. The fixes are
per-layer cache partitioning or a frequency-biased policy. Check where LRU
crosses LFU before designing Stage 1.

**Only below one token's working set.** On the real 24,576-token trace at 25%
capacity (256 slots, twice a token's 128-slot sweep) LRU reaches 54.5% against
LFU's 42.1% — recency wins comfortably. The cliff is genuine, but it lives
below the capacity anyone would actually provision. What matters at realistic
capacities is corpus diversity, not the sweep.

## Layout

```
flashforge/
  models.py    model loading + structural MoE router discovery
  tracing.py   forward hooks; captures router logits + gate-input hidden states
  prompts.py   domain-tagged prompt set
  analysis.py  Q1-Q6 — the routing questions, from a trace
  hardware.py  Q7-Q8 — the cost model and storage curve, from the machine
  cachesim.py  Belady / LRU / LFU / static sweep
  plots.py     charts
  viz.py       validated palette + matplotlib style
  runtime/     Stage 1 — the offloaded MoE runtime
    store.py   expert weights in host RAM, one contiguous row each
    cache.py   preallocated GPU slot pool, LRU eviction
    block.py   drop-in replacement for the model's MoE block
    patch.py   installs the above into a loaded model
notebooks/
  stage0.ipynb interactive version of the analysis
```

Router discovery is structural, not model-specific: it looks for
`layers.<i>.mlp.gate` modules that are `nn.Linear` projecting into
`num_experts`. That covers OLMoE and Qwen3-MoE unchanged. DeepSeek's `MoEGate`
is a custom module and will need its own branch in `discover_moe()`.

## Roadmap

- **Stage 0** — instrumentation and routing analysis *(complete; see "Measured results")*
- **Stage 1** — expert cache in pure PyTorch, experts in host RAM, LRU eviction. *(done: 4.41 tok/s, 12.1x over a measured offload baseline. The Stage 1b run re-measured the same loop path at 4.69 against a 0.40 baseline, 11.8x — two runs, each internally consistent; do not cross them.)*
- **Stage 1b** — grouped expert GEMM, promoted from Stage 2 because 95% of decode was per-expert dispatch rather than transfer. *(done: 5.98 tok/s, +27% on decode, 15.0x over the baseline measured in the same run. Prefill unchanged within noise.)*
- **Stage 1c** — async prefetch on a side stream, one layer ahead with `stale_router` (Q3). *(built, measured, and **closed negative**. The "worth at most ~5%" this line used to carry came from a profile of an invalid configuration; the real fill budget at 256 slots is 50–71% of decode. The prefetcher works — blocking fill 116 → 36 ms/token, decode hit rate 47.5% → 82.4% — and is still 14% slower, p=0.010, because the link is saturated and it adds 23% more bytes. Off by default. The finding is that **transfer reordering cannot help here at all**; see "Stage 1c" above.)*
- **Stage 1d — move fewer bytes.** The sustained-bandwidth column is now the design constraint: ~5.8 GB/s against Q7's 10.4 GB/s for a pinned transfer. In rough order of expected payoff: (1) **finish pinning** — only 9 of 16 layers fit under the 11.8 GB pinned ceiling today, and this is a memory-budget problem rather than a kernel one; (2) **Q7's CPU path** for experts under ~11 routed tokens, which at decode is all of them; (3) **eviction policy**, where Q5 measured 23.4 points of Belady headroom over LRU; (4) **quantised experts**, which halves bytes per miss outright.
- **Stage 2** — Triton kernels: fused gather-GEMM, dequant, fused router. *Needs sm_80+.*
- **Stage 3** — scale to V4-Flash, where the routed pool may not fit in RAM either and experts stream disk→RAM→GPU. *Needs RAM and NVMe headroom.* Q8 is the go/no-go: if `t_disk` needs more lookahead than Q3 shows prediction surviving, the disk tier has to be driven by a longer-horizon signal rather than per-layer routing prediction.

### Prior art worth reading before Stage 1

- Mixtral-offloading — LRU expert cache + speculative prefetch from the prior layer's hidden state. The Stage 1 baseline.
- [LayerScope](https://arxiv.org/pdf/2509.23638) — predictive cross-layer scheduling. Directly relevant to Q3 (its layer-group finding is why Q3 reports per band) and to Q7 (its `t_c = β·m + C` is the cost model Q7 calibrates).
- [DraftExpert](https://arxiv.org/abs/2607.24434) — self-speculative decoding costed in expert loads rather than tokens. Relevant to Q6, and the source of the block-level lookahead a disk tier would need.
- [PIPO](https://arxiv.org/pdf/2504.03664) — pipelined offloading on consumer devices.
- [CPU-GPU hybrid MoE SLOs](https://arxiv.org/pdf/2606.10493), [activation sparsity](https://arxiv.org/pdf/2509.00454).

One caveat for Stage 3: the [llama.cpp V4 port](https://blog.teamblobfish.com/posts/deepseek-v4-flash-llama-cpp/)
found V4 decode is compute-bound on the indexer/sinkhorn path rather than purely
bandwidth-bound, so caching wins will be partly masked by compute that can't be
cached away.
