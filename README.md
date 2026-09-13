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

So a single global LRU is the wrong default here. If real traces show the same
cliff, the fixes are per-layer cache partitioning or a frequency-biased policy.
Check where LRU crosses LFU before designing Stage 1.

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
notebooks/
  stage0.ipynb interactive version of the analysis
```

Router discovery is structural, not model-specific: it looks for
`layers.<i>.mlp.gate` modules that are `nn.Linear` projecting into
`num_experts`. That covers OLMoE and Qwen3-MoE unchanged. DeepSeek's `MoEGate`
is a custom module and will need its own branch in `discover_moe()`.

## Roadmap

- **Stage 0** — instrumentation and routing analysis *(current; runs on existing hardware)*
- **Stage 1** — expert cache + async prefetch in pure PyTorch, pinned RAM, separate CUDA stream. Most of the wall-clock win lives here.
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
