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

Answers six questions, each of which gates a later design decision:

| | Question | Decides |
|---|---|---|
| **Q1** | How skewed is expert usage? | whether a small pinned cache is viable |
| **Q2** | Does token *t+1* reuse token *t*'s experts? | whether recency-based eviction makes sense |
| **Q3** | Can layer *N* predict layer *N+k*'s routing? | **whether prefetch works at all** |
| **Q4** | Does usage cluster by domain? | whether per-conversation cache warming pays |
| **Q5** | Hit rate vs capacity, incl. Belady | how much headroom any online policy has |
| **Q6** | How fast does the expert set grow with block size? | whether batching/speculation amortises loads |

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
uv run ff-analyze --traces traces/olmoe --bytes-per-expert 12.6e6
```

`ff-analyze` writes CSVs and charts to `traces/olmoe/report/` and prints a
verdict line for each question. `--bytes-per-expert` turns hit rates into
**MB fetched per token**, which divided by your storage bandwidth is seconds
per token — the number that actually matters.

Useful flags:

| Flag | Effect |
|---|---|
| `--gen-tokens N` | also capture real greedy decode steps (true serving access pattern) |
| `--load-4bit` | ~4GB instead of ~14GB, much faster iteration, **perturbs routing** |
| `--no-hidden` | skip hidden states, disables Q3 |
| `--prompts file.jsonl` | your own corpus: `{"id":…, "domain":…, "text":…}` per line |

The built-in prompt set is a starting point sized for a few thousand tokens.
For numbers you intend to trust, point `--prompts` at a real corpus — a couple
of hundred sequences of 512+ tokens.

## Validating changes without a model

```bash
uv run python tests/synthetic_check.py
```

Builds a synthetic trace with known planted structure and asserts each analysis
recovers it — no download, no GPU, ~2 minutes. Q4 runs a positive **and** a
negative control, so the metric has to discriminate rather than always answer
"yes". Run it after touching `analysis.py`, `cachesim.py`, or `plots.py`.

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
  analysis.py  the five questions
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
- **Stage 3** — scale to V4-Flash. *Needs RAM and NVMe headroom.*

### Prior art worth reading before Stage 1

- Mixtral-offloading — LRU expert cache + speculative prefetch from the prior layer's hidden state. The Stage 1 baseline.
- [LayerScope](https://arxiv.org/pdf/2509.23638) — predictive cross-layer scheduling. Directly relevant to Q3.
- [PIPO](https://arxiv.org/pdf/2504.03664) — pipelined offloading on consumer devices.
- [CPU-GPU hybrid MoE SLOs](https://arxiv.org/pdf/2606.10493), [activation sparsity](https://arxiv.org/pdf/2509.00454).

One caveat for Stage 3: the [llama.cpp V4 port](https://blog.teamblobfish.com/posts/deepseek-v4-flash-llama-cpp/)
found V4 decode is compute-bound on the indexer/sinkhorn path rather than purely
bandwidth-bound, so caching wins will be partly masked by compute that can't be
cached away.
