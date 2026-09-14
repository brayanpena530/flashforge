# troubleshoot.md

Mistakes this project actually made, written down so the next agent doesn't
repeat them. Every entry happened here. Nothing in this file is hypothetical.

Read the first section before publishing any number.

---

## 1. Measurement traps

This is the dominant category. Six of the bugs below produced *plausible,
publishable results* — a tidy table, a monotonic trend, a clean speedup — and
were only caught by someone asking where a number came from.

The shape is always the same: **two things were compared that were gathered
under conditions differing in a way the table didn't show.**

### 1.1 Never quote a baseline. Measure it.

The README claimed 9.8x over a 0.44 tok/s baseline. That 0.44 came from
`collect_decode.log` — a *tracing* run with forward hooks writing router logits
and hidden states for 16 layers to disk on every token, at a different prompt
length (512 tokens, 41.6 s/sequence). It was never a throughput measurement.

Measured properly through the same harness, same prompt, the baseline is
**0.36 tok/s** and the speedup is 12.1x.

**Rule:** a baseline is a measurement, not a citation. If you can't point at the
command that produced it *in this run*, it isn't a baseline. `ff-serve
--baseline` exists for exactly this; use it rather than reaching for a log.

### 1.2 One pass is not a measurement

The first published capacity sweep read 3.94 → 4.18 → 4.33 tok/s, monotonic in
cache size. Beautiful. It was noise.

Five passes per capacity:

```
128 slots -> median 3.84  (2.87-3.89)
256 slots -> median 4.41  (3.94-4.47)
384 slots -> median 3.53  (3.21-3.63)
```

Spread *across* capacities: 20%. Spread *within* a single capacity: 26%. The
experiment could not resolve its own differences, and the monotonic version was
one sample from each bucket landing in a flattering order.

**Rule:** report a median of N (`--repeats`, N>=5) and the min–max spread beside
it. If within-condition spread exceeds between-condition spread, say so in the
output — don't let the reader infer a trend that isn't there.

**A median printed without its spread is the same bug.** Stage 1b hit this
immediately after the rule above was written. The decode column had a spread
column; the prefill column did not. The first run showed prefill dropping 36%
on the grouped path, with a *convincing mechanism* ready to explain it — a
prefill batch routes to all 64 experts, so the gather copies 805 MB per layer
to save launches the loop path was not wasting. The next run reversed the sign.
Prefill is one short timed region per pass and its real spread is 63.3–94.9
tok/s, which swallows the difference whole.

The available explanation is the trap. A number that confirms a mechanism you
already believe gets less scrutiny, not more. Put a spread on **every**
reported median, not just the one you are currently arguing about.

### 1.3 Don't mix constants from different runs

The README's Q7 row quoted `0.0338m + 0.774`, m\* = 18.0, disk ratio 40.1x from
`bench.log` — an old run on an **HDD scratch path** — in the same table as Q8
numbers from `bench_nvme.log`. All three saved `hardware.json` files agreed the
current answer was m\* = **10.8**. The roadmap threshold ("~17 routed tokens")
had been derived from the stale row and was wrong by 60%.

**Rule:** Q7 and Q8 have to come from the *same* `ff-bench` invocation. When a
table cell has a provenance, put the run in the caption.

### 1.4 Truncating an ordered stream is not sampling

Q5's simulator took `--max-accesses`, applied to a trace ordered by
`(seq_id, pos, layer)`. Truncation therefore handed the simulator the first
5 documents — all `code` domain — instead of a cross-section of 48. With one
domain in the working set, LFU beat LRU. With the real corpus, LRU wins by 12
points. The conclusion had *inverted*.

Fixed with `subsample_sequences()`, which samples whole sequences.

**Rule:** if a trace is sorted by anything, a prefix is a biased sample. Subsample
at the granularity of the unit the sort key groups by.

### 1.5 Isolate phases before averaging them

The first `ff-serve` printed a single hit rate over prefill + decode. Meaningless:
prefill misses are compulsory (a prefill batch touches nearly every expert), so
the blended figure mostly measures the prefill/decode token ratio.

`_run_phase()` now snapshots the counters between phases. Prefill hit 8.3% and
decode 46.8% at the same capacity — averaging those describes nothing.

### 1.6 A leak in the sweep loop reads exactly like a cache-policy finding

In the capacity sweep the previous iteration's `report` (12 GB host store + 3 GB
VRAM pool) stayed alive while the next model loaded. The machine swapped.

What the table showed: capacity 256 had a *better* hit rate (46.8%) and *worse*
throughput (2.33 vs 3.67 tok/s). That is a perfectly respectable-looking
"diminishing returns from cache pressure" result. It was the OS paging.

Fixed with an explicit `del model, report` per iteration, plus
`available_ram_bytes()` reporting and a warning under 2 GB free.

**Rule:** in any sweep, free the previous iteration explicitly and print free RAM
per row. A result where hit rate and throughput move in opposite directions is a
resource bug until proven otherwise.

### 1.7 Environmental contamination

- **Page cache.** A disk benchmark whose scratch file is smaller than RAM
  measures the page cache. Size the file above RAM or drop caches.
- **CPU contention.** A microbenchmark asserting a timing bound will fail under
  load from another process. Don't make wall-clock a correctness assertion in a
  test; measure and report instead.

---

## 2. Correctness traps

### 2.1 Cache capacity is the batch *union*, not `top_k`

`ValueError: Layer 0 routes to 16 experts but the cache holds 5`.

A single decode token needs `top_k` slots. A prefill batch needs the **union over
its tokens** — 24 tokens × top-4 touched all 16 experts. Size for `num_experts`,
or prefill in chunks. The error message in `cache.py` now says this; the test
suite asserts a `top_k`-sized cache is refused on a prefill batch.

### 2.2 accelerate does not release what you `del`

After `del baseline_model; gc.collect(); torch.cuda.empty_cache()`, accelerate
still held the offloaded CPU weights behind its hooks. The subsequent 13.8 GB
load **segfaulted**.

Fix: run the baseline in a subprocess (`_measure_baseline_subprocess`).
**Process exit is the only teardown that is actually guaranteed.**

Related: with `device_map="auto"`, parameters may be **meta tensors**. Real
values live in `module._hf_hook.weights_map`. Reading `.data` off a meta tensor
gives you shape and zero bytes, silently.

### 2.3 Ordering matters for bit-exactness

`CachedMoEBlock` visits `torch.unique(selected_experts).tolist()` (ascending) so
that fp16 `index_add_` accumulation order matches the stock block. Parity is then
max-abs-difference **exactly zero**. Any reordering of expert visits turns that
into "close enough", which is not a test.

### 2.4 Write the manifest last

An early trace collector wrote per-layer gate files before `meta.json`. A failure
near the end left a directory full of valid data that nothing could interpret,
indistinguishable from a partial one. Write the manifest after the payload; its
presence is the completion signal.

### 2.5 `unflatten` vs `view`

The per-projection cache views slice columns out of a `(capacity, numel)` pool.
Those slices are non-contiguous; `view` raises, `unflatten` is correct. Reaching
for `.contiguous()` to make `view` work would silently copy the entire pool.

---

## 3. Platform / tooling traps (Windows)

- **Two shells, two syntaxes.** The Bash tool is Git Bash. A PowerShell
  here-string (`@'...'@`) passed to it leaks a literal `@` — it put one in a commit
  subject. Use `-F - <<'EOF'` in Bash, here-strings only in PowerShell.
- **cp1252 kills completed runs.** A `UnicodeEncodeError` on an em-dash at print
  time discarded a finished multi-minute measurement. Every entry point calls
  `_force_utf8_stdout()`; new scripts must too.
- **PowerShell 5.1 writes UTF-16 by default**, and UTF-8 with a BOM. Pass
  `-Encoding utf8` when writing anything Python will read, and read with
  `utf-8-sig` if PowerShell may have produced it.
- **`PermissionError` at `C:\`.** Scratch paths belong under a real writable
  directory, not the drive root.
- **robocopy mangles HF cache symlinks** without `/SL`, silently producing a copy
  that partly duplicates and partly breaks the blob layout.

---

## 4. Process traps

### 4.1 A subagent's self-report is a claim, not a result

The corpus-generation subagent reported its token counts twice, and was wrong
both times. The corpus was only trusted after running the real OLMoE tokenizer
over it here: 48 samples, 8 per domain, min 529 / max 1704 tokens, zero under 512.

**Rule:** verify a delegated numeric claim with your own tool call before it
enters a commit or a README.

### 4.2 A test can pass against the bug

The first Q5 sampling test asserted on the *number* of sequences returned. The
truncating implementation returned the right count — of the wrong sequences. The
assertion had to be "the result is not a contiguous prefix of the input".

**Rule:** assert on the property that the bug would violate, not on a quantity
the bug happens to preserve.

### 4.3 Don't write a prediction as if it were a finding

`flashforge/runtime/__init__.py` originally asserted the system would be
bandwidth-bound, with arithmetic to back it, before `ff-serve` existed. The sweep
said otherwise: 9x less transfer volume moved decode speed not at all,
`aten::copy_` is 4.95% of decode, and the real cost is **465 GEMM dispatches per
token** (self CPU within 7% of self CUDA — launch-bound). That reversed the
roadmap: grouped GEMM, filed as Stage 2, is the actual Stage 1 bottleneck;
prefetch is worth at most 5% and moved after it.

**Rule:** docstrings and READMEs state what was measured. Predictions get labelled
as predictions, in the future tense, with the experiment that would settle them.

### 4.4 A tolerance in a unit test is not a tolerance in the model

Stage 1b's grouped path is not bit-exact — one `index_add_` over every expert
has no defined accumulation order. The test suite held it to `2e-19` in fp32 on
a toy block, which is a fine test and answers the wrong question. In fp16 on a
7B model one rounding difference can flip an argmax, and from that token on the
two paths generate different text.

So `ff-serve` decodes greedily on both paths and reports the first token id that
differs. It was identical for 65 tokens, and *that* is why the grouped path is
the default. Collect the ids as device tensors and convert after the timed
region — an `.item()` per step forces a synchronize and makes the harness
measure itself.

**Rule:** when an optimisation trades exactness, the acceptance test has to run
at the real dtype, the real scale, and on the observable the user actually sees.

---

## Checklist before publishing a number

1. Was the baseline measured in this run, by the same harness, same prompt?
2. Is it a median of >=5 passes, with the spread printed next to it?
3. Do all the constants in this table come from one run?
4. Are prefill and decode reported separately?
5. Was free RAM/VRAM checked between sweep iterations?
6. If it came from a subagent or an old log — did you re-run it yourself?
