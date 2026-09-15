# troubleshoot.md

Mistakes this project actually made, written down so the next agent doesn't
repeat them. Every entry happened here. Nothing in this file is hypothetical.

Read the first section before publishing any number.

---

## 1. Measurement traps

This is the dominant category. Most of the bugs below produced *plausible,
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

**But do not make the spread the verdict.** The rule above — "call it real if
the change exceeds the min–max spread" — is right for *describing* one
measurement and wrong for *deciding* between two, because min–max **grows with
sample count**. Every extra pass raises the bar. Stage 1c ran the same
prefetch-vs-grouped comparison three times, got "inside the spread — no
finding" every time, and could never separate *the effect is absent* from *the
test cannot see it*.

`ff-serve` now runs a two-sided permutation test on the difference of medians:
could relabelling this pool of passes produce this gap? That tightens as passes
accumulate, which is the behaviour you want from a stopping rule. It assumes
only that passes are exchangeable under the null — which is why the paths are
interleaved on one loaded model and one warm cache rather than timed in
separate invocations.

**Rule:** spread describes, a test decides. If a comparison keeps coming back
"inside the noise", check whether your noise rule is capable of ever saying
anything else.

**And the baseline is a sample too.** Within an hour of writing the rule above,
Stage 1c ran a five-pass sweep of prefetch budgets and found a clean inverted-U:
k=8 at 5.80, k=4 at **6.50**, k=2 at 6.22, against 5.97 for no prefetch. It
matched a mechanism that had just been predicted out loud — fewer speculative
fetches, higher precision, less wasted bandwidth — and every supporting column
moved the right way. It was reported as a confirmation.

Nine passes reversed it. The no-prefetch baseline read **6.86**, with a
5.77–6.90 spread; the five-pass run had caught it low. Prefetch never beat it at
any budget, and the full-budget version is a *measured regression* (−14%,
p=0.010).

Note what was and was not noisy: the treatments were fine. The **baseline** was
the unstable one, and it is the number least likely to be re-examined, because
it is not the thing being argued about. Section 1.2 already said an available
explanation gets less scrutiny rather than more. It happens to the person who
wrote that sentence.

**Rule:** run the control at the same power as the treatment, and when a result
confirms a mechanism you predicted in advance, that is the moment to add passes
— not the moment to write it up.

### 1.3 Don't mix constants from different runs

The README's Q7 row quoted `0.0338m + 0.774`, m\* = 18.0, disk ratio 40.1x from
`bench.log` — an old run on an **HDD scratch path** — in the same table as Q8
numbers from `bench_nvme.log`. All three saved `hardware.json` files agreed the
current answer was m\* = **10.8**. The roadmap threshold ("~17 routed tokens")
had been derived from the stale row and was wrong by 60%.

**Rule:** Q7 and Q8 have to come from the *same* `ff-bench` invocation. When a
table cell has a provenance, put the run in the caption.

### 1.4 A profile taken at one capacity does not describe another

Stage 1 profiled decode, found `aten::copy_` at 4.95%, and concluded transfers
were nearly free. That reordered the whole roadmap: grouped GEMM moved up,
prefetch moved down to "worth at most ~5%", and the README said so.

The profile was taken at **384 slots**. The runtime ships at **256**, where
transfer volume is seven times higher (0.859 vs 0.120 GB/token). Measured
there, fill is **71% of decode**, not 5%. The prefetch ceiling was not 5%; it
was +244%.

Worse, the 384-slot row should never have been used for anything — see 1.10.

**Rule:** a breakdown is a property of an operating point, not of a system. If
the number is going to be used to rank work, measure it at the configuration
the work will ship in, and put the configuration next to the number.

### 1.5 Instrument every path, or the column means two things

The fill timer above wrapped `ExpertCache._fill`, which is the *demand* path.
When prefetch moved most fills to a side stream, the column dropped from 126.3
to 50.5 ms/token and read as a 60% improvement. It was not measuring an
improvement; it was measuring less of the program.

The fix is two columns — demand ms (blocking, on the critical path) and
speculative ms (side stream, supposed to be hidden) — because adding them
would make better overlap look like a regression, and reporting only the first
makes moving work out of frame look like removing it.

**Rule:** when an optimisation *relocates* work, a counter scoped to the old
location will show the win whether or not the win happened. Scope counters to
the work, not to the function.

### 1.6 A default is a claim, and nobody audits the defaults

Every throughput number this project published before Stage 1d ran with a fully
pageable expert store. `--pin-gb` existed, defaulted to 0, and the store printed
`0.00 GB pinned (0% of it, 0 layers)` on every single run — in a line that read
as configuration rather than as a finding.

Pinning 88% of it is **+38% decode, p=0.003**, with bytes per token and hit rate
unchanged. It is the largest single lever found in the project, larger than
either optimisation that was actually designed and built, and it was a flag.

The tell was available from Stage 0: Q7 measured pinned transfer at 10.4 GB/s,
the runtime was sustaining 6.08 GB/s on the fill path, and nothing compared the
two numbers because they lived in different tools.

**Rule:** before optimising, list the knobs already in the code and what they are
set to. A default that has never been varied has never been measured, and "it
defaults to off" is not the same as "off is right". Where a microbenchmark and a
runtime measure the same physical thing, make one of them print the ratio.

### 1.7 An advisory that fires on a sweep of its own assumption

`ff-serve` grew a check that flags a saturated link: if sustained bandwidth is
flatter than throughput across the rows, then throughput is bandwidth over
bytes-per-token and only byte reduction can help. Correct reasoning — and it
fired on a **pin sweep**, which varies bandwidth deliberately, off a 31%-vs-31%
comparison, printing "the link is the constraint at ~5.2 GB/s" under a table
demonstrating the opposite.

The rule was `if gbs < tps`. Two quantities that vary by the same amount satisfy
it half the time by chance. It now requires bandwidth to be at least twice as
stable as throughput, which is what "roughly constant" has to mean if the
conclusion is going to follow.

**Rule:** an automated verdict inherits every assumption of the argument behind
it. Write down what has to be *constant* for the conclusion to hold, and make
the check refuse to fire when that thing is the variable being swept.

### 1.8 Truncating an ordered stream is not sampling

Q5's simulator took `--max-accesses`, applied to a trace ordered by
`(seq_id, pos, layer)`. Truncation therefore handed the simulator the first
5 documents — all `code` domain — instead of a cross-section of 48. With one
domain in the working set, LFU beat LRU. With the real corpus, LRU wins by 12
points. The conclusion had *inverted*.

Fixed with `subsample_sequences()`, which samples whole sequences.

**Rule:** if a trace is sorted by anything, a prefix is a biased sample. Subsample
at the granularity of the unit the sort key groups by.

### 1.9 Isolate phases before averaging them

The first `ff-serve` printed a single hit rate over prefill + decode. Meaningless:
prefill misses are compulsory (a prefill batch touches nearly every expert), so
the blended figure mostly measures the prefill/decode token ratio.

`_run_phase()` now snapshots the counters between phases. Prefill hit 8.3% and
decode 46.8% at the same capacity — averaging those describes nothing.

### 1.10 A resource bug reads exactly like a cache-policy finding

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

**The VRAM version of this is worse, because it does not raise.** At 384 slots
the cache is 5.39 GB resident on a 6 GB card. On Windows that does not OOM —
WDDM silently backs the overflow with host memory, so every read of a "resident"
expert crosses PCIe again. Decode: **0.83 tok/s at a 92.7% hit rate**, against
5.53 tok/s at 46.9% with 256 slots. Hit rate doubled, throughput fell 6.7x. The
same signature, one tier down, and it had been sitting in the published capacity
sweep as the 384-slot row — the row the "transfers are nearly free" profile was
taken from (1.4).

Two things follow. Print VRAM headroom per row, not just host RAM. And call
`torch.cuda.empty_cache()` before `mem_get_info()`: PyTorch's caching allocator
counts as *used* at the driver level, so without it the check reads 0.00 GB free
at every capacity including the ones with 3.6 GB genuinely spare. A warning that
fires on every row is not a warning.

**Also: rows inside one sweep are not comparable to each other.** Even with the
explicit `del`, free host RAM fell 12.4 → 7.5 → 5.8 GB across a 128/256/320
sweep, and 256 slots measured **2.77 tok/s** there against **5.53** in a run of
its own. Same capacity, same code, same machine — different amount of the
machine left. Row one is the only row measured on a clean box. `ff-serve` now
warns when free RAM has drifted more than 1 GB from row one's, but the real fix
is one capacity per invocation.

### 1.11 Environmental contamination

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

### 4.4 Removing a stall is not the same as removing the cost

Stage 1c's prefetcher did exactly what it was designed to do. Blocking fill fell
from 100.5 to 32.7 ms/token, decode hit rate went 46.8% -> 81.8%, and the
predictor ran at 78% precision in the runtime. Every intermediate metric said it
worked.

Decode throughput moved +4%, inside a 20% spread.

Sixty-eight milliseconds per token left the critical path and did not come back
as tokens, because prefetch also moved **24% more bytes** over the link that was
already the bottleneck — 22% of its guesses are wrong, and it pays full price
for them. Overlapping a transfer helps when ordering is the problem. It does not
help when bandwidth is the problem; it makes bandwidth worse.

**Rule:** an intermediate metric moving the right way is evidence the mechanism
is wired up, not evidence the change is worth having. Name the end-to-end number
before you start, and if the mechanism improves while it does not, look for the
resource the change is *spending* rather than the one it is saving.

**How that one ended.** Spending less did not rescue it either — truncating the
speculation to the four most confident predictions raised precision to 92.8% and
brought bandwidth back to within 3% of the demand path, and it was still not
distinguishable from no prefetch at all (p=0.09). The answer came from a column
nobody had thought to print:

```
            tok/s  ×  GB/token  =  GB/s sustained
grouped      6.86     0.846        5.80
pf-all       5.92     1.045        6.19
pf-k4        6.36     0.872        5.55
```

Throughput varied 8%, bytes varied 11%, and the product was flat. The link was
saturated, so `tok/s = bandwidth / bytes-per-token` and *any* scheme that adds
bytes loses regardless of how well it overlaps them.

**Rule:** when throughput and the resource it consumes move in opposite
directions, multiply them. If the product is flatter than either factor, you
have found the bottleneck, and every optimisation that does not reduce that
resource is already dead — including the one you are holding.

### 4.5 Count the synchronizations, not just the milliseconds

Between them the router and the prefetcher were doing **three** device-to-host
copies per layer — `unique(...).tolist()`, `counts.tolist()`, and the predicted
set — sixteen times per token. Each is tiny and each one drains the pipeline.

`torch.bincount(..., minlength=num_experts)` returns the routed set and the
group sizes in one fixed-size vector, so the prediction concatenates onto it and
the whole layer costs one copy. That took the loop path's blocking fill from
130.4 to 98.4 ms/token — a bigger change than most of what was being measured
around it, from code that was never the subject of a benchmark.

Walking the count vector host-side also yields ascending expert order, which is
what `unique` gave and what fp16 `index_add_` needs to stay bit-exact against
the stock block. That property had to survive the rewrite, and the parity test
asserting a difference of *exactly zero* is what proved it did.

**Rule:** on a hot path, `.tolist()`, `.item()` and `.cpu()` are the expensive
operations regardless of how little data they move. Count them per layer before
optimising anything measured in milliseconds.

### 4.6 A tolerance in a unit test is not a tolerance in the model

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
5. Was free RAM *and* VRAM checked per row, VRAM after an `empty_cache()`?
6. If it came from a subagent or an old log — did you re-run it yourself?
7. If it is a breakdown or a profile, was it taken at the operating point the
   conclusion will be applied to?
8. If the change *moved* work rather than removing it, does the counter still
   cover both places it can now be?
9. Is every row in this table from its own invocation, or did a sweep hand you
   row three on a machine that row one had already used up?
