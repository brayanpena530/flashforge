"""Score int8 experts against fp16 on one model load.

    uv run python tools/int8_ab.py

WHY THIS EXISTS RATHER THAN `ff-serve --int8-capacity`
-----------------------------------------------------
`ff-serve` sweeps capacities by reloading the checkpoint once per capacity,
because `install_expert_cache` moves the experts out and leaves a hollow model
behind. That is correct and it does not fit on this machine when the arms are
different dtypes. Three attempts failed three different ways, all memory:

* the fp16 arm holds a 12 GB store and runs with ~1 GB of host RAM free, so the
  second arm's 7 GB pin budget is simply not there. It got 3.00 GB against the
  control's 6.75 GB and would have measured slower for a reason that has
  nothing to do with quantisation — a confound, not a crash, and the dangerous
  kind because the table still prints.
* pinned host memory has its own caching allocator that `empty_cache()` does
  not touch, so the first arm's page-locked GB stay spent.
* both int8 arms then died inside the timed region with a raw
  `CUDA error: out of memory` while holding *more* free VRAM than the fp16 arm
  that had just succeeded, which is accumulated sweep state rather than
  anything about int8 — a fresh single-arm process runs the same config fine.

The fix is to stop reloading. Quantisation rewrites the store in place, so one
load can serve every arm: time fp16, convert, re-pin, time int8. That removes
the 13.8 GB checkpoint transient that was the actual constraint, and it makes
the comparison stronger rather than weaker — same process, same weights, same
page cache, one variable.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flashforge.cli import (  # noqa: E402
    _corpus_input_ids,
    _force_utf8_stdout,
    _median,
    _permutation_p,
    _run_phase,
    _setup_logging,
)
from flashforge.models import DEFAULT_MODEL, load_model  # noqa: E402
from flashforge.runtime import ExpertCache, QuantSpec, install_expert_cache  # noqa: E402

PROMPT_TOKENS = 128
GEN_TOKENS = 64
REPEATS = 9
WARMUP = 1
PIN_GB = 7.0

# Accuracy is no longer measured here. It was, and the reading was not
# resolvable: this tool scored on the *grouped* path, whose `index_add_` has no
# defined summation order, so two runs of identical arithmetic disagreed by 0.41
# points — nearly twice the 0.233-point gap to the bar. And `DIVERGENCE_DOCS`
# was set to 48 to push n to ~5,300, which it never did, because
# `_corpus_input_ids` reads the built-in 24-prompt starter set and a slice of a
# short list is silently short. n was 2,545.
#
# `tools/int8_accuracy.py` fixes both — loop path, real 512-token corpus, 24,576
# positions, a floor control that reads exactly 100.000% — and settles it:
# **98.767% +/- 0.070%, failing the 99% bar by 0.233 points at 3.3 sigma**.
# Capacity cannot change that number, so there is nothing for a capacity sweep
# to add. This tool keeps the question it can answer: throughput.
DIVERGENCE_DOCS = 0

# 238 fp16 slots is 3.0 GB, the shipping operating point. The int8 ladder exists
# because the first measurement inverted the premise of the stage: 357 int8
# slots had a better hit rate (63.5% vs 47.3%), moved fewer bytes (0.392 vs
# 0.566 GB/token), blocked less on fill (37.2 vs 53.3 ms) — and ran 8% slower
# than 238. Every intermediate metric improved while throughput fell, which is
# the signature of troubleshoot.md 1.10: past some pool size the driver pages
# the slot pool to host memory and the timings measure that. So sweep the
# capacity and find the cliff rather than assuming either endpoint.
FP16_SLOTS = 238
INT8_SLOTS = (200, 238, 270, 300, 330, 357)
# `None` disables logit collection entirely; see the note on DIVERGENCE_DOCS.
SCORED_SLOTS = None
SPEC = QuantSpec(projections=("gate_proj", "up_proj"), group_size=0)


def _arm(model, report, cache, input_ids, corpus, label: str, *, score: bool = False) -> dict:
    """Warm up, time `REPEATS` passes, optionally collect teacher-forced logits."""
    for block in report.blocks:
        block.cache = cache
    stats = cache.stats
    cache.time_fills = True

    for _ in range(WARMUP):
        _run_phase(model, input_ids, min(4, GEN_TOKENS))

    # Read after the warmup, so the gather and activation buffers the timed
    # region will need are already allocated. This is the column the capacity
    # ladder exists to expose: throughput tracked it, not the hit rate.
    torch.cuda.empty_cache()
    vram_free = torch.cuda.mem_get_info()[0] / (1 << 30)

    prefill, decode, fill_ms = [], [], []
    bytes_fetched = hits = misses = 0
    for _ in range(REPEATS):
        stats.reset()
        timing = _run_phase(model, input_ids, GEN_TOKENS, stats, cache)
        if timing["prefill_s"]:
            prefill.append(timing["prefill_tokens"] / timing["prefill_s"])
        if timing["decode_s"]:
            decode.append(timing["decode_tokens"] / timing["decode_s"])
        d_hits, d_misses, d_bytes = timing["decode_counts"]
        hits += d_hits
        misses += d_misses
        bytes_fetched += d_bytes
        fill_ms.append(timing["decode_fill_ms"] / max(1, timing["decode_tokens"]))

    # Untimed, and after the timed region on purpose: a 12-document sweep at
    # fp32 logits is hundreds of MB of host memory, and doing it between passes
    # would put an allocator spike inside the measurement.
    logits = []
    if score:
        with torch.no_grad():
            for ids in corpus:
                out = model(ids, use_cache=False)
                # fp16 on the host, not fp32: 48 documents of 50k-wide logits is
                # 1.2 GB per arm at fp32 and this machine is already holding an
                # 8 GB store. The comparison casts back up before the softmax,
                # so the KL is computed in fp32 either way.
                logits.append(out.logits[0].detach().to("cpu", torch.float16))
                del out

    cache.time_fills = False
    lookups = hits + misses
    return {
        "label": label,
        "slots": cache.capacity,
        "vram_gb": cache.bytes_resident / (1 << 30),
        "mb_per_expert": cache.store.shape.nbytes / 1e6,
        "prefill_tok_s": _median(prefill),
        "decode_tok_s": _median(decode),
        "decode_lo": min(decode) if decode else 0.0,
        "decode_hi": max(decode) if decode else 0.0,
        "decode_samples": decode,
        "vram_free_gb": vram_free,
        "hit_rate": hits / lookups if lookups else 0.0,
        "gb_per_token": bytes_fetched / 1e9 / max(1, REPEATS * GEN_TOKENS),
        "fill_ms": _median(fill_ms),
        "logits": logits,
    }


def main() -> int:
    _force_utf8_stdout()
    _setup_logging(verbose=True)
    if not torch.cuda.is_available():
        print("This needs a CUDA device; the whole point is the PCIe crossing.")
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    prompt = "The history of computing is" * 64
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :PROMPT_TOKENS]
    input_ids = input_ids.to("cuda")
    corpus = list(_corpus_input_ids(tokenizer, DIVERGENCE_DOCS, PROMPT_TOKENS))

    model, _ = load_model(DEFAULT_MODEL, dtype="float16", device_map=None)
    report = install_expert_cache(
        model, capacity=FP16_SLOTS, device="cuda", pin_gb=PIN_GB, grouped=True
    )
    print(report.describe())

    rows = [_arm(model, report, report.cache, input_ids, corpus, "fp16")]
    print(f"  fp16 {rows[0]['decode_tok_s']:.2f} tok/s, "
          f"hit {rows[0]['hit_rate']:.1%}, {rows[0]['gb_per_token']:.3f} GB/token, "
          f"{rows[0]['vram_free_gb']:.2f} GB VRAM free")

    # The pool goes first: quantising reallocates every row, and the cache built
    # against the old geometry would still accept the copy_ and serve half an
    # expert spliced to half a neighbour. `acquire` refuses it, but freeing the
    # 3 GB here is also what makes room for the next pool.
    store = report.store
    report.cache.release()
    report.cache = None
    for block in report.blocks:
        block.cache = None
    gc.collect()
    torch.cuda.empty_cache()

    error = store.quantize_int8(SPEC)
    # Quantising freed 12 GB of pinned rows into PyTorch's *host* caching
    # allocator, which `empty_cache()` does not drain. Without this the repin
    # below silently gets whatever is left and the int8 arm runs at half the
    # control's pin coverage.
    gc.collect()
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
    pinned_layers = store.repin(PIN_GB)
    gc.collect()
    print(f"\nquantised: {100 * error['rel_rms_error']:.2f}% relative RMS weight "
          f"error, {store.shape.nbytes / 1e6:.2f} MB per expert, "
          f"{pinned_layers} of {len(store.layers)} layers pinned "
          f"({store.pinned_bytes / (1 << 30):.2f} GB)")

    for slots in INT8_SLOTS:
        # The scored capacity runs twice. The second pass is not a repeat of the
        # throughput measurement — it is the only way to know how much of an
        # agreement difference is the grouped path's own nondeterminism before
        # attributing any of it to quantisation.
        passes = 2 if slots == SCORED_SLOTS else 1
        for attempt in range(passes):
            cache = ExpertCache(store, slots, device="cuda", dtype=store.shape.dtype)
            label = f"int8-{slots}" + ("'" if attempt else "")
            row = _arm(
                model, report, cache, input_ids, corpus, label,
            )
            rows.append(row)
            print(f"  {label}: {row['decode_tok_s']:.2f} tok/s, "
                  f"hit {row['hit_rate']:.1%}, {row['gb_per_token']:.3f} GB/token, "
                  f"{row['vram_free_gb']:.2f} GB VRAM free")
            cache.release()
            del cache
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 112)
    print(f"{'arm':>10} {'slots':>6} {'pool GB':>8} {'free GB':>8} {'prefill':>8} "
          f"{'decode':>7} {'spread':>13} {'hit':>7} {'GB/tok':>7} {'fill ms':>8}")
    for row in rows:
        print(f"{row['label']:>10} {row['slots']:>6} {row['vram_gb']:>8.2f} "
              f"{row['vram_free_gb']:>8.2f} {row['prefill_tok_s']:>8.1f} "
              f"{row['decode_tok_s']:>7.2f} "
              f"{row['decode_lo']:.2f}-{row['decode_hi']:<8.2f} "
              f"{row['hit_rate']:>6.1%} {row['gb_per_token']:>7.3f} "
              f"{row['fill_ms']:>8.1f}")

    base = rows[0]
    print()
    for row in rows[1:]:
        change = row["decode_tok_s"] / base["decode_tok_s"] - 1.0
        p = _permutation_p(base["decode_samples"], row["decode_samples"])
        verdict = (
            "too few passes to test" if p is None
            else f"real (p={p:.3f}, {REPEATS} passes each)" if p < 0.05
            else f"not distinguishable from noise (p={p:.2f})"
        )
        print(f"[int8] {row['label']:>10}: {base['decode_tok_s']:.2f} -> "
              f"{row['decode_tok_s']:.2f} tok/s, {change:+5.0%}. {verdict}")

    print("\n[bar]  accuracy is not measured here. Run "
          "tools/int8_accuracy.py — it scores on the deterministic loop path "
          "against the 512-token corpus, where the instrument's own floor is "
          "exactly 100.000% and the answer is 98.767% +/- 0.070%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
