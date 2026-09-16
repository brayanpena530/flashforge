"""Is the 26 ms/token at 270 int8 slots a property of 270, or of running fourth?

    uv run python tools/slot_cliff.py

THE OPEN QUESTION
-----------------
Stage 1e-2's int8 capacity ladder has a hole in it. Two adjacent arms:

    slots   decode t/s   hit    GB/token   fill ms
      238         8.32   47.3%     0.566      54.2
      270         6.86   48.1%     0.557      53.4

Identical bytes across the link, identical time blocked on fill, a slightly
*better* hit rate — and 18% less throughput, which is 25.6 ms per token going
somewhere that is neither transfer nor cache behaviour. The README records it as
an open question rather than a mechanism, which was honest and is not an answer.
It sits directly under the shipped `+17% at 238 slots` headline: if 238 is a
spike rather than a peak, the operating point is wrong.

THE FIRST HYPOTHESIS IS NOT ABOUT CAPACITY
------------------------------------------
`int8_ab.py` runs seven arms in one process, in ascending order, and 270 runs
fourth — right after 238 has been built, timed, released, and built again. The
ladder therefore confounds capacity with position in the sequence, and this
project has already been bitten by exactly that: checklist item 9 in
troubleshoot.md ("did a sweep hand you row three on a machine that row one had
already used up?") exists because three int8 sweeps died on accumulated state,
and the dangerous failure there was the one that *did not* crash.

So before explaining 270, establish that 270 needs explaining. **Alternate** the
two capacities — 238, 270, 238, 270, 238, 270 — in one process. Then:

* if the gap survives alternation, it is capacity, and the next question is
  where the 26 ms goes;
* if it vanishes, the ladder measured its own sequence and the +17% peak has to
  be re-read;
* if throughput drifts monotonically with position regardless of capacity, the
  process degrades as it runs and every multi-arm number in this repo inherits
  it.

These are distinguishable because position and capacity are crossed here and
were nested in the original.

WHAT IS INSTRUMENTED THAT WAS NOT
---------------------------------
`torch.cuda.memory_stats()["num_alloc_retries"]`. A retry is the caching
allocator failing to serve a request, synchronizing, freeing cached blocks and
trying again — a real stall, invisible to the fill timer because it is not a
fill, and invisible to the hit rate because it is not a lookup. It is the
leading candidate for 26 ms that no existing counter covers, and it costs one
dictionary read per arm to find out.
"""
from __future__ import annotations

import gc

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flashforge.cli import (  # noqa: E402
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
SPEC = QuantSpec(projections=("gate_proj", "up_proj"), group_size=0)

# (capacity, widen) pairs. The first pass of this tool ran
# (238, 270) x 3 alternating to cross capacity with sweep position, and found
# capacity: -17.4% at p<0.0001, with position worth -0.7%. That question is
# closed, so the arms now spend their time on the one that replaced it —
# `tools/gather_cliff.py` located the step at exactly 256 slots and identified
# it as `index_select` falling to 64-bit index math, and `ExpertCache._widen`
# fixes it by selecting through an int64 view of the same bytes.
#
# `widen=False` restores the old kernel by pointing the gather back at the raw
# pool, so before and after are measured in one process on one set of weights.
# 476 is included because it is what the fix is *for*: past the old cliff, where
# the hit rate keeps climbing and the gather no longer charges for it.
ORDER = (
    (238, False), (238, True),
    (270, False), (270, True),
    (357, False), (357, True),
    (476, True),
)


def _arm(model, blocks, cache, input_ids, position: int) -> dict:
    for block in blocks:
        block.cache = cache
    stats = cache.stats
    cache.time_fills = True

    for _ in range(WARMUP):
        _run_phase(model, input_ids, min(4, GEN_TOKENS))

    torch.cuda.empty_cache()
    vram_free = torch.cuda.mem_get_info()[0] / (1 << 30)
    # Read after the warmup so the arm is charged only for retries it causes
    # inside its own timed region, not for the pool allocation that preceded it.
    retries_before = torch.cuda.memory_stats().get("num_alloc_retries", 0)

    decode, fill_ms = [], []
    hits = misses = bytes_fetched = 0
    for _ in range(REPEATS):
        stats.reset()
        timing = _run_phase(model, input_ids, GEN_TOKENS, stats, cache)
        if timing["decode_s"]:
            decode.append(timing["decode_tokens"] / timing["decode_s"])
        d_hits, d_misses, d_bytes = timing["decode_counts"]
        hits += d_hits
        misses += d_misses
        bytes_fetched += d_bytes
        fill_ms.append(timing["decode_fill_ms"] / max(1, timing["decode_tokens"]))

    cache.time_fills = False
    retries = torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries_before
    lookups = hits + misses
    rate = _median(decode)
    return {
        "slots": cache.capacity,
        "position": position,
        "tok_s": rate,
        "ms_per_token": 1000.0 / rate if rate else float("nan"),
        "samples": decode,
        "lo": min(decode),
        "hi": max(decode),
        "hit_rate": hits / lookups if lookups else 0.0,
        "gb_per_token": bytes_fetched / 1e9 / max(1, REPEATS * GEN_TOKENS),
        "fill_ms": _median(fill_ms),
        "retries": retries,
        "vram_free_gb": vram_free,
    }


def main() -> int:
    _force_utf8_stdout()
    _setup_logging(verbose=True)
    if not torch.cuda.is_available():
        print("This needs a CUDA device; the question is about the PCIe crossing.")
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    prompt = "The history of computing is" * 64
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :PROMPT_TOKENS]
    input_ids = input_ids.to("cuda")

    # `--fp16` skips quantisation and re-measures the *baseline* the int8 win is
    # quoted against. Widening is not an int8 feature — an fp16 pool divides
    # into int64 just as well, and while it never crosses INT32_MAX at these
    # capacities it still gains the vectorised loads. If the baseline moved, the
    # "+17% for int8" on record was measured against a slower fp16 than the one
    # that now ships, and the honest comparison is widened against widened.
    #
    # Its own process, deliberately. Pinning an fp16 store and *then* quantising
    # is how the first int8 sweep died: `repin` page-locks 12 GB of rows that
    # are about to be discarded, and the int8 copies then have nowhere to go.
    # patch.py carries the same note. Two runs is cheaper than that failure.
    fp16_only = "--fp16" in sys.argv
    order = ((238, False), (238, True), (200, True)) if fp16_only else ORDER

    model, _ = load_model(DEFAULT_MODEL, dtype="float16", device_map=None)
    # Built at the smaller capacity and immediately released: `install` needs a
    # cache to hand the blocks, but every timed pool below is constructed after
    # quantisation, from the int8 row geometry, exactly like its neighbours. No
    # arm inherits a pool another arm sized.
    report = install_expert_cache(
        model, capacity=min(s for s, _ in order), device="cuda",
        pin_gb=PIN_GB if fp16_only else 0.0, grouped=True,
    )
    store = report.store
    report.cache.release()
    for block in report.blocks:
        block.cache = None
    gc.collect()
    torch.cuda.empty_cache()

    if fp16_only:
        print(f"fp16, unquantised: {store.shape.nbytes / 1e6:.2f} MB per expert, "
              f"{store.pinned_bytes / (1 << 30):.2f} GB pinned\n")
    else:
        error = store.quantize_int8(SPEC)
        gc.collect()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
        pinned = store.repin(PIN_GB)
        gc.collect()
        print(f"quantised: {100 * error['rel_rms_error']:.2f}% relative RMS weight "
              f"error, {store.shape.nbytes / 1e6:.2f} MB per expert, "
              f"{pinned} of {len(store.layers)} layers pinned "
              f"({store.pinned_bytes / (1 << 30):.2f} GB)\n")

    rows = []
    for position, (slots, widen) in enumerate(order, start=1):
        try:
            cache = ExpertCache(store, slots, device="cuda", dtype=store.shape.dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  #{position} {slots:>3} slots: pool does not fit beside the "
                  "residual model on this card")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if not widen:
            # Point the gather back at the raw pool. This is the pre-fix kernel
            # exactly: same bytes, same rows, one element per byte.
            cache._gather_pool = cache._slots
        row = _arm(model, report.blocks, cache, input_ids, position)
        row["widen"] = widen
        rows.append(row)
        print(f"  #{position} {slots:>3} slots {'widened' if widen else 'raw    '}: "
              f"{row['tok_s']:.2f} tok/s "
              f"({row['ms_per_token']:.1f} ms/token), hit {row['hit_rate']:.1%}, "
              f"{row['gb_per_token']:.3f} GB/token, fill {row['fill_ms']:.1f} ms, "
              f"non-fill {row['ms_per_token'] - row['fill_ms']:.1f} ms, "
              f"{row['retries']} retries")
        cache.release()
        del cache
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 104)
    print(f"{'#':>2} {'slots':>6} {'gather':>8} {'tok/s':>7} {'ms/tok':>7} "
          f"{'fill':>7} {'non-fill':>9} {'hit':>7} {'GB/tok':>7} {'free GB':>8}")
    for row in rows:
        print(f"{row['position']:>2} {row['slots']:>6} "
              f"{'widened' if row['widen'] else 'raw':>8} {row['tok_s']:>7.2f} "
              f"{row['ms_per_token']:>7.1f} {row['fill_ms']:>7.1f} "
              f"{row['ms_per_token'] - row['fill_ms']:>9.1f} "
              f"{row['hit_rate']:>6.1%} {row['gb_per_token']:>7.3f} "
              f"{row['vram_free_gb']:>8.2f}")

    # -- what widening is worth, capacity by capacity -----------------------
    print("\n" + "=" * 104)
    print("THE FIX, paired within each capacity so nothing else can move")
    paired = {}
    for row in rows:
        paired.setdefault(row["slots"], {})[row["widen"]] = row
    for slots in sorted(paired):
        arms = paired[slots]
        if len(arms) < 2:
            continue
        raw, wide = arms[False], arms[True]
        p = _permutation_p(raw["samples"], wide["samples"])
        saved = raw["ms_per_token"] - wide["ms_per_token"]
        print(f"  {slots:>3} slots: {raw['tok_s']:.2f} -> {wide['tok_s']:.2f} tok/s "
              f"({wide['tok_s'] / raw['tok_s'] - 1:+.1%}), "
              f"non-fill {raw['ms_per_token'] - raw['fill_ms']:.1f} -> "
              f"{wide['ms_per_token'] - wide['fill_ms']:.1f} ms, "
              f"{saved:+.1f} ms/token, "
              + (f"p={p:.4f}" if p is not None else "untestable"))

    # -- the capacity curve, now that the cliff is out of it ----------------
    print("\nTHE CURVE, widened arms only. The cliff was the reason this was")
    print("non-monotonic, so if it is still non-monotonic something else is wrong.")
    widened = sorted((r for r in rows if r["widen"]), key=lambda r: r["slots"])
    for row in widened:
        print(f"  {row['slots']:>3} slots: {row['tok_s']:>5.2f} tok/s, "
              f"non-fill {row['ms_per_token'] - row['fill_ms']:>5.1f} ms, "
              f"hit {row['hit_rate']:>5.1%}, {row['gb_per_token']:.3f} GB/token")
    # An arm that has exhausted VRAM is measuring the driver paging the pool to
    # host memory, which is troubleshoot.md 1.10 and a different phenomenon from
    # the one this tool is about. Its signature is unmistakable and worth
    # printing rather than hiding: every intermediate metric at its best — hit
    # rate highest, bytes per token lowest, fill time lowest — and throughput
    # off a cliff. Judge flatness on the arms that still had room.
    housed = [r for r in widened if r["vram_free_gb"] > 0.2]
    starved = [r for r in widened if r["vram_free_gb"] <= 0.2]
    best = max(housed or widened, key=lambda r: r["tok_s"])
    spans = [r["ms_per_token"] - r["fill_ms"] for r in housed]
    flat = max(spans) - min(spans) if spans else float("nan")
    print(f"\n  best: {best['slots']} slots at {best['tok_s']:.2f} tok/s")
    print(f"  non-fill time varies {flat:.1f} ms across the {len(housed)} arms that "
          "fit" + (" -- flat, which is what a cliff-free gather looks like."
                   if flat < 8.0 else " -- still stepped. Something else is in there."))
    for row in starved:
        print(f"  {row['slots']} slots excluded: {row['vram_free_gb']:.2f} GB VRAM "
              f"free, hit {row['hit_rate']:.1%}, {row['gb_per_token']:.3f} GB/token, "
              f"fill {row['fill_ms']:.1f} ms — every metric ideal and "
              f"{row['tok_s']:.2f} tok/s. That is the driver paging the pool, not "
              "the gather.")

    baseline = paired.get(238, {}).get(False)
    if baseline:
        print(f"\n  against the shipped operating point (238 raw, "
              f"{baseline['tok_s']:.2f} tok/s): "
              f"{best['tok_s'] / baseline['tok_s'] - 1:+.1%}")
    total_retries = sum(r["retries"] for r in rows)
    print(f"  allocator retries inside timed regions, all arms: {total_retries}"
          + ("" if total_retries else "  (so no allocator stall is hiding here)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
