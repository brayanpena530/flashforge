"""Settle the int8 accuracy bar with an instrument that can resolve it.

    uv run python tools/make_corpus.py corpus.jsonl
    uv run python tools/int8_accuracy.py

WHY THIS EXISTS RATHER THAN `tools/int8_ab.py`
----------------------------------------------
Stage 1e-2 scored int8 experts at **98.76% +/- 0.15%** against a pre-committed
99% bar and recorded the bar as missed. That verdict was not measurable. The
gap to the bar is 0.24 points, and the instrument that produced it has two
independent error sources, each of them the same size or larger:

* **The path.** Agreement was collected with `grouped=True`. The grouped path
  sorts every (token, expert) pair and accumulates with one `index_add_` whose
  indices collide, and CUDA gives `index_add_` no defined summation order. Two
  runs of *identical arithmetic on identical weights* therefore agree only to
  99.72% — a **0.28-point floor**, larger than the gap being measured. That
  floor is a property of the accumulator, not of quantisation.

  The loop path has no such floor. It also calls `index_add_`, but once per
  expert over that expert's own token list, where the indices are unique: no
  collisions, no ordering freedom, bit-reproducible run to run. Timing needs
  the grouped path; *scoring* does not, and the two questions were conflated.

* **The corpus.** `int8_ab.py` sets `DIVERGENCE_DOCS = 48` with a comment
  explaining that 48 puts n near 5,300 and the standard error near 0.15%, which
  is the resolution the decision needs. It does not. `_corpus_input_ids` reads
  `load_prompts()` with no path — the **built-in 24-prompt starter set**, whose
  docstring says in as many words that it is "a starting set" and that
  publishable numbers need a real corpus. `interleaved[:48]` of a 24-item list
  is 24 items, silently, and those prompts median 104 tokens against a 128-token
  window. n was 2,545, not 5,300; SE was 0.22% per arm, not 0.15%.

  The corpus the comment describes already exists in this repo, unused by this
  measurement: `tools/make_corpus.py` writes 48 documents of 529-1,704 tokens
  each. At a 512-token window that is **24,576 positions**, 9.7x the n, and an
  SE of **0.07%** — which resolves a 0.24-point gap at better than 3 sigma.

So the old number is not wrong so much as unreadable, and it was unreadable in
the specific way troubleshoot.md 4.8 warns about: a bar applied to a statistic
whose resolution is worse than the margin. Both faults are fixed here, and the
control that proves it is the floor row — on the loop path two identical arms
must agree to exactly 100.00%, or something else is nondeterministic and no
conclusion below that residue means anything.

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
Agreement, on every position. KL only on the first `KL_DOCS` documents: a full
fp16 logit tensor for 48 documents at 512 tokens is 2.5 GB per arm and this
process is already holding a multi-GB expert store. Agreement needs only the
argmax, which is 2 KB per document, so it is collected everywhere and the
expensive quantity is sampled. KL was never the disputed number — Stage 1e-2
measured 0.00127 nats against a 0.01 bar, an 8x margin that no amount of
sampling error closes.

Throughput is not measured here at all. `int8_ab.py` owns that question and its
answer (+17% at 238 slots) does not depend on this one.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flashforge.cli import _force_utf8_stdout, _setup_logging  # noqa: E402
from flashforge.models import DEFAULT_MODEL, load_model  # noqa: E402
from flashforge.runtime import ExpertCache, QuantSpec, install_expert_cache  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "corpus.jsonl"
WINDOW = 512
KL_DOCS = 8
PIN_GB = 7.0
# Capacity cannot change the arithmetic — the same weights are read into the
# same slots either way — so this is chosen to fit alongside the store rather
# than to match any operating point. It is held fixed across every arm.
SLOTS = 238
SPEC = QuantSpec(projections=("gate_proj", "up_proj"), group_size=0)
BAR = 0.99


def _corpus(tokenizer) -> list:
    """The 512-token evaluation corpus, not the built-in starter prompts."""
    from flashforge.prompts import load_prompts

    if not CORPUS.exists():
        raise SystemExit(
            f"{CORPUS.name} not found. Run:  uv run python tools/make_corpus.py "
            f"{CORPUS.name}\nThe built-in prompt set is deliberately not used "
            "here — it is 24 documents of ~104 tokens and it is what made the "
            "first reading of this bar unresolvable."
        )
    return [
        tokenizer(item["text"], return_tensors="pt").input_ids[:, :WINDOW].to("cuda")
        for item in load_prompts(CORPUS)
    ]


def _score(model, blocks, corpus, *, grouped: bool, keep_kl: bool) -> dict:
    """One teacher-forced sweep. Returns argmax per document, KL logits for a few.

    `grouped` is set on the blocks rather than baked in at install time, because
    the whole point is to compare the two accumulators on the *same* weights in
    the same process — reinstalling would reload or requantise and reintroduce
    the confound this tool exists to remove.
    """
    for block in blocks:
        block.grouped = grouped

    picks, logits = [], []
    with torch.no_grad():
        for i, ids in enumerate(corpus):
            out = model(ids, use_cache=False)
            row = out.logits[0].detach()
            picks.append(row.argmax(-1).to("cpu"))
            if keep_kl and i < KL_DOCS:
                logits.append(row.to("cpu", torch.float16))
            del out, row
    torch.cuda.empty_cache()
    return {"picks": picks, "logits": logits}


def _agreement(base: dict, other: dict) -> tuple[float, float, int, float]:
    """Top-1 agreement, its binomial standard error, n, and the worst document."""
    agreed = positions = 0
    worst = 1.0
    for mine, theirs in zip(base["picks"], other["picks"]):
        match = int((mine == theirs).sum())
        agreed += match
        positions += mine.numel()
        worst = min(worst, match / mine.numel())
    rate = agreed / positions
    return rate, (rate * (1 - rate) / positions) ** 0.5, positions, worst


def _kl(base: dict, other: dict) -> float:
    """Mean KL in nats over the sampled documents. fp16 storage, fp32 softmax."""
    total = positions = 0.0
    for mine, theirs in zip(base["logits"], other["logits"]):
        p = mine.float().log_softmax(-1)
        q = theirs.float().log_softmax(-1)
        total += float((p.exp() * (p - q)).sum())
        positions += mine.shape[0]
    return total / positions if positions else float("nan")


def _verdict(label: str, base: dict, other: dict, *, kl: bool = False) -> float:
    rate, stderr, n, worst = _agreement(base, other)
    sigma = abs(rate - BAR) / stderr if stderr else float("inf")
    line = (
        f"{label:>22}: {rate:7.3%} +/- {stderr:.3%} over {n:,} positions, "
        f"worst doc {worst:.2%}"
    )
    if kl:
        line += f", KL {_kl(base, other):.5f} nats ({KL_DOCS} docs)"
    print(line)
    print(
        f"{'':>22}  {'PASSES' if rate >= BAR else 'FAILS'} the >={BAR:.0%} bar "
        f"by {abs(rate - BAR) * 100:.3f} points = {sigma:.1f} sigma"
    )
    return rate


def main() -> int:
    _force_utf8_stdout()
    _setup_logging(verbose=True)
    if not torch.cuda.is_available():
        print("This needs a CUDA device; the expert cache has nowhere to live.")
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    corpus = _corpus(tokenizer)
    print(f"corpus: {len(corpus)} documents, "
          f"{sum(int(ids.numel()) for ids in corpus):,} positions, "
          f"{WINDOW}-token window")

    model, _ = load_model(DEFAULT_MODEL, dtype="float16", device_map=None)
    report = install_expert_cache(
        model, capacity=SLOTS, device="cuda", pin_gb=PIN_GB, grouped=False
    )
    print(report.describe())

    # fp16 reference, on the deterministic path. Everything below is scored
    # against this one sweep, so if it were nondeterministic every row would
    # inherit the noise and the floor control would not reveal it.
    fp16 = _score(model, report.blocks, corpus, grouped=False, keep_kl=True)

    # The pool goes before the quantisation and a new one comes after it. A
    # cache built on fp16 geometry cannot serve int8 rows — `acquire` refuses
    # it rather than splicing half an expert to half a neighbour — and freeing
    # the fp16 pool here is also what makes room for the smaller one.
    store = report.store
    report.cache.release()
    for block in report.blocks:
        block.cache = None
    gc.collect()
    torch.cuda.empty_cache()

    error = store.quantize_int8(SPEC)
    gc.collect()
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
    pinned = store.repin(PIN_GB)
    gc.collect()

    report.cache = ExpertCache(store, SLOTS, device="cuda", dtype=store.shape.dtype)
    for block in report.blocks:
        block.cache = report.cache
    print(f"\nquantised: {100 * error['rel_rms_error']:.2f}% relative RMS weight "
          f"error, {report.store.shape.nbytes / 1e6:.2f} MB per expert, "
          f"{pinned} of {len(report.store.layers)} layers pinned")

    int8_a = _score(model, report.blocks, corpus, grouped=False, keep_kl=True)
    int8_b = _score(model, report.blocks, corpus, grouped=False, keep_kl=False)
    grouped_a = _score(model, report.blocks, corpus, grouped=True, keep_kl=False)
    grouped_b = _score(model, report.blocks, corpus, grouped=True, keep_kl=False)

    print("\n" + "=" * 96)
    print("THE BAR — int8 experts against fp16, both on the deterministic loop path")
    print("=" * 96)
    _verdict("int8 vs fp16", fp16, int8_a, kl=True)

    print("\n" + "=" * 96)
    print("THE CONTROLS — identical weights twice, which is the resolution floor")
    print("=" * 96)
    loop = _verdict("loop floor", int8_a, int8_b)
    _verdict("grouped floor", grouped_a, grouped_b)
    print()
    if loop < 1.0:
        print("The loop path is NOT bit-reproducible. Something outside the "
              "accumulator is nondeterministic, and no agreement difference "
              "smaller than this residue can be attributed to quantisation.")
    else:
        print("The loop path reproduces exactly, so the int8 row above carries "
              "sampling error only — the instrument contributes nothing. The "
              "grouped floor is what Stage 1e-2 was reading through.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
