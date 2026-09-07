"""Command-line entry points: ff-collect and ff-analyze."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("flashforge")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_cache_space(cache_dir: str | None, needed_gb: float = 16.0) -> None:
    """Warn early if the Hugging Face cache lives on a nearly full drive.

    Worth a dedicated check here: a default Windows install puts the cache on
    C:, and a 7B checkpoint in fp16 is ~14GB. Failing at 95% of a download is a
    miserable way to find that out.
    """
    target = Path(cache_dir or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    log.info("Model cache: %s  (%.1f GB free)", target, free_gb)
    if free_gb < needed_gb:
        log.warning(
            "Only %.1f GB free where the model cache lives. A 7B fp16 checkpoint "
            "needs ~14 GB. Set HF_HOME to a roomier drive, e.g. "
            "$env:HF_HOME = 'D:\\ai\\hf-cache'  (or pass --cache-dir).",
            free_gb,
        )


# --------------------------------------------------------------------------
# ff-collect
# --------------------------------------------------------------------------

def collect_main(argv: list[str] | None = None) -> int:
    from tqdm import tqdm

    from .models import DEFAULT_MODEL, discover_moe, gate_weight_matrices, load_model
    from .prompts import load_prompts
    from .tracing import RouterTracer, trace_prompt

    parser = argparse.ArgumentParser(
        prog="ff-collect", description="Collect MoE routing traces from a model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="traces/olmoe", help="output directory")
    parser.add_argument("--prompts", default=None, help="JSONL prompt file (default: built-in set)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--gen-tokens", type=int, default=0,
        help="also capture N greedy decode steps per prompt (real serving access pattern)",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--load-4bit", action="store_true", help="faster iteration, perturbs routing")
    parser.add_argument("--no-hidden", action="store_true", help="skip hidden states (disables Q3)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-memory", default="4.5GiB")
    parser.add_argument("--cpu-memory", default="22GiB")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    _check_cache_space(args.cache_dir)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.model)
    model, tokenizer = load_model(
        args.model,
        dtype=args.dtype,
        load_4bit=args.load_4bit,
        cache_dir=args.cache_dir,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
    )
    spec = discover_moe(model)
    log.info("Discovered MoE: %s", spec.describe())

    prompts = load_prompts(args.prompts)
    log.info("Tracing %d prompts (max_length=%d, gen_tokens=%d)",
             len(prompts), args.max_length, args.gen_tokens)

    seq_domains: dict[int, str] = {}
    total_tokens = 0

    with RouterTracer(spec, out_dir, save_hidden=not args.no_hidden) as tracer:
        for seq_id, prompt in enumerate(tqdm(prompts, desc="prompts", unit="seq")):
            n_tokens = trace_prompt(
                model, tokenizer, tracer, prompt["text"], seq_id,
                max_length=args.max_length, gen_tokens=args.gen_tokens,
            )
            seq_domains[seq_id] = prompt["domain"]
            total_tokens += n_tokens

        parquet_path = tracer.write_parquet()

    np.savez(
        out_dir / "gates.npz",
        **{k: v.numpy() for k, v in gate_weight_matrices(spec).items()},
    )

    meta = {
        "model_id": args.model,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "hidden_size": spec.hidden_size,
        "norm_topk_prob": spec.norm_topk_prob,
        "moe_layers": spec.moe_layers,
        "seq_domains": {str(k): v for k, v in seq_domains.items()},
        "n_sequences": len(prompts),
        "total_tokens": total_tokens,
        "dtype": args.dtype,
        "load_4bit": args.load_4bit,
        "gen_tokens": args.gen_tokens,
        "saved_hidden": not args.no_hidden,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    log.info("Wrote %s (%d tokens across %d sequences)",
             parquet_path, total_tokens, len(prompts))
    log.info("Next: ff-analyze --traces %s", out_dir)
    return 0


# --------------------------------------------------------------------------
# ff-analyze
# --------------------------------------------------------------------------

def analyze_main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from . import analysis, cachesim, plots, viz

    parser = argparse.ArgumentParser(
        prog="ff-analyze", description="Answer the five Stage 0 questions from a trace."
    )
    parser.add_argument("--traces", default="traces/olmoe")
    parser.add_argument("--out", default=None, help="report directory (default: <traces>/report)")
    parser.add_argument(
        "--max-accesses", type=int, default=300_000,
        help="cap on cache-sim accesses; the sweep is O(n) per policy per capacity",
    )
    parser.add_argument(
        "--bytes-per-expert", type=float, default=None,
        help="size of one expert's weights, for the bytes-per-token column",
    )
    parser.add_argument("--skip-q3", action="store_true", help="skip the probe (needs hidden states)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    viz.use_style()

    store = analysis.TraceStore(args.traces)
    out_dir = Path(args.out) if args.out else Path(args.traces) / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = store.routing
    log.info("Loaded %s: %d routing rows, %d experts, top-%d, %d MoE layers",
             args.traces, len(frame), store.num_experts, store.top_k, len(store.moe_layers))

    def save(name: str, table: pd.DataFrame) -> None:
        table.to_csv(out_dir / f"{name}.csv", index=False)

    # Q1 -------------------------------------------------------------
    log.info("Q1: expert usage skew")
    freq = analysis.expert_frequency(frame, store.num_experts)
    skew = analysis.skew_summary(freq, store.num_experts)
    lorenz = analysis.lorenz_curve(freq)
    save("q1_expert_frequency", freq)
    save("q1_skew_summary", skew)
    plots.plot_skew(lorenz, skew).savefig(out_dir / "q1_skew.png")
    print(f"\n[Q1] hottest 10% of experts serve "
          f"{skew['top10pct_mass'].mean():.1%} of accesses (uniform would be 10.0%)")
    print(f"     mean Gini {skew['gini'].mean():.3f} | "
          f"never-routed experts: {int(skew['unused_experts'].sum())}")

    # Q2 -------------------------------------------------------------
    log.info("Q2: temporal locality")
    overlap = analysis.consecutive_overlap(frame, store.num_experts, store.top_k)
    save("q2_consecutive_overlap", overlap)
    plots.plot_locality(overlap).savefig(out_dir / "q2_locality.png")
    lag1 = overlap[overlap["lag"] == 1]["mean_overlap"].mean()
    print(f"\n[Q2] {lag1:.1%} of a token's experts were also used by the previous token")
    print(f"     (random baseline: {store.top_k / store.num_experts:.1%})")

    # Q3 -------------------------------------------------------------
    if not args.skip_q3 and store.meta.get("saved_hidden", False):
        log.info("Q3: cross-layer predictability (this is the slow one)")
        try:
            detail = analysis.cross_layer_predictability(store)
            summary = analysis.predictability_summary(detail)
            save("q3_predictability_detail", detail)
            save("q3_predictability_summary", summary)
            plots.plot_predictability(summary, store.top_k).savefig(out_dir / "q3_predictability.png")
            print("\n[Q3] recall of the true top-k, averaged over layer pairs:")
            print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        except FileNotFoundError as exc:
            log.warning("Skipping Q3: %s", exc)
    else:
        log.info("Q3 skipped")

    # Q6 -------------------------------------------------------------
    log.info("Q6: expert-set expansion")
    expansion_detail = analysis.expert_set_expansion(frame, store.num_experts, store.top_k)
    expansion = analysis.expansion_summary(expansion_detail, store.num_experts, store.top_k)
    save("q6_expansion", expansion)
    plots.plot_expansion(expansion, store.top_k).savefig(out_dir / "q6_expansion.png")

    indexed = expansion.set_index("block_size")
    print("\n[Q6] unique experts touched by a block of B tokens:")
    for block in [b for b in (2, 4, 8) if b in indexed.index]:
        row = indexed.loc[block]
        print(f"     B={block:>2}  {row['expansion_ratio']:.2f}x top-k "
              f"(random routing would be {row['random_ratio']:.2f}x)  "
              f"→ {row['bytes_amortization']:.2f}x fewer bytes/token")
    if 4 in indexed.index:
        payoff = float(indexed.loc[4, "bytes_amortization"])
        print("     " + (
            "batching is a large win — speculative decoding belongs in the design early"
            if payoff > 2.0 else
            "expert scattering dominates — speculation needs draft-footprint control first"
            if payoff < 1.5 else
            "moderate payoff — worth it if draft acceptance is high"))

    # Q4 -------------------------------------------------------------
    log.info("Q4: domain clustering")
    domains = store.domains
    if len(set(domains.values())) > 1:
        result = analysis.domain_divergence(frame, domains, store.num_experts)
        result.matrix.to_csv(out_dir / "q4_domain_divergence.csv")
        plots.plot_domain(result).savefig(out_dir / "q4_domain.png")
        print(f"\n[Q4] between-domain JS {result.between_matched:.4f} vs "
              f"within-domain noise floor {result.within_matched:.4f} "
              f"({result.ratio:.1f}x, sample-matched)")
        print("     " + ("domains separate — cache warming has something to exploit"
                         if result.separates
                         else "domains do NOT separate cleanly — warming is unlikely to pay"))
    else:
        log.info("Q4 skipped: only one domain in the trace")

    # Q5 -------------------------------------------------------------
    log.info("Q5: cache simulation")
    keys = cachesim.build_access_sequence(frame, store.num_experts)
    if keys.size > args.max_accesses:
        log.info("Truncating cache sim to the first %d of %d accesses",
                 args.max_accesses, keys.size)
        keys = keys[: args.max_accesses]

    total_slots = store.num_experts * len(store.moe_layers)
    capacities = sorted({
        max(1, int(total_slots * fraction))
        for fraction in (0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00)
    })
    sweep = cachesim.sweep(
        keys, capacities,
        bytes_per_expert=args.bytes_per_expert,
        accesses_per_token=store.top_k * len(store.moe_layers),
    )
    save("q5_cache_sweep", sweep)
    plots.plot_cache(sweep, total_slots=total_slots).savefig(out_dir / "q5_cache.png")

    quarter = max(1, int(total_slots * 0.25))
    at_quarter = sweep[sweep["capacity"] == min(capacities, key=lambda c: abs(c - quarter))]
    print(f"\n[Q5] at ~25% of experts resident ({quarter} of {total_slots} slots):")
    for _, row in at_quarter.iterrows():
        line = f"     {row['policy']:>7}  hit rate {row['hit_rate']:.1%}"
        if "fetch_bytes_per_token" in row and pd.notna(row.get("fetch_bytes_per_token")):
            line += f"  |  {row['fetch_bytes_per_token'] / 1e6:.1f} MB fetched/token"
        print(line)
    gap = at_quarter.set_index("policy")["hit_rate"]
    if {"belady", "lru"} <= set(gap.index):
        headroom = gap["belady"] - gap["lru"]
        print(f"\n     Belady - LRU headroom: {headroom:.1%} "
              + ("→ eviction policy is worth real work"
                 if headroom > 0.05 else "→ eviction is near-optimal; put the effort into prefetch"))

    print(f"\nReport written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(collect_main())
