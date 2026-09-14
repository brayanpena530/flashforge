"""Command-line entry points: ff-collect, ff-analyze and ff-bench."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("flashforge")


def _force_utf8_stdout() -> None:
    """Stop a Windows console from killing a finished run over an arrow glyph.

    The default stdout encoding here is cp1252, which cannot represent the
    arrows and dashes in the verdict lines. Printing one raised
    UnicodeEncodeError *after* the analysis had completed, discarding it. Errors
    are set to "replace" so an unrepresentable character degrades to "?" rather
    than taking the process down.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


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

    _force_utf8_stdout()
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

    # Order matters. Tracing is the expensive part — minutes of forward passes —
    # and everything after it is small bookkeeping. Writing the router gates
    # first meant one failure there discarded a complete trace, so the cheap,
    # never-fails write goes first and the fragile one is allowed to fail.
    saved_gates = True
    try:
        np.savez(
            out_dir / "gates.npz",
            **{k: v.numpy() for k, v in gate_weight_matrices(spec).items()},
        )
    except RuntimeError as exc:
        saved_gates = False
        log.warning(
            "Could not save router gates: %s\nThe trace is still complete and Q1, Q2, "
            "Q4, Q5 and Q6 will run. Q3 loses its stale_router predictor.", exc,
        )

    meta = {
        "saved_gates": saved_gates,
        "model_id": args.model,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "hidden_size": spec.hidden_size,
        # Needed by ff-bench (Q7) to build an expert of the right shape and to
        # derive bytes-per-expert without reloading the checkpoint.
        "intermediate_size": spec.intermediate_size,
        "expert_bytes_fp16": spec.expert_bytes(2.0),
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
    parser.add_argument(
        "--q3-offsets", default="1,2,3,4,6,8",
        help="lookahead depths to sweep, comma-separated. Deeper is slower but a "
             "disk tier needs more lead time than one layer buys",
    )
    parser.add_argument(
        "--edge-fraction", type=float, default=0.25,
        help="share of layers assigned to each of the input/output bands in Q3",
    )
    parser.add_argument(
        "--required-lookahead", type=int, default=None,
        help="draw this depth on the Q3 chart (default: read it from ff-bench's hardware.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _force_utf8_stdout()
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

    # The stack-wide mean above hides the shape, and the shape is what picks the
    # predictor: concentration and weight dominance are expected to move in
    # opposite directions with depth.
    weights = analysis.routing_weight_profile(frame, store.top_k)
    profile = analysis.annotate_bands(
        skew.merge(weights, on="layer", how="left") if not weights.empty else skew,
        store.moe_layers, edge_fraction=args.edge_fraction,
    )
    bands = analysis.band_summary(
        skew, weights, store.moe_layers, edge_fraction=args.edge_fraction
    )
    save("q1_layer_profile", profile)
    save("q1_band_summary", bands)
    plots.plot_layer_bands(profile, top_k=store.top_k).savefig(out_dir / "q1_layer_bands.png")

    print("\n[Q1] by layer band:")
    print(f"     {'band':>7}  {'layers':>6}  {'top-10% mass':>12}  {'gini':>6}  "
          f"{'top-1 weight':>12}")
    for _, row in bands.iterrows():
        share = f"{row['top1_share']:.3f}" if "top1_share" in bands.columns else "n/a"
        print(f"     {row['band']:>7}  {int(row['n_layers']):>6}  "
              f"{row['top10pct_mass']:>11.1%}  {row['gini']:>6.3f}  {share:>12}")

    if "top1_share" in bands.columns and len(bands) == 3:
        keyed = bands.set_index("band")
        edges = (keyed.loc["input", "top1_share"] + keyed.loc["output", "top1_share"]) / 2
        mid = keyed.loc["middle", "top1_share"]
        print("     " + (
            "edge layers lean on one dominant expert, middle layers blend — "
            "so edges want previous-layer expert IDs as the prefetch feature, "
            "middle wants the hidden state"
            if edges > mid + 0.02 else
            "middle layers lean harder on one expert than the edges do — the "
            "inverse of the published pattern, worth a second look"
            if mid > edges + 0.02 else
            "routing-weight dominance is flat across depth; the layer-band split "
            "buys nothing on this model and one predictor should serve the stack"))

    # Q2 -------------------------------------------------------------
    log.info("Q2: temporal locality")
    overlap = analysis.consecutive_overlap(frame, store.num_experts, store.top_k)
    save("q2_consecutive_overlap", overlap)
    plots.plot_locality(overlap).savefig(out_dir / "q2_locality.png")
    lag1 = overlap[overlap["lag"] == 1]["mean_overlap"].mean()
    print(f"\n[Q2] {lag1:.1%} of a token's experts were also used by the previous token")
    print(f"     (random baseline: {store.top_k / store.num_experts:.1%})")

    # Prefill and decode are different access patterns, and only one of them is
    # what a serving cache sees. Reporting the blend hides that.
    phases = analysis.phase_frames(frame)
    if len(phases) > 1:
        for name, part in phases.items():
            sub = analysis.consecutive_overlap(part, store.num_experts, store.top_k)
            value = sub[sub["lag"] == 1]["mean_overlap"].mean()
            tokens = part.groupby(["seq_id", "pos"], observed=True).ngroups
            print(f"     {name:>8}: {value:.1%} over {tokens:,} tokens")

    # Q3 -------------------------------------------------------------
    # ff-bench writes hardware.json next to the report; if it is there, the
    # lookahead depth a disk read needs gets drawn straight onto the chart, so
    # Q3 and Q8 can be read against each other without copying numbers by hand.
    required_lookahead = args.required_lookahead
    hardware_path = out_dir / "hardware.json"
    if required_lookahead is None and hardware_path.exists():
        try:
            # utf-8-sig, not utf-8: anything that has passed through a Windows
            # shell may carry a BOM, and json.load rejects it outright.
            with hardware_path.open("r", encoding="utf-8-sig") as handle:
                required_lookahead = json.load(handle).get("required_lookahead")
            if required_lookahead:
                log.info("Using required_lookahead=%d from %s", required_lookahead, hardware_path)
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s: %s", hardware_path, exc)

    if not args.skip_q3 and store.meta.get("saved_hidden", False):
        log.info("Q3: cross-layer predictability (this is the slow one)")
        try:
            offsets = tuple(int(v) for v in args.q3_offsets.split(",") if v.strip())
            detail = analysis.cross_layer_predictability(
                store, offsets=offsets, edge_fraction=args.edge_fraction
            )
            summary = analysis.predictability_summary(detail)
            by_group = analysis.predictability_by_group(detail)
            save("q3_predictability_detail", detail)
            save("q3_predictability_summary", summary)
            save("q3_predictability_by_group", by_group)
            plots.plot_predictability(summary, store.top_k).savefig(out_dir / "q3_predictability.png")
            plots.plot_predictability_groups(
                by_group, store.top_k, required_lookahead=required_lookahead
            ).savefig(out_dir / "q3_predictability_groups.png")

            print("\n[Q3] recall of the true top-k, averaged over layer pairs:")
            print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

            # Every predictor, not just the cheapest one. Predictors differ in
            # how fast they *decay* with depth as much as in peak accuracy, and
            # a slow storage tier lives or dies on the far end of that curve —
            # judging it by one predictor understates what is reachable.
            for budget in sorted(by_group["budget_mult"].unique()):
                reach = analysis.lookahead_reach(by_group, budget_mult=int(budget))
                if reach.empty:
                    continue
                best = analysis.best_lookahead_by_band(reach)
                save(f"q3_lookahead_reach_{int(budget)}x", reach)

                print(f"\n[Q3] deepest lookahead still clearing 80% recall "
                      f"({int(budget)}x prefetch budget):")
                note = {
                    "accuracy": "falls off past here",
                    "coverage": "ran out of layers, not accuracy",
                    "none": "still holding at the deepest offset swept",
                }
                for _, row in best.iterrows():
                    depth = int(row["deepest_offset"])
                    verdict = f"{depth} layer(s)" if depth else "not even 1 layer"
                    print(f"     {row['src_group']:>7}  {verdict:<16} "
                          f"best: {row['predictor']:<13} "
                          f"(k=1 {row['recall_at_1']:.3f} -> deepest {row['recall_at_deepest_swept']:.3f}; "
                          f"{note.get(row['limited_by'], row['limited_by'])})")

                if required_lookahead:
                    # A band that ran out of layer pairs has not demonstrated a
                    # ceiling, so scoring it against the disk requirement would
                    # manufacture a failure out of arithmetic.
                    real = best[best["limited_by"] != "coverage"]
                    if real.empty:
                        print(f"     a disk read needs {required_lookahead} layer(s); no band "
                              "was measured deep enough to judge that")
                        continue
                    worst = int(real["deepest_offset"].min())
                    band = real.loc[real["deepest_offset"].idxmin(), "src_group"]
                    who = real.loc[real["deepest_offset"].idxmin(), "predictor"]
                    if worst >= required_lookahead:
                        print(f"     -> covers a disk read ({required_lookahead} layers): even the "
                              f"weakest band ({band}) reaches {worst} with {who}")
                    else:
                        print(f"     -> short of a disk read ({required_lookahead} layers): the "
                              f"{band} band tops out at {worst}, best predictor {who}")
                        print("        a longer-horizon signal (block-level speculation, or Q4 "
                              "domain warming) has to carry the disk->RAM decision")
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
    # Budget by dropping whole sequences. Truncating the flattened stream takes
    # the first few documents instead of a sample of them, and the policy
    # ranking flips with how many documents are in view.
    sim_frame, kept, available = cachesim.subsample_sequences(frame, args.max_accesses)
    if kept < available:
        log.info("Cache sim on %d of %d sequences (budget %d accesses)",
                 kept, available, args.max_accesses)
        print(f"\n[Q5] sampling {kept} of {available} sequences to stay under "
              f"--max-accesses {args.max_accesses:,}. Raise it to use the whole corpus; "
              "policy ranking is sensitive to how many documents are in view.")
    keys = cachesim.build_access_sequence(sim_frame, store.num_experts)

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

    # The decode-only sweep is the one that describes serving. Prefill amortises
    # a whole sequence's experts across one forward pass and flatters every
    # policy; decode pays per token.
    if len(phases) > 1 and "decode" in phases:
        decode_part, _, _ = cachesim.subsample_sequences(phases["decode"], args.max_accesses)
        decode_keys = cachesim.build_access_sequence(decode_part, store.num_experts)
        if decode_keys.size:
            decode_sweep = cachesim.sweep(
                decode_keys, [quarter],
                bytes_per_expert=args.bytes_per_expert,
                accesses_per_token=store.top_k * len(store.moe_layers),
            )
            save("q5_cache_sweep_decode", decode_sweep)
            print(f"\n[Q5] decode only ({decode_keys.size:,} accesses), same capacity:")
            for _, row in decode_sweep.iterrows():
                line = f"     {row['policy']:>7}  hit rate {row['hit_rate']:.1%}"
                if pd.notna(row.get("fetch_bytes_per_token")):
                    line += f"  |  {row['fetch_bytes_per_token'] / 1e6:.1f} MB fetched/token"
                print(line)
            d = decode_sweep.set_index("policy")["hit_rate"]
            if {"belady", "lru"} <= set(d.index):
                print(f"     Belady - LRU headroom on decode: {d['belady'] - d['lru']:.1%}")

    print(f"\nReport written to {out_dir}")
    return 0


# --------------------------------------------------------------------------
# ff-bench
# --------------------------------------------------------------------------

def _bench_dims(args) -> tuple[int, int, int | None, float]:
    """Resolve expert shape from meta.json, with explicit flags winning."""
    meta: dict = {}
    if args.traces:
        meta_path = Path(args.traces) / "meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8-sig") as handle:
                meta = json.load(handle)
        else:
            log.warning("No meta.json at %s; falling back to explicit dimensions", meta_path)

    hidden = args.hidden_size or meta.get("hidden_size")
    intermediate = args.intermediate_size or meta.get("intermediate_size")
    top_k = args.top_k or meta.get("top_k")

    if not hidden or not intermediate:
        raise SystemExit(
            "Need the expert's shape. Either point --traces at a directory whose "
            "meta.json carries hidden_size and intermediate_size, or pass "
            "--hidden-size and --intermediate-size directly.\n"
            "Traces collected before intermediate_size was recorded will not have "
            "it — pass it explicitly (OLMoE-1B-7B: 1024, Qwen3-30B-A3B: 768)."
        )

    hidden, intermediate = int(hidden), int(intermediate)
    bytes_each = args.expert_bytes or hardware_expert_bytes(hidden, intermediate, args.bytes_per_param)
    return hidden, intermediate, (int(top_k) if top_k else None), float(bytes_each)


def hardware_expert_bytes(hidden: int, intermediate: int, bytes_per_param: float) -> float:
    from .hardware import expert_bytes

    return expert_bytes(hidden, intermediate, bytes_per_param)


def bench_main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from . import hardware, plots, viz

    parser = argparse.ArgumentParser(
        prog="ff-bench",
        description="Q7 (cost model) and Q8 (storage curve) — measure the machine, not the model.",
    )
    parser.add_argument("--traces", default=None, help="trace dir to read expert dimensions from")
    parser.add_argument("--out", default=None, help="report directory")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None,
                        help="expert FFN inner dimension")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--expert-bytes", type=float, default=None,
                        help="override bytes per expert (default: derived from the shape)")
    parser.add_argument("--bytes-per-param", type=float, default=2.0,
                        help="2.0 for fp16/bf16, 0.5 for 4-bit")

    parser.add_argument("--skip-q7", action="store_true")
    parser.add_argument("--skip-q8", action="store_true")
    parser.add_argument("--cpu-dtype", default="float32", choices=["float32", "bfloat16"],
                        help="CPU float16 GEMM is emulated in most builds and would "
                             "measure the emulation")
    parser.add_argument("--gpu-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: all)")

    parser.add_argument("--scratch", default=".ff_scratch.bin", help="scratch file for Q8")
    parser.add_argument("--file-size-gb", type=float, default=2.0,
                        help="scratch file size. Must exceed free RAM to escape the page cache")
    parser.add_argument("--queue-depths", default="1,2,4,8,16,32")
    parser.add_argument("--read-sizes-mib", default="0.0625,0.25,1,4,16")
    parser.add_argument("--remove-scratch", action="store_true",
                        help="delete the scratch file afterwards (it is recreated next run)")

    parser.add_argument("--layer-time-ms", type=float, default=None,
                        help="measured per-layer decode time; used to convert a disk read "
                             "into a lookahead depth")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _force_utf8_stdout()
    _setup_logging(args.verbose)
    viz.use_style()

    hidden, intermediate, top_k, bytes_each = _bench_dims(args)
    out_dir = Path(args.out) if args.out else (
        Path(args.traces) / "report" if args.traces else Path("bench_report")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = hardware.HardwareReport(
        hidden_size=hidden, intermediate_size=intermediate, expert_bytes=bytes_each
    )
    log.info("Expert shape: hidden %d x FFN %d  ->  %.1f MB at %.1f bytes/param",
             hidden, intermediate, bytes_each / 1e6, args.bytes_per_param)

    def save(name: str, table: pd.DataFrame) -> None:
        table.to_csv(out_dir / f"{name}.csv", index=False)

    # Q7 -------------------------------------------------------------
    gpu_path_ms = None
    if not args.skip_q7:
        log.info("Q7: cost model calibration")
        cpu_curve = hardware.cpu_expert_curve(
            hidden, intermediate, dtype=args.cpu_dtype, threads=args.threads
        )
        fit = hardware.fit_linear_cost(cpu_curve)
        save("q7_cpu_expert_curve", cpu_curve)
        report.cpu_beta_ms_per_token = fit.beta_ms_per_token
        report.cpu_const_ms = fit.const_ms
        report.cpu_r_squared = fit.r_squared
        report.cpu_threads = int(cpu_curve["threads"].iloc[0])

        print(f"\n[Q7] CPU expert cost, {report.cpu_threads} threads, {args.cpu_dtype}:")
        print(f"     t_c = {fit.beta_ms_per_token:.4f} ms/token * m + {fit.const_ms:.3f} ms "
              f"(r2 {fit.r_squared:.3f})")
        if fit.r_squared < 0.95:
            print("     r2 is low — CPU cost is not linear across this range, so a single "
                  "beta is the wrong abstraction. Look at the per-token panel for the kink.")

        gpu_curve = None
        try:
            gpu_curve = hardware.gpu_expert_cost(hidden, intermediate, dtype=args.gpu_dtype)
            pcie = hardware.pcie_transfer_cost(hidden, intermediate, dtype=args.gpu_dtype)
            save("q7_gpu_expert_curve", gpu_curve)
            save("q7_pcie_transfer", pcie)

            best = pcie.loc[pcie["ms"].idxmin()]
            report.gpu_expert_ms = float(gpu_curve["ms"].median())
            report.pcie_best_ms = float(best["ms"])
            report.pcie_best_mode = str(best["mode"])
            report.pcie_gbps = float(best["gbps"])
            gpu_path_ms = report.pcie_best_ms + report.gpu_expert_ms

            print(f"\n[Q7] PCIe transfer of one expert ({bytes_each / 1e6:.1f} MB):")
            for _, row in pcie.sort_values("ms").iterrows():
                print(f"     {row['mode']:>13}  {row['ms']:7.3f} ms   {row['gbps']:5.2f} GB/s")
            print(f"     GPU compute {report.gpu_expert_ms:.3f} ms (flat in m) "
                  f"→ full GPU path {gpu_path_ms:.3f} ms")

            m_star = hardware.break_even_tokens(fit, gpu_path_ms)
            report.break_even_tokens = m_star
            print(f"\n[Q7] break-even m* = {m_star:.1f} tokens")
            if not math.isfinite(m_star):
                print("     the CPU is cheaper at every token count measured — "
                      "CPU-side execution is a real lever, not a rounding error")
            elif m_star < 1:
                print("     the GPU path wins even for a single token. CPU-in-place "
                      "execution buys nothing here; spend the effort on prefetch instead")
            else:
                print(f"     experts with fewer than ~{m_star:.0f} tokens routed to them are "
                      "cheaper computed in place on the CPU — and each one you keep there "
                      "frees a PCIe slot for a prefetch")
        except RuntimeError as exc:
            log.warning("Skipping the GPU half of Q7: %s", exc)
            print("\n[Q7] no CUDA device — CPU curve only, no break-even")

        plots.plot_cost_model(
            cpu_curve, fit,
            gpu_path_ms=gpu_path_ms,
            break_even=report.break_even_tokens,
            gpu_curve=gpu_curve,
        ).savefig(out_dir / "q7_cost_model.png")

    # Q8 -------------------------------------------------------------
    if not args.skip_q8:
        log.info("Q8: storage read curve")
        read_sizes = tuple(
            int(float(v) * (1 << 20)) for v in args.read_sizes_mib.split(",") if v.strip()
        )
        depths = tuple(int(v) for v in args.queue_depths.split(",") if v.strip())
        scratch = Path(args.scratch)

        curve = hardware.storage_read_curve(
            scratch,
            file_size_bytes=int(args.file_size_gb * (1 << 30)),
            read_sizes=read_sizes,
            queue_depths=depths,
        )
        knees = hardware.saturation_knee(curve)
        save("q8_storage_curve", curve)
        save("q8_storage_knees", knees)
        plots.plot_storage(curve, knees).savefig(out_dir / "q8_storage.png")

        disk_ms, provenance = hardware.disk_time_for_expert(curve, bytes_each)
        report.disk_expert_ms = disk_ms
        report.disk_peak_gbps = float(curve["gbps"].max())
        report.disk_knee_queue_depth = int(knees["knee_queue_depth"].max())

        ram = hardware.total_ram_bytes()
        report.disk_measurement_trusted = bool(
            ram and int(args.file_size_gb * (1 << 30)) >= ram
        )

        print(f"\n[Q8] peak read bandwidth {report.disk_peak_gbps:.2f} GB/s")
        if not report.disk_measurement_trusted:
            print("     *** these are page-cache numbers, not disk numbers — the scratch "
                  "file fits in RAM.")
            print("     *** the harness is working; the figures are not usable for design. "
                  "Raise --file-size-gb past your RAM.")
        print("     request size   peak GB/s   knee QD   depth-1 penalty")
        for _, row in knees.iterrows():
            # iterrows upcasts a mixed-dtype row to float64, so the integer
            # column arrives as a float and an int format code would blow up.
            print(f"     {row['read_mib']:>8.3g} MiB {row['peak_gbps']:>10.2f} "
                  f"{int(row['knee_queue_depth']):>9d}   {row['qd1_penalty']:.2f}x")
        print(f"\n     one expert ({bytes_each / 1e6:.1f} MB) off disk ≈ {disk_ms:.2f} ms "
              f"(from {provenance['measured_read_bytes'] / (1 << 20):g} MiB reads "
              f"at queue depth {provenance['queue_depth']})")

        deepest_knee = report.disk_knee_queue_depth
        print("     " + (
            "depth-1 already saturates the device — a serial I/O cost model is fine here"
            if deepest_knee <= 1 else
            f"the device needs {deepest_knee} concurrent reads to saturate. A prefetcher "
            "issuing one expert at a time will leave most of it idle no matter how good "
            "its predictions are — the disk tier wants batched requests"))

        if report.pcie_best_ms:
            report.disk_to_pcie_ratio = disk_ms / report.pcie_best_ms
            print(f"\n[Q8] t_disk / t_pcie = {report.disk_to_pcie_ratio:.1f}x")

        layer_ms = args.layer_time_ms
        if layer_ms is None and report.gpu_expert_ms and top_k:
            # A floor, not an estimate: real per-layer time also carries
            # attention, the router and norms. Underestimating the denominator
            # overestimates the required depth, which is the safe direction.
            layer_ms = report.gpu_expert_ms * top_k
            log.info("No --layer-time-ms; using a floor of top_k * gpu_expert_ms = %.3f ms", layer_ms)
        if layer_ms:
            report.layer_time_ms = float(layer_ms)
            report.required_lookahead = hardware.required_lookahead(disk_ms, layer_ms)
            print(f"     at {layer_ms:.2f} ms per layer, hiding one disk read needs "
                  f"{report.required_lookahead} layer(s) of lookahead")
            print(f"     → run: ff-analyze --traces {args.traces or '<traces>'} "
                  f"--q3-offsets 1,2,3,4,6,8")
            print("       and check whether recall survives out that far, per band")

        if args.remove_scratch and scratch.exists():
            scratch.unlink()
            log.info("Removed scratch file %s", scratch)
        elif scratch.exists():
            log.info("Scratch file kept at %s (%.1f GiB) — pass --remove-scratch to delete",
                     scratch, scratch.stat().st_size / (1 << 30))

    report.to_json(out_dir / "hardware.json")
    print(f"\nWrote {out_dir / 'hardware.json'} — ff-analyze picks it up automatically.")
    return 0


# --------------------------------------------------------------------------
# ff-serve
# --------------------------------------------------------------------------

def _cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_phase(model, input_ids, gen_tokens: int, stats=None) -> dict:
    """Time prefill and decode, keeping their cache statistics apart.

    Blending the two produces a hit rate that means nothing. Prefill touches
    the union of experts over every token in the batch — past a few dozen
    tokens that is every expert in the layer, so the misses are compulsory and
    no cache size changes them. Decode touches exactly top_k per layer, which
    is the regime Q5's capacity sweep actually modelled. Only the decode number
    is comparable to Stage 0's 54.5%.
    """
    import time

    import torch

    def snapshot():
        if stats is None:
            return (0, 0, 0)
        return (stats.hits, stats.misses, stats.bytes_fetched)

    with torch.no_grad():
        before = snapshot()
        _cuda_sync()
        start = time.perf_counter()
        out = model(input_ids, use_cache=True)
        _cuda_sync()
        prefill_s = time.perf_counter() - start
        after_prefill = snapshot()

        result = {
            "prefill_s": prefill_s,
            "prefill_tokens": int(input_ids.shape[1]),
            "prefill_counts": tuple(a - b for a, b in zip(after_prefill, before)),
            "decode_s": 0.0,
            "decode_tokens": 0,
            "decode_counts": (0, 0, 0),
        }
        if gen_tokens <= 0:
            return result

        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)

        _cuda_sync()
        start = time.perf_counter()
        for _ in range(gen_tokens):
            out = model(next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
        _cuda_sync()
        result["decode_s"] = time.perf_counter() - start
        result["decode_tokens"] = gen_tokens
        result["decode_counts"] = tuple(a - b for a, b in zip(snapshot(), after_prefill))

    return result


def _rate(hits: int, misses: int) -> float:
    total = hits + misses
    return hits / total if total else 0.0


def serve_main(argv: list[str] | None = None) -> int:
    """Measure the Stage 1 offload runtime on a real model.

    Reports prefill and decode separately because they stress the cache in
    completely different ways, and averaging them hides both. A decode step
    touches exactly top_k experts per layer, so the working set is tiny and the
    cache has a real chance. A prefill batch touches the *union* over all its
    tokens, which past a few dozen tokens is essentially every expert in the
    layer — no cache smaller than the model can help, and the hit rate is
    reporting the layer sweep, not locality.
    """
    import torch

    from . import hardware
    from .runtime import install_expert_cache

    parser = argparse.ArgumentParser(
        prog="ff-serve", description="Benchmark the offloaded MoE runtime."
    )
    parser.add_argument("--model", default=None, help="model id (default: the Stage 0 dev model)")
    parser.add_argument(
        "--capacity", default="256",
        help="expert slots to cache, comma-separated to sweep (e.g. 128,256,512)",
    )
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1, help="untimed passes before measuring")
    parser.add_argument(
        "--pin-gb", type=float, default=0.0,
        help="host RAM to page-lock for async DMA. Pinned pages cannot be swapped, "
             "so this is a hard claim on physical memory — see runtime/store.py",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _force_utf8_stdout()
    _setup_logging(args.verbose)

    if not torch.cuda.is_available():
        print("ff-serve needs a CUDA device; the whole point is the PCIe crossing.")
        return 1

    from .models import DEFAULT_MODEL, load_model

    model_id = args.model or DEFAULT_MODEL
    capacities = [int(c) for c in args.capacity.split(",") if c.strip()]

    # device_map=None loads everything to CPU. That is deliberate: with
    # device_map="auto" accelerate would scatter experts across devices and
    # meta tensors and then fight the runtime for control of placement.
    log.info("Loading %s to CPU (%s)...", model_id, args.dtype)
    model, tokenizer = load_model(
        model_id, dtype=args.dtype, device_map=None, cache_dir=args.cache_dir
    )

    prompt = "The history of computing is" * 64
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.prompt_tokens]
    input_ids = input_ids.to("cuda")

    import gc

    rows = []
    report = None
    for i, capacity in enumerate(capacities):
        # Each capacity needs a fresh model: install_expert_cache moves the
        # experts out, so the previous iteration left a hollow shell behind.
        #
        # Dropping `report` first is not tidiness. It owns the previous store
        # (12 GB of host RAM) and the previous slot pool (3 GB of VRAM), and
        # loading the next 13.8 GB checkpoint while both are still live would
        # exhaust host memory on any machine that can only just hold one copy.
        if i > 0:
            del model, report
            report = None
            gc.collect()
            torch.cuda.empty_cache()
            model, _ = load_model(
                model_id, dtype=args.dtype, device_map=None, cache_dir=args.cache_dir
            )

        report = install_expert_cache(model, capacity=capacity, device="cuda", pin_gb=args.pin_gb)
        print(f"\n=== capacity {capacity}")
        print(report.describe())

        free = hardware.available_ram_bytes()
        if free is not None:
            free_gb = free / (1 << 30)
            print(f"  host RAM free: {free_gb:.1f} GB")
            # Swapping shows up as a throughput collapse with an unchanged or
            # better hit rate, which reads exactly like a cache-policy finding.
            # It is not one, so say so here rather than letting the table imply it.
            if free_gb < 2.0:
                print("  WARNING: under 2 GB free — these timings are measuring swap, "
                      "not the cache. Compare capacities across separate runs instead.")

        stats = report.cache.stats
        for _ in range(args.warmup):
            _run_phase(model, input_ids, min(4, args.gen_tokens))

        stats.reset()
        timing = _run_phase(model, input_ids, args.gen_tokens, stats)

        p_hits, p_misses, p_bytes = timing["prefill_counts"]
        d_hits, d_misses, d_bytes = timing["decode_counts"]
        prefill_tps = timing["prefill_tokens"] / timing["prefill_s"] if timing["prefill_s"] else 0.0
        decode_tps = timing["decode_tokens"] / timing["decode_s"] if timing["decode_s"] else 0.0

        rows.append({
            "capacity": capacity,
            "prefill_tok_s": prefill_tps,
            "decode_tok_s": decode_tps,
            "prefill_hit_rate": _rate(p_hits, p_misses),
            "decode_hit_rate": _rate(d_hits, d_misses),
            "decode_gb_per_token": d_bytes / 1e9 / max(1, timing["decode_tokens"]),
            "vram_gb": report.resident_bytes / (1 << 30),
        })
        print(f"  prefill {prefill_tps:8.1f} tok/s  hit {_rate(p_hits, p_misses):5.1%}  "
              f"{p_bytes / 1e9:5.2f} GB")
        print(f"  decode  {decode_tps:8.2f} tok/s  hit {_rate(d_hits, d_misses):5.1%}  "
              f"{d_bytes / 1e9:5.2f} GB  ({stats.evictions:,} evictions)")

    print("\n" + "=" * 78)
    print(f"{'slots':>7} {'VRAM GB':>8} {'prefill t/s':>12} {'pf hit':>7} "
          f"{'decode t/s':>11} {'dec hit':>8} {'GB/tok':>7}")
    for row in rows:
        print(f"{row['capacity']:>7} {row['vram_gb']:>8.2f} {row['prefill_tok_s']:>12.1f} "
              f"{row['prefill_hit_rate']:>6.1%} {row['decode_tok_s']:>11.2f} "
              f"{row['decode_hit_rate']:>7.1%} {row['decode_gb_per_token']:>7.3f}")

    best = max(rows, key=lambda r: r["decode_tok_s"])
    print(f"\n[serve] best decode {best['decode_tok_s']:.2f} tok/s at {best['capacity']} slots "
          f"({best['vram_gb']:.2f} GB resident, {best['decode_hit_rate']:.1%} hit rate)")
    print("        Stage 0 measured the accelerate-offload baseline at 0.44 tok/s "
          "(2.25 s/token) on this model and card.")
    print("        Prefill's hit rate is low by construction, not by failure: a batch "
          "touches the union of its tokens' experts,")
    print("        which at these lengths is every expert in the layer. Only the "
          "decode column is comparable to Q5's 54.5%.")
    return 0


if __name__ == "__main__":
    sys.exit(collect_main())
