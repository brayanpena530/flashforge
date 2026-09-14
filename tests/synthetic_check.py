"""Synthetic end-to-end validation of the Stage 0 analysis stack.

Plants known structure (skew, temporal locality, cross-layer predictability,
domain clustering) and checks each analysis actually recovers it. Needs no
model download and no GPU — run it after touching anything in analysis.py,
cachesim.py, or plots.py:

    uv run python tests/synthetic_check.py

Takes a couple of minutes; the Belady sweep dominates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

from flashforge import analysis, cachesim, hardware, plots, viz  # noqa: E402

ROOT = Path(__file__).parent / "_synth_traces"
E, K, L, H = 64, 8, 16, 128
DOMAINS = ["code", "math", "prose", "dialogue"]
SEQS_PER_DOMAIN, T = 6, 180

rng = np.random.default_rng(7)
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "logits").mkdir(exist_ok=True)
(ROOT / "hidden").mkdir(exist_ok=True)

# Router matrices, plus a per-layer popularity bias to create realistic skew.
gates = {layer: rng.normal(0, 0.35, (E, H)).astype(np.float32) for layer in range(L)}
popularity = {
    layer: (rng.pareto(1.6, E) * 0.8).astype(np.float32) for layer in range(L)
}
# Each domain prefers a different slice of experts.
domain_bias = {
    d: {layer: (rng.normal(0, 3.0, E) * (rng.random(E) < 0.30)).astype(np.float32)
        for layer in range(L)}
    for d in DOMAINS
}

records, seq_domains = [], {}
seq_id = 0
for domain in DOMAINS:
    for _ in range(SEQS_PER_DOMAIN):
        # Slow random walk -> temporal locality in routing.
        steps = rng.normal(0, 0.18, (T, H)).astype(np.float32)
        base = np.cumsum(steps, axis=0) + rng.normal(0, 1.0, (1, H)).astype(np.float32)

        logits_by_layer, hidden_by_layer = {}, {}
        for layer in range(L):
            # Layer-specific drift: correlated with, but not equal to, its
            # neighbours. The scale is U-shaped in depth — edge layers wander
            # further from the shared base than middle layers do — which plants
            # the layer-group structure Q3's band split exists to detect.
            # Middle layers should therefore come out more cross-layer
            # predictable than either end.
            # The spread has to be wide to be visible: stale_router is also
            # blind to the per-layer popularity term, and that error floor is
            # layer-invariant. A narrow drift range gets lost underneath it.
            edge = abs(layer / max(1, L - 1) - 0.5) * 2  # 1 at the ends, 0 mid-stack
            drift = rng.normal(0, 0.05 + 1.0 * edge, (T, H)).astype(np.float32)
            hidden = base + drift
            logits = hidden @ gates[layer].T + popularity[layer] + domain_bias[domain][layer]
            hidden_by_layer[str(layer)] = hidden.astype(np.float16)
            logits_by_layer[str(layer)] = logits.astype(np.float16)

            # Peaked routing at the edges, flat mid-stack: plants the
            # weight-dominance pattern Q1's band split looks for. Top-k
            # selection is invariant to a positive temperature scale, so this
            # moves the `weight` column only and leaves every other question's
            # input — which experts get chosen — exactly as it was.
            temperature = 0.35 + 1.4 * (1 - edge)
            scaled = logits / temperature
            probs = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            top = np.argpartition(-probs, K - 1, axis=1)[:, :K]
            order = np.argsort(-np.take_along_axis(probs, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)
            weights = np.take_along_axis(probs, top, axis=1)

            records.append(pd.DataFrame({
                "seq_id": np.full(T * K, seq_id, np.int32),
                "pos": np.repeat(np.arange(T, dtype=np.int32), K),
                "layer": np.full(T * K, layer, np.int16),
                "rank": np.tile(np.arange(K, dtype=np.int8), T),
                "expert": top.astype(np.int16).ravel(),
                "weight": weights.astype(np.float32).ravel(),
            }))

        np.savez_compressed(ROOT / "logits" / f"seq_{seq_id:05d}.npz", **logits_by_layer)
        np.savez(ROOT / "hidden" / f"seq_{seq_id:05d}.npz", **hidden_by_layer)
        seq_domains[str(seq_id)] = domain
        seq_id += 1

frame = pd.concat(records, ignore_index=True).sort_values(
    ["seq_id", "pos", "layer", "rank"], ignore_index=True)
frame.to_parquet(ROOT / "routing.parquet", engine="pyarrow", index=False)
np.savez(ROOT / "gates.npz", **{str(k): v for k, v in gates.items()})
json.dump({
    "model_id": "synthetic", "num_experts": E, "top_k": K, "hidden_size": H,
    "norm_topk_prob": False, "moe_layers": list(range(L)), "seq_domains": seq_domains,
    "n_sequences": seq_id, "total_tokens": seq_id * T, "saved_hidden": True,
}, (ROOT / "meta.json").open("w"), indent=2)

print(f"synthetic trace: {len(frame):,} rows, {seq_id} sequences\n")

# ---------------------------------------------------------------- checks
store = analysis.TraceStore(ROOT)
viz.use_style()
out = ROOT / "report"
out.mkdir(exist_ok=True)
failures = []

def check(name, condition, detail):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}: {detail}")
    if not condition:
        failures.append(name)

print("Q1 skew")
freq = analysis.expert_frequency(frame, E)
skew = analysis.skew_summary(freq, E)
lorenz = analysis.lorenz_curve(freq)
check("planted skew detected", skew["top10pct_mass"].mean() > 0.10,
      f"top-10% mass {skew['top10pct_mass'].mean():.1%} (uniform 10.0%)")
check("gini in range", 0 <= skew["gini"].mean() <= 1, f"gini {skew['gini'].mean():.3f}")
check("lorenz monotonic", bool(np.all(np.diff(lorenz["access_fraction"]) >= -1e-9)),
      f"ends at {lorenz['access_fraction'].iloc[-1]:.3f}")
check("skew is reported per layer", len(skew) == L, f"{len(skew)} rows for {L} layers")
plots.plot_skew(lorenz, skew).savefig(out / "q1.png")

print("\nQ1 depth profile")
wprof = analysis.routing_weight_profile(frame, K)
check("weight profile covers every layer", len(wprof) == L, f"{len(wprof)} rows")
check("top-1 share is a share", bool(wprof["top1_share"].between(1.0 / K, 1.0).all()),
      f"range {wprof['top1_share'].min():.3f}-{wprof['top1_share'].max():.3f} "
      f"(even split would be {1/K:.3f})")
check("weight entropy is normalised", bool(wprof["weight_entropy"].between(0, 1).all()),
      f"range {wprof['weight_entropy'].min():.3f}-{wprof['weight_entropy'].max():.3f}")
check("every token contributed once per layer",
      bool((wprof["n_tokens"] == seq_id * T).all()),
      f"{int(wprof['n_tokens'].iloc[0])} tokens per layer")

bands = analysis.band_summary(skew, wprof, list(range(L)))
check("band summary has three rows", len(bands) == 3, f"{list(bands['band'])}")
check("band layer counts sum to the stack", int(bands["n_layers"].sum()) == L,
      f"{dict(zip(bands['band'], bands['n_layers']))}")
# The generator plants peaked routing at the edges and flat routing mid-stack.
# Recovering that ordering is what proves the weight column is being read
# correctly — counting accesses alone cannot see this at all.
keyed = bands.set_index("band")
check("planted weight dominance recovered",
      min(keyed.loc["input", "top1_share"], keyed.loc["output", "top1_share"])
      > keyed.loc["middle", "top1_share"],
      f"input {keyed.loc['input','top1_share']:.3f} / "
      f"output {keyed.loc['output','top1_share']:.3f} vs "
      f"middle {keyed.loc['middle','top1_share']:.3f}")

ragged = frame.iloc[:-1]
try:
    analysis.routing_weight_profile(ragged, K)
    check("ragged trace is rejected", False, "no error raised")
except ValueError as exc:
    check("ragged trace is rejected", True, f"{type(exc).__name__} raised, not silent misalignment")

plots.plot_layer_bands(
    analysis.annotate_bands(skew.merge(wprof, on="layer"), list(range(L))), top_k=K
).savefig(out / "q1_bands.png")

print("\nQ2 locality")
overlap = analysis.consecutive_overlap(frame, E, K)
lag1 = overlap[overlap.lag == 1]["mean_overlap"].mean()
lag8 = overlap[overlap.lag == 8]["mean_overlap"].mean()
check("lag-1 beats random", lag1 > K / E, f"lag-1 {lag1:.1%} vs random {K/E:.1%}")
check("overlap decays with lag", lag1 > lag8, f"lag-1 {lag1:.1%} > lag-8 {lag8:.1%}")
reuse = analysis.reuse_distance(frame)
check("reuse distances computed", len(reuse) > 0, f"{len(reuse)} (seq,layer,expert) groups")
plots.plot_locality(overlap).savefig(out / "q2.png")

print("\nQ3 predictability")
detail = analysis.cross_layer_predictability(store, offsets=(1, 2, 4), max_tokens=8000)
summary = analysis.predictability_summary(detail)
at1 = summary[(summary.offset == 1) & (summary.budget_mult == 1)].set_index("predictor")["recall"]
check("stale_router beats prior", at1["stale_router"] > at1["prior"],
      f"stale {at1['stale_router']:.3f} vs prior {at1['prior']:.3f}")
check("recall in [0,1]", bool(summary["recall"].between(0, 1).all()),
      f"max {summary['recall'].max():.3f}")
at2 = summary[(summary.offset == 1) & (summary.budget_mult == 2)].set_index("predictor")["recall"]
check("2x budget raises recall", at2["stale_router"] >= at1["stale_router"],
      f"1x {at1['stale_router']:.3f} -> 2x {at2['stale_router']:.3f}")
check("all four predictors present", len(at1) == 4, f"{sorted(at1.index)}")
plots.plot_predictability(summary, K).savefig(out / "q3.png")

print("\nQ3 layer bands")
bands = analysis.layer_groups(list(range(L)))
check("bands partition the stack", set(bands) == set(range(L)),
      f"{len(bands)} layers assigned")
check("all three bands populated", set(bands.values()) == {"input", "middle", "output"},
      f"{ {g: sum(v == g for v in bands.values()) for g in ('input','middle','output')} }")
check("bands are contiguous in depth",
      [bands[i] for i in range(L)] == sorted([bands[i] for i in range(L)],
                                             key=lambda g: {"input": 0, "middle": 1, "output": 2}[g]),
      "input -> middle -> output, no interleaving")
check("short stacks degrade gracefully",
      set(analysis.layer_groups([0, 1]).values()) == {"middle"},
      "a 2-layer stack is all middle rather than an empty band")

by_group = analysis.predictability_by_group(detail)
check("group summary carries every band", set(by_group["src_group"]) <= {"input", "middle", "output"},
      f"{sorted(set(by_group['src_group']))}")
stale1 = by_group[(by_group.predictor == "stale_router") & (by_group.offset == 1)
                  & (by_group.budget_mult == 1)].set_index("src_group")["recall"]
# The generator plants exactly this: middle layers drift least, so they should
# be the most predictable band. This is the positive control for the split —
# without it, bucketing by band could be pure noise and nothing would notice.
check("planted band structure recovered",
      stale1["middle"] > max(stale1["input"], stale1["output"]),
      f"middle {stale1['middle']:.3f} vs input {stale1['input']:.3f} / "
      f"output {stale1['output']:.3f}")

deep = analysis.cross_layer_predictability(store, offsets=(1, 2, 4, 8), max_tokens=6000)
deep_summary = analysis.predictability_summary(deep)
stale_by_offset = deep_summary[(deep_summary.predictor == "stale_router")
                               & (deep_summary.budget_mult == 1)].set_index("offset")["recall"]
check("deeper lookahead is reachable", 8 in stale_by_offset.index,
      f"offsets measured: {sorted(stale_by_offset.index)}")
check("accuracy decays with lookahead depth",
      stale_by_offset[1] > stale_by_offset[8],
      f"k=1 {stale_by_offset[1]:.3f} -> k=8 {stale_by_offset[8]:.3f}")

reach = analysis.deepest_usable_lookahead(
    analysis.predictability_by_group(deep), threshold=0.5)
check("reach table covers every band", len(reach) == 3, f"{list(reach['src_group'])}")
check("reach is bounded by the offsets swept",
      bool((reach["deepest_offset"] <= 8).all()),
      f"max {int(reach['deepest_offset'].max())}")
# The output band physically cannot have an 8-layer-ahead pair — there is no
# layer 8 past the end of the stack. That has to be reported as missing data,
# not as a prediction failure, or the disk-tier verdict manufactures a ceiling
# out of arithmetic.
check("output band is limited by coverage, not accuracy",
      reach.set_index("src_group").loc["output", "limited_by"] == "coverage",
      f"output measured only to offset "
      f"{int(reach.set_index('src_group').loc['output', 'deepest_offset_measured'])} of 8")
check("mid-stack bands are judged on accuracy",
      reach.set_index("src_group").loc["input", "limited_by"] in {"accuracy", "none"},
      f"input limited_by={reach.set_index('src_group').loc['input', 'limited_by']}")

deep_groups = analysis.predictability_by_group(deep)
all_reach = analysis.lookahead_reach(deep_groups, threshold=0.5)
check("reach covers every band x predictor pair",
      len(all_reach) == 3 * deep_groups["predictor"].nunique(),
      f"{len(all_reach)} rows for 3 bands x {deep_groups['predictor'].nunique()} predictors")
check("single-predictor view agrees with the all-predictor view",
      all_reach[all_reach.predictor == 'stale_router']
        .set_index('src_group')['deepest_offset'].to_dict()
      == reach.set_index('src_group')['deepest_offset'].to_dict(),
      "deepest_usable_lookahead is a filtered view, not a second implementation")
best = analysis.best_lookahead_by_band(all_reach)
check("best-per-band picks one predictor per band", len(best) == 3,
      ", ".join(f"{r.src_group}:{r.predictor}@{int(r.deepest_offset)}" for r in best.itertuples()))
# The winner must genuinely be a winner: no other predictor may reach deeper.
merged = all_reach.merge(best[['src_group','deepest_offset']], on='src_group',
                         suffixes=('', '_best'))
check("no predictor beats the chosen best",
      bool((merged['deepest_offset'] <= merged['deepest_offset_best']).all()),
      "selection is a true argmax over depth")
plots.plot_predictability_groups(by_group, K, required_lookahead=4).savefig(out / "q3_groups.png")

print("\nphase split")
# Traces predating is_decode, and traces collected without --gen-tokens, must
# still analyse cleanly rather than silently returning an empty decode frame.
legacy = analysis.phase_frames(frame)
check("trace without is_decode is all prefill",
      set(legacy) == {"prefill"} and len(legacy["prefill"]) == len(frame),
      f"keys {sorted(legacy)}, {len(legacy['prefill']):,} rows")

tagged = frame.copy()
tagged["is_decode"] = (tagged["pos"] >= T - 20).astype("int8")
split = analysis.phase_frames(tagged)
check("tagged trace splits into both phases", set(split) == {"prefill", "decode"},
      f"prefill {len(split['prefill']):,} rows, decode {len(split['decode']):,} rows")
check("split is lossless and disjoint",
      len(split["prefill"]) + len(split["decode"]) == len(tagged)
      and split["prefill"]["pos"].max() < split["decode"]["pos"].min(),
      "every row lands in exactly one phase, and the phases do not interleave")

allpre = frame.copy(); allpre["is_decode"] = np.int8(0)
check("all-prefill tagged trace reports only prefill",
      set(analysis.phase_frames(allpre)) == {"prefill"},
      "no empty decode frame is handed downstream")

print("\nQ4 domain")
# Positive and negative controls. The main synthetic trace gives every sequence
# its own random walk through hidden space, which swamps any domain term, so it
# is not a usable test of the metric. These two frames isolate it: identical
# generators except that `strength` switches the domain preference on and off.
def domain_frame(strength, *, n_domains=4, seqs=8, T=200, layers=8, seed=11):
    gen = np.random.default_rng(seed)
    parts, tags = [], {}
    sid = 0
    pools = [gen.choice(E, size=E // 3, replace=False) for _ in range(n_domains)]
    for d in range(n_domains):
        weights = np.ones(E)
        weights[pools[d]] += strength * 12.0
        logp = np.log(weights / weights.sum())[None, :]
        for _ in range(seqs):
            for layer in range(layers):
                # Gumbel top-k: samples K distinct experts proportional to weights.
                scores = logp + gen.gumbel(size=(T, E))
                top = np.argpartition(-scores, K - 1, axis=1)[:, :K]
                parts.append(pd.DataFrame({
                    "seq_id": np.full(T * K, sid, np.int32),
                    "pos": np.repeat(np.arange(T, dtype=np.int32), K),
                    "layer": np.full(T * K, layer, np.int16),
                    "rank": np.tile(np.arange(K, dtype=np.int8), T),
                    "expert": top.astype(np.int16).ravel(),
                    "weight": np.ones(T * K, np.float32),
                }))
            tags[sid] = f"domain_{d}"
            sid += 1
    return pd.concat(parts, ignore_index=True), tags

pos_frame, pos_tags = domain_frame(strength=1.0)
neg_frame, neg_tags = domain_frame(strength=0.0)
positive = analysis.domain_divergence(pos_frame, pos_tags, E)
negative = analysis.domain_divergence(neg_frame, neg_tags, E)

check("positive control separates", positive.separates,
      f"between {positive.between_matched:.4f} vs within {positive.within_matched:.4f} "
      f"({positive.ratio:.1f}x)")
check("negative control does NOT separate", not negative.separates,
      f"between {negative.between_matched:.4f} vs within {negative.within_matched:.4f} "
      f"({negative.ratio:.1f}x)")

result = analysis.domain_divergence(frame, store.domains, E)
print(f"        (main synthetic trace, no assertion: {result.ratio:.1f}x — "
      f"per-sequence variance dominates by construction)")
check("diagonal is zero", bool(np.allclose(np.diag(result.matrix.to_numpy()), 0)),
      "self-divergence 0")
plots.plot_domain(positive).savefig(out / "q4.png")

print("\nQ6 expansion")
expansion_detail = analysis.expert_set_expansion(frame, E, K, block_sizes=(1, 2, 4, 8))
expansion = analysis.expansion_summary(expansion_detail, E, K)
idx = expansion.set_index("block_size")
check("B=1 is the identity", abs(idx.loc[1, "expansion_ratio"] - 1.0) < 1e-9,
      f"ratio {idx.loc[1, 'expansion_ratio']:.4f}, amortization "
      f"{idx.loc[1, 'bytes_amortization']:.4f}")
check("expansion is monotonic in block size",
      bool(np.all(np.diff(expansion["expansion_ratio"].to_numpy()) >= -1e-9)),
      f"{[round(v, 2) for v in expansion['expansion_ratio']]}")
check("expansion bounded by B (union cannot exceed B*top_k)",
      bool((expansion["expansion_ratio"] <= expansion["block_size"] + 1e-9).all()),
      f"max ratio {expansion['expansion_ratio'].max():.2f} at B="
      f"{int(expansion['block_size'].max())}")
# The generator plants strong temporal locality, so consecutive tokens should
# reuse experts far more than independent uniform draws would. B=1 is excluded:
# with a single token there is no union to grow, so observed and null are both
# exactly 1.0 by construction and no strict inequality can hold.
multi = expansion[expansion["block_size"] > 1]
check("sticky routing beats the random-routing null",
      bool((multi["expansion_ratio"] < multi["random_ratio"] - 1e-6).all()),
      f"B=4: observed {idx.loc[4, 'expansion_ratio']:.2f}x vs "
      f"random {idx.loc[4, 'random_ratio']:.2f}x")
check("amortization exceeds 1 for B>1", bool((idx.loc[[2, 4, 8], "bytes_amortization"] > 1).all()),
      f"B=4 gives {idx.loc[4, 'bytes_amortization']:.2f}x fewer bytes/token")

# Independent-routing control: with no temporal locality the observed union
# must land on the random-routing null, not below it.
gen = np.random.default_rng(23)
iid_parts = []
for s in range(6):
    for layer in range(8):
        scores = gen.gumbel(size=(300, E))
        top = np.argpartition(-scores, K - 1, axis=1)[:, :K]
        iid_parts.append(pd.DataFrame({
            "seq_id": np.full(300 * K, s, np.int32),
            "pos": np.repeat(np.arange(300, dtype=np.int32), K),
            "layer": np.full(300 * K, layer, np.int16),
            "rank": np.tile(np.arange(K, dtype=np.int8), 300),
            "expert": top.astype(np.int16).ravel(),
            "weight": np.ones(300 * K, np.float32),
        }))
iid = pd.concat(iid_parts, ignore_index=True)
iid_summary = analysis.expansion_summary(
    analysis.expert_set_expansion(iid, E, K, block_sizes=(1, 2, 4, 8)), E, K)
worst = (iid_summary["expansion_ratio"] - iid_summary["random_ratio"]).abs().max()
check("i.i.d. control matches the null model", worst < 0.05,
      f"max deviation {worst:.4f} across block sizes")

plots.plot_expansion(expansion, K).savefig(out / "q6.png")

print("\nQ5 cache")
keys = cachesim.build_access_sequence(frame, E)
total_slots = E * L
caps = [int(total_slots * f) for f in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00)]
sweep = cachesim.sweep(keys[:120_000], caps, bytes_per_expert=12.6e6,
                       accesses_per_token=K * L)
pivot = sweep.pivot(index="capacity", columns="policy", values="hit_rate")
# Belady bounds DEMAND-PAGING policies only. `static` prefetches at t=0 and so
# skips compulsory misses, which is exactly why it can edge ahead.
check("belady >= demand-paging policies",
      bool((pivot["belady"] >= pivot[["lru", "lfu"]].max(axis=1) - 1e-9).all()),
      f"min margin vs lru/lfu {(pivot['belady'] - pivot[['lru','lfu']].max(axis=1)).min():.4f}")
compulsory = pivot["static"] - pivot["belady"]
check("static's edge over belady is bounded by compulsory misses",
      bool((compulsory <= pivot.index.to_numpy() / 120_000 + 1e-9).all()),
      f"max edge {compulsory.max():.4f} (prefetch beats optimal reactive)")
check("hit rate monotonic in capacity",
      bool(all(np.all(np.diff(pivot[p].to_numpy()) >= -1e-9) for p in pivot.columns)),
      "all policies non-decreasing")
check("full capacity ~ perfect", pivot.loc[total_slots, "belady"] > 0.99,
      f"belady at full capacity {pivot.loc[total_slots, 'belady']:.4f}")
check("bytes/token column present", "fetch_bytes_per_token" in sweep.columns,
      f"{sweep['fetch_bytes_per_token'].min()/1e6:.1f}-{sweep['fetch_bytes_per_token'].max()/1e6:.1f} MB/token")

# Budgeting the cache sim by truncating the flattened key stream takes the
# first few documents rather than a sample of them, and the policy ranking
# depends on how many documents are in view. On the real OLMoE trace a 300k
# prefix covered 5 of 48 sequences and reversed the LRU/LFU ordering.
budget = len(frame) // 4
trimmed, kept, available = cachesim.subsample_sequences(frame, budget)
check("subsampling keeps whole sequences", kept < available and kept > 0,
      f"kept {kept} of {available} sequences for a {budget:,}-row budget")
check("every kept sequence is complete",
      trimmed.groupby("seq_id").size().nunique() == frame.groupby("seq_id").size().nunique(),
      "no sequence is cut mid-stream")
check("subsampling respects the budget", len(trimmed) <= budget * 1.05,
      f"{len(trimmed):,} rows against a {budget:,} budget")
check("a budget above the trace is a no-op",
      cachesim.subsample_sequences(frame, len(frame) * 2)[0].equals(frame),
      "no sampling when the whole trace fits")

# The count of sequences a prefix reaches can match the sampled count when
# sequences are equal length — the damage is *which* ones. Corpora are usually
# grouped (this project's is ordered by domain), so a prefix is a monoculture:
# on the real trace the 300k prefix was five `code` documents and nothing else.
# What matters is that the sample is not a contiguous run from the start.
prefix_ids = set(frame.iloc[:budget]["seq_id"].unique())
sampled_ids = set(trimmed["seq_id"].unique())
check("sample is not a contiguous prefix",
      sampled_ids != prefix_ids and sampled_ids != set(range(len(sampled_ids))),
      f"sampled {sorted(sampled_ids)} vs prefix {sorted(prefix_ids)}")
check("sample spans the corpus",
      max(sampled_ids) > max(prefix_ids),
      f"sampling reaches seq {max(sampled_ids)}, prefix stops at {max(prefix_ids)}")
plots.plot_cache(sweep, total_slots=total_slots).savefig(out / "q5.png")

print("\nQ7 cost model")
# The fit is checked against a curve with known coefficients, not against a
# live measurement. A microbenchmark on a busy desktop is genuinely noisy —
# asserting r2 on it would be testing whether the machine happened to be quiet,
# which flakes and tells us nothing about the code.
known = pd.DataFrame({"tokens": [1, 2, 4, 8, 16, 32, 64, 128, 256]})
known["ms"] = 0.05 * known["tokens"] + 2.0
exact = hardware.fit_linear_cost(known)
check("fit recovers known coefficients",
      abs(exact.beta_ms_per_token - 0.05) < 1e-9 and abs(exact.const_ms - 2.0) < 1e-9,
      f"beta {exact.beta_ms_per_token:.6f} (0.05), const {exact.const_ms:.6f} (2.0)")
check("noiseless fit is exact", abs(exact.r_squared - 1.0) < 1e-9, f"r2 {exact.r_squared:.6f}")

noisy = known.copy()
noisy["ms"] += np.random.default_rng(3).normal(0, 0.05, len(noisy))
jittered = hardware.fit_linear_cost(noisy)
check("fit is robust to small noise",
      abs(jittered.beta_ms_per_token - 0.05) < 0.005 and jittered.r_squared > 0.99,
      f"beta {jittered.beta_ms_per_token:.5f}, r2 {jittered.r_squared:.4f}")

# The live curve is here to prove the harness runs and the shape is right, so
# it is asserted only on direction — never on goodness of fit.
cpu_curve = hardware.cpu_expert_curve(
    256, 512, token_counts=(1, 4, 16, 64, 256), repeats=5, warmup=2)
fit = hardware.fit_linear_cost(cpu_curve)
check("cpu curve covers every token count", len(cpu_curve) == 5, f"{list(cpu_curve['tokens'])}")

# A 256x512 expert on one token is sub-millisecond on any machine that is not
# busy. Well above that means another process owned the CPU for whole
# scheduling quanta, and the curve is measuring contention rather than cost.
# Asserting through that tests whether the machine happened to be idle, which
# is not what this suite is for -- the arithmetic is already covered above
# against known coefficients.
single = float(cpu_curve.loc[cpu_curve["tokens"] == 1, "ms"].iloc[0])
if single > 2.0:
    print(f"        SKIPPED directional checks: 1-token call took {single:.2f} ms, "
          f"~{single / 0.08:.0f}x the idle-machine figure. Curve: "
          f"{[round(v, 2) for v in cpu_curve['ms']]} ms")
else:
    check("cost rises with token count",
          bool(np.all(np.diff(cpu_curve["ms"].to_numpy()) > 0)),
          f"{[round(v, 3) for v in cpu_curve['ms']]} ms")
    check("beta is positive", fit.beta_ms_per_token > 0,
          f"beta {fit.beta_ms_per_token:.5f} ms/token, const {fit.const_ms:.4f} ms")
    print(f"        (live r2 {fit.r_squared:.4f} — reported, not asserted; "
          f"ff-bench warns below 0.95 on real shapes)")

# Break-even is pure arithmetic, so it can be asserted exactly rather than
# measured. t_c = 0.1m + 1.0 crosses a 5 ms GPU path at m = 40.
synthetic_fit = hardware.LinearCost(0.1, 1.0, 1.0)
check("break-even solves the crossing",
      abs(hardware.break_even_tokens(synthetic_fit, 5.0) - 40.0) < 1e-9,
      f"m* {hardware.break_even_tokens(synthetic_fit, 5.0):.1f} for t_c=0.1m+1 vs 5ms")
check("break-even clamps at zero when the GPU always wins",
      hardware.break_even_tokens(synthetic_fit, 0.5) == 0.0,
      "a GPU path cheaper than the CPU's startup cost yields m*=0")
check("expert bytes match the SwiGLU triple",
      hardware.expert_bytes(2048, 1024, 2.0) == 3 * 2048 * 1024 * 2,
      f"{hardware.expert_bytes(2048, 1024, 2.0) / 1e6:.1f} MB for OLMoE at fp16")
plots.plot_cost_model(cpu_curve, fit, gpu_path_ms=float(cpu_curve["ms"].median()),
                      break_even=hardware.break_even_tokens(
                          fit, float(cpu_curve["ms"].median()))
                      ).savefig(out / "q7.png")

print("\nQ8 storage curve")
scratch = ROOT / "scratch.bin"
storage = hardware.storage_read_curve(
    scratch,
    file_size_bytes=64 << 20,
    read_sizes=(64 << 10, 1 << 20),
    queue_depths=(1, 2, 4),
    target_bytes_per_point=8 << 20,
)
knees = hardware.saturation_knee(storage)
check("curve covers every size x depth", len(storage) == 6,
      f"{len(storage)} rows, peak {storage['gbps'].max():.2f} GB/s")
check("bandwidth is positive everywhere", bool((storage["gbps"] > 0).all()),
      f"min {storage['gbps'].min():.2f} GB/s")
check("latency rises with request size",
      storage[storage.read_bytes == (1 << 20)]["mean_ms"].mean()
      > storage[storage.read_bytes == (64 << 10)]["mean_ms"].mean(),
      "1 MiB reads take longer than 64 KiB reads")
check("percentiles are ordered", bool((storage["p99_ms"] >= storage["p50_ms"]).all()),
      "p99 >= p50 on every row")
check("knee is one of the depths swept",
      bool(knees["knee_queue_depth"].isin([1, 2, 4]).all()),
      f"knees {list(knees['knee_queue_depth'])}")
check("knee row exists per read size", len(knees) == 2, f"{list(knees['read_mib'])}")

disk_ms, provenance = hardware.disk_time_for_expert(storage, 12.6e6)
check("disk time is positive and attributed", disk_ms > 0 and "queue_depth" in provenance,
      f"{disk_ms:.2f} ms for a 12.6 MB expert, from "
      f"{provenance['measured_read_bytes'] >> 10} KiB reads")
check("required lookahead rounds up",
      hardware.required_lookahead(10.0, 3.0) == 4,
      "10 ms hidden behind 3 ms layers needs 4 layers, not 3")
check("required lookahead is zero when the layer time is unknown",
      hardware.required_lookahead(10.0, 0.0) == 0, "no division by zero")
plots.plot_storage(storage, knees).savefig(out / "q8.png")
scratch.unlink(missing_ok=True)

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("All checks passed. Charts written to", out)
