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

from flashforge import analysis, cachesim, plots, viz  # noqa: E402

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
            # Layer-specific drift: correlated with, but not equal to, its neighbours.
            drift = rng.normal(0, 0.12, (T, H)).astype(np.float32) * (layer + 1) ** 0.5
            hidden = base + drift
            logits = hidden @ gates[layer].T + popularity[layer] + domain_bias[domain][layer]
            hidden_by_layer[str(layer)] = hidden.astype(np.float16)
            logits_by_layer[str(layer)] = logits.astype(np.float16)

            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
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
plots.plot_skew(lorenz, skew).savefig(out / "q1.png")

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
plots.plot_cache(sweep, total_slots=total_slots).savefig(out / "q5.png")

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("All checks passed. Charts written to", out)
