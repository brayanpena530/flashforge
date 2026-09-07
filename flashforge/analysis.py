"""The five Stage 0 questions.

Every function returns a DataFrame. Plots are separate and optional — the
numbers are the deliverable, the charts are for reading them quickly.

  Q1 skew            how concentrated is expert usage? -> is a small cache viable
  Q2 locality        does token t+1 reuse token t's experts? -> is LRU sensible
  Q3 predictability  can layer N predict layer N+k's routing? -> is prefetch viable
  Q4 domain          does usage cluster by domain? -> is cache warming viable
  Q5 cache           hit rate vs capacity, incl. Belady -> how much headroom exists

Q3 is the one that decides the project. Prefetch only pays if you have lead
time: predicting layer N+k at layer N buys you k layers of compute to hide the
transfer behind. If accuracy collapses at k=1, the design has to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# trace store
# --------------------------------------------------------------------------

@dataclass
class TraceStore:
    """Reader for a collected trace directory."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        with (self.root / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.routing = pd.read_parquet(self.root / "routing.parquet")

    @property
    def num_experts(self) -> int:
        return int(self.meta["num_experts"])

    @property
    def top_k(self) -> int:
        return int(self.meta["top_k"])

    @property
    def moe_layers(self) -> list[int]:
        return [int(x) for x in self.meta["moe_layers"]]

    @property
    def domains(self) -> dict[int, str]:
        return {int(k): v for k, v in self.meta["seq_domains"].items()}

    def seq_ids(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in (self.root / "logits").glob("seq_*.npz"))

    def logits(self, seq_id: int) -> dict[int, np.ndarray]:
        with np.load(self.root / "logits" / f"seq_{seq_id:05d}.npz") as data:
            return {int(k): data[k].astype(np.float32) for k in data.files}

    def hidden(self, seq_id: int) -> dict[int, np.ndarray]:
        path = self.root / "hidden" / f"seq_{seq_id:05d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No hidden states at {path}. Re-collect without --no-hidden to run Q3."
            )
        with np.load(path) as data:
            return {int(k): data[k].astype(np.float32) for k in data.files}

    def gates(self) -> dict[int, np.ndarray]:
        with np.load(self.root / "gates.npz") as data:
            return {int(k): data[k].astype(np.float32) for k in data.files}


# --------------------------------------------------------------------------
# Q1 — how skewed is expert usage?
# --------------------------------------------------------------------------

def _gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=np.float64))
    n = values.size
    total = values.sum()
    if n == 0 or total == 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * cumulative.sum() / total) / n)


def expert_frequency(frame: pd.DataFrame, num_experts: int) -> pd.DataFrame:
    """Access count per (layer, expert), including experts never routed to."""
    counts = (
        frame.groupby(["layer", "expert"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    layers = sorted(frame["layer"].unique())
    full = pd.MultiIndex.from_product(
        [layers, range(num_experts)], names=["layer", "expert"]
    ).to_frame(index=False)
    merged = full.merge(counts, on=["layer", "expert"], how="left").fillna({"count": 0})
    merged["count"] = merged["count"].astype(np.int64)
    merged["share"] = merged.groupby("layer")["count"].transform(lambda s: s / s.sum())
    return merged


def skew_summary(freq: pd.DataFrame, num_experts: int) -> pd.DataFrame:
    """Per-layer concentration metrics.

    top10pct_mass is the headline: the share of all accesses served by the
    hottest 10% of experts. High values mean a small pinned cache captures most
    traffic.
    """
    rows = []
    for layer, group in freq.groupby("layer"):
        counts = group["count"].to_numpy(np.float64)
        share = counts / counts.sum() if counts.sum() else counts
        ordered = np.sort(share)[::-1]
        cut = max(1, int(round(num_experts * 0.10)))
        nonzero = share[share > 0]
        entropy = -(nonzero * np.log(nonzero)).sum() if nonzero.size else 0.0
        rows.append(
            {
                "layer": int(layer),
                "gini": _gini(counts),
                "norm_entropy": entropy / np.log(num_experts),
                "top10pct_mass": float(ordered[:cut].sum()),
                "top25pct_mass": float(ordered[: max(1, num_experts // 4)].sum()),
                "unused_experts": int((counts == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def lorenz_curve(freq: pd.DataFrame) -> pd.DataFrame:
    """Cumulative access share vs cumulative expert share, pooled across layers."""
    counts = np.sort(freq["count"].to_numpy(np.float64))[::-1]
    total = counts.sum()
    return pd.DataFrame(
        {
            "expert_fraction": np.arange(1, counts.size + 1) / counts.size,
            "access_fraction": np.cumsum(counts) / total if total else counts,
        }
    )


# --------------------------------------------------------------------------
# Q2 — temporal locality
# --------------------------------------------------------------------------

def _membership(group: pd.DataFrame, num_experts: int) -> tuple[np.ndarray, np.ndarray]:
    positions = group["pos"].to_numpy()
    experts = group["expert"].to_numpy()
    unique_pos, inverse = np.unique(positions, return_inverse=True)
    matrix = np.zeros((unique_pos.size, num_experts), dtype=bool)
    matrix[inverse, experts] = True
    return unique_pos, matrix


def consecutive_overlap(
    frame: pd.DataFrame, num_experts: int, top_k: int, *, lags: tuple[int, ...] = (1, 2, 4, 8)
) -> pd.DataFrame:
    """Fraction of a token's experts that were also used `lag` tokens earlier."""
    rows = []
    for (seq_id, layer), group in frame.groupby(["seq_id", "layer"], observed=True):
        _, matrix = _membership(group, num_experts)
        if matrix.shape[0] < 2:
            continue
        for lag in lags:
            if matrix.shape[0] <= lag:
                continue
            overlap = (matrix[:-lag] & matrix[lag:]).sum(axis=1) / top_k
            rows.append(
                {
                    "seq_id": int(seq_id),
                    "layer": int(layer),
                    "lag": lag,
                    "mean_overlap": float(overlap.mean()),
                }
            )
    return pd.DataFrame(rows)


def reuse_distance(frame: pd.DataFrame) -> pd.DataFrame:
    """Token gap between consecutive uses of the same (layer, expert).

    Short reuse distances are what recency-based eviction exploits; a long tail
    is what a frequency-based policy catches instead.
    """
    rows = []
    ordered = frame.sort_values(["seq_id", "layer", "expert", "pos"], kind="stable")
    for (seq_id, layer, expert), group in ordered.groupby(
        ["seq_id", "layer", "expert"], observed=True
    ):
        positions = group["pos"].to_numpy()
        if positions.size < 2:
            continue
        gaps = np.diff(positions)
        rows.append(
            {
                "seq_id": int(seq_id),
                "layer": int(layer),
                "expert": int(expert),
                "median_gap": float(np.median(gaps)),
                "p90_gap": float(np.percentile(gaps, 90)),
                "n_reuses": int(gaps.size),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q3 — cross-layer predictability  (the decisive one)
# --------------------------------------------------------------------------

def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return part


def _ridge_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    """Closed-form ridge with intercept, via the normal equations.

    Equivalent to sklearn's Ridge(fit_intercept=True) but without the
    dependency — it is one solve of a (hidden_size x hidden_size) system, and
    the probe is the only thing that needed sklearn at all.
    """
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    xc = x_train - x_mean
    yc = y_train - y_mean
    gram = xc.T @ xc
    gram.flat[:: gram.shape[0] + 1] += alpha  # add alpha to the diagonal
    weights = np.linalg.solve(gram, xc.T @ yc)
    return (x_test - x_mean) @ weights + y_mean


def _recall_at(pred_idx: np.ndarray, true_matrix: np.ndarray, top_k: int) -> float:
    hits = np.take_along_axis(true_matrix, pred_idx, axis=1).sum(axis=1)
    return float((hits / top_k).mean())


def cross_layer_predictability(
    store: TraceStore,
    *,
    offsets: tuple[int, ...] = (1, 2, 4),
    budget_multipliers: tuple[int, ...] = (1, 2),
    max_tokens: int = 20000,
    train_fraction: float = 0.6,
    ridge_alpha: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Can layer N's hidden state predict which experts layer N+k will route to?

    Four predictors, cheapest first:

      prior         static global top-k for the target layer (the floor —
                    beat this or prefetch is pointless)
      identity      reuse layer N's own expert indices at layer N+k
      stale_router  run layer N+k's *actual* router on layer N's hidden state.
                    Free at inference time: no training, no extra parameters.
                    This is the one to beat, and often the one to ship.
      probe         ridge regression from layer N's hidden state to layer N+k's
                    routing probabilities. An approximate ceiling on what a
                    learned predictor of this size could do.

    budget_multipliers models over-prefetching: at 2x you fetch 2*top_k experts
    and hope the true top_k are among them. Recall is capped at 1.0.
    """
    rng = np.random.default_rng(seed)
    layers = store.moe_layers
    top_k = store.top_k
    num_experts = store.num_experts
    gates = store.gates()

    seq_ids = store.seq_ids()
    rng.shuffle(seq_ids)
    n_train = max(1, int(len(seq_ids) * train_fraction))
    train_ids, test_ids = set(seq_ids[:n_train]), set(seq_ids[n_train:])
    if not test_ids:  # tiny trace: fall back to evaluating in-sample
        test_ids = train_ids

    # Cache per-sequence arrays once; the loop below reuses them per layer pair.
    hidden_by_seq: dict[int, dict[int, np.ndarray]] = {}
    logits_by_seq: dict[int, dict[int, np.ndarray]] = {}
    budget_tokens = 0
    for seq_id in seq_ids:
        hidden_by_seq[seq_id] = store.hidden(seq_id)
        logits_by_seq[seq_id] = store.logits(seq_id)
        budget_tokens += next(iter(logits_by_seq[seq_id].values())).shape[0]
        if budget_tokens >= max_tokens:
            break

    loaded = list(hidden_by_seq)
    global_prior = (
        store.routing.groupby(["layer", "expert"], observed=True).size().rename("count").reset_index()
    )

    def stack(seq_subset, layer: int, source: dict) -> np.ndarray:
        parts = [source[s][layer] for s in seq_subset if s in source and layer in source[s]]
        return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), np.float32)

    rows = []
    for offset in offsets:
        for i, src_layer in enumerate(layers):
            j = i + offset
            if j >= len(layers):
                continue
            tgt_layer = layers[j]

            train_seqs = [s for s in loaded if s in train_ids]
            test_seqs = [s for s in loaded if s in test_ids]
            if not test_seqs:
                continue

            h_test = stack(test_seqs, src_layer, hidden_by_seq)
            tgt_test = stack(test_seqs, tgt_layer, logits_by_seq)
            src_test = stack(test_seqs, src_layer, logits_by_seq)
            if h_test.size == 0 or tgt_test.size == 0:
                continue

            true_idx = _topk_indices(tgt_test, top_k)
            true_matrix = np.zeros_like(tgt_test, dtype=bool)
            np.put_along_axis(true_matrix, true_idx, True, axis=1)

            # prior
            layer_prior = np.zeros(num_experts, np.float64)
            sub = global_prior[global_prior["layer"] == tgt_layer]
            layer_prior[sub["expert"].to_numpy()] = sub["count"].to_numpy()
            prior_scores = np.tile(layer_prior, (tgt_test.shape[0], 1))

            # stale router: target layer's router applied to source hidden state
            stale_scores = h_test @ gates[tgt_layer].T

            # learned probe
            probe_scores = None
            h_train = stack(train_seqs, src_layer, hidden_by_seq)
            y_train = stack(train_seqs, tgt_layer, logits_by_seq)
            if h_train.shape[0] >= 64 and h_train.shape[0] == y_train.shape[0]:
                targets = np.exp(y_train - y_train.max(axis=1, keepdims=True))
                targets /= targets.sum(axis=1, keepdims=True)
                probe_scores = _ridge_fit_predict(h_train, targets, h_test, ridge_alpha)

            candidates = {
                "prior": prior_scores,
                "identity": src_test,
                "stale_router": stale_scores,
            }
            if probe_scores is not None:
                candidates["probe"] = probe_scores

            for multiplier in budget_multipliers:
                budget = min(num_experts, top_k * multiplier)
                for name, scores in candidates.items():
                    pred_idx = _topk_indices(scores, budget)
                    rows.append(
                        {
                            "offset": offset,
                            "src_layer": int(src_layer),
                            "tgt_layer": int(tgt_layer),
                            "predictor": name,
                            "budget_mult": multiplier,
                            "recall": _recall_at(pred_idx, true_matrix, top_k),
                            "n_tokens": int(h_test.shape[0]),
                        }
                    )

    return pd.DataFrame(rows)


def predictability_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Average recall across layer pairs, per (offset, predictor, budget)."""
    return (
        detail.groupby(["offset", "budget_mult", "predictor"], observed=True)["recall"]
        .mean()
        .reset_index()
        .sort_values(["budget_mult", "offset", "recall"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Q4 — does usage cluster by domain?
# --------------------------------------------------------------------------

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def domain_profiles(frame: pd.DataFrame, domains: dict[int, str], num_experts: int) -> pd.DataFrame:
    """Normalised (layer, expert) usage distribution per domain."""
    tagged = frame.assign(domain=frame["seq_id"].map(domains))
    tagged = tagged.dropna(subset=["domain"])
    tagged["key"] = tagged["layer"].astype(np.int64) * num_experts + tagged["expert"]
    counts = tagged.groupby(["domain", "key"], observed=True).size().rename("count").reset_index()
    counts["share"] = counts.groupby("domain")["count"].transform(lambda s: s / s.sum())
    return counts


# --------------------------------------------------------------------------
# Q6 — expert-set expansion  (does batching tokens amortise expert loads?)
# --------------------------------------------------------------------------

def expert_set_expansion(
    frame: pd.DataFrame,
    num_experts: int,
    top_k: int,
    *,
    block_sizes: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
    max_sequences: int | None = None,
) -> pd.DataFrame:
    """Unique experts touched by a block of B consecutive tokens, per layer.

    When you process B tokens in one forward pass — speculative-decoding
    verification, or just prefetching a block ahead — you pay for the *union*
    of experts those tokens touch, not B times one token's worth. That union is
    the quantity that decides whether batching helps.

    Two regimes:

      slow growth   union(4) ~ 1.3 * top_k. One load serves four tokens, bytes
                    per token collapse, and speculative decoding is a large win
                    on memory-bound hardware before you even count acceptance.
      fast growth   union(4) ~ 3.5 * top_k ("expert scattering"). Batching
                    loads nearly B times the weights and speculation buys
                    little until the draft's expert footprint is constrained.

    Uses sliding windows rather than disjoint blocks — same expectation, more
    samples per sequence.
    """
    if max_sequences is not None:
        keep = sorted(frame["seq_id"].unique())[:max_sequences]
        frame = frame[frame["seq_id"].isin(keep)]

    rows = []
    for (seq_id, layer), group in frame.groupby(["seq_id", "layer"], observed=True):
        _, matrix = _membership(group, num_experts)
        n_positions = matrix.shape[0]
        for block in block_sizes:
            if n_positions < block:
                continue
            n_windows = n_positions - block + 1
            union = matrix[:n_windows].copy()
            for offset in range(1, block):
                union |= matrix[offset : offset + n_windows]
            rows.append(
                {
                    "seq_id": int(seq_id),
                    "layer": int(layer),
                    "block_size": int(block),
                    "mean_unique": float(union.sum(axis=1).mean()),
                }
            )
    return pd.DataFrame(rows)


def expansion_summary(detail: pd.DataFrame, num_experts: int, top_k: int) -> pd.DataFrame:
    """Aggregate expansion across layers, against a random-routing null model.

    random_unique is what you would see if each token drew an independent
    uniform top-k subset: E * (1 - (1 - k/E)^B). Observed union well below that
    line means genuine block-level reuse, not just the birthday-problem
    arithmetic of drawing repeatedly from a small pool.

    bytes_amortization is the headline: B * top_k / union(B), the factor by
    which bytes-fetched-per-token drops when B tokens share one set of loads.
    It is 1.0 at B=1 by construction; higher is better, and B is the ceiling.
    """
    summary = (
        detail.groupby("block_size", observed=True)["mean_unique"].mean().reset_index()
    )
    blocks = summary["block_size"].to_numpy(np.float64)
    summary["expansion_ratio"] = summary["mean_unique"] / top_k
    summary["random_unique"] = num_experts * (1.0 - (1.0 - top_k / num_experts) ** blocks)
    summary["random_ratio"] = summary["random_unique"] / top_k
    summary["bytes_amortization"] = blocks * top_k / summary["mean_unique"]
    return summary


@dataclass
class DomainResult:
    """Domain-clustering verdict.

    matrix is the best estimate of between-domain divergence, computed on all
    available data — use it for display. The verdict, however, comes from
    between_matched vs within_matched, both computed on *half*-sized samples.

    That distinction matters: a JS divergence estimated from fewer tokens is
    biased upward simply because sparse profiles look different by chance.
    Comparing a full-data between-domain number against a half-data
    within-domain baseline makes the noise floor look artificially high and can
    hide a real domain signal entirely. Both sides are therefore measured on
    equal-sized samples.
    """

    matrix: pd.DataFrame
    between_matched: float
    within_matched: float

    @property
    def separates(self) -> bool:
        """True if domains diverge clearly beyond the sampling noise floor."""
        if not np.isfinite(self.within_matched) or self.within_matched <= 0:
            return False
        return self.between_matched > 2 * self.within_matched

    @property
    def ratio(self) -> float:
        return self.between_matched / self.within_matched if self.within_matched else float("nan")


def domain_divergence(
    frame: pd.DataFrame, domains: dict[int, str], num_experts: int, *, seed: int = 0
) -> DomainResult:
    """Jensen-Shannon divergence between domain routing profiles."""
    rng = np.random.default_rng(seed)
    tagged = frame.assign(domain=frame["seq_id"].map(domains)).dropna(subset=["domain"])
    n_keys = int(frame["layer"].max() + 1) * num_experts

    def profile(subset: pd.DataFrame) -> np.ndarray:
        key = subset["layer"].to_numpy(np.int64) * num_experts + subset["expert"].to_numpy(np.int64)
        vector = np.bincount(key, minlength=n_keys).astype(np.float64)
        total = vector.sum()
        return vector / total if total else vector

    names = sorted(tagged["domain"].unique())
    by_domain = {name: tagged[tagged["domain"] == name] for name in names}

    # Full-data matrix: the better estimate, for display.
    full = {name: profile(subset) for name, subset in by_domain.items()}
    matrix = pd.DataFrame(
        [[_js_divergence(full[a], full[b]) for b in names] for a in names],
        index=names,
        columns=names,
    )

    # Sample-matched halves: both sides of the verdict use the same token budget.
    halves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, subset in by_domain.items():
        seqs = subset["seq_id"].unique()
        if seqs.size < 2:
            continue
        shuffled = rng.permutation(seqs)
        cut = max(1, shuffled.size // 2)
        left = subset[subset["seq_id"].isin(shuffled[:cut])]
        right = subset[subset["seq_id"].isin(shuffled[cut:])]
        if left.empty or right.empty:
            continue
        halves[name] = (profile(left), profile(right))

    within = [_js_divergence(left, right) for left, right in halves.values()]
    between = [
        _js_divergence(halves[a][0], halves[b][0])
        for i, a in enumerate(halves)
        for b in list(halves)[i + 1:]
    ]

    return DomainResult(
        matrix=matrix,
        between_matched=float(np.mean(between)) if between else float("nan"),
        within_matched=float(np.mean(within)) if within else float("nan"),
    )
