"""Expert cache simulation over a routing trace (Stage 0, question 5).

The access sequence is what a *decoder* sees: for each token position, walk the
layers in order and touch that layer's top-k experts. Cache keys are
(layer, expert) pairs — expert 3 of layer 0 is a different tensor from expert 3
of layer 5, so they never share a slot.

Belady is included because it is the upper bound on any online policy. If LRU
already sits close to Belady at your capacity, there is no headroom and effort
should go into prefetch (hiding the misses) rather than smarter eviction. If the
gap is wide, eviction policy is worth real work. Deciding which of those two
worlds you are in is the point of this file.

WATCH FOR THE LRU CLIFF
-----------------------
MoE decode is a *cyclic* access pattern: every token sweeps layer 0..N-1, and
each layer touches top_k experts, so one token cycles through roughly
`top_k * n_layers` distinct keys before returning to layer 0. That is LRU's
textbook worst case. Below a capacity of one token's working set, LRU evicts
every entry exactly before its next use and its hit rate collapses — in the
synthetic check it goes to *precisely zero* while LFU and static are still at
25-40% on the same trace.

So a single global LRU over all (layer, expert) pairs is the wrong default for
this workload. If the sweep shows that cliff on real traces, the fixes are
per-layer cache partitioning (each layer gets its own budget, so a layer's
entries are not evicted by later layers in the same token), or a
frequency-biased policy that does not treat the cyclic sweep as recency
information. Look at where LRU crosses LFU before designing anything.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict

import numpy as np
import pandas as pd

POLICIES = ("belady", "lru", "lfu", "static")


def build_access_sequence(frame: pd.DataFrame, num_experts: int) -> np.ndarray:
    """Flatten a routing trace into the decode-order (layer, expert) key stream."""
    ordered = frame.sort_values(["seq_id", "pos", "layer", "rank"], kind="stable")
    keys = ordered["layer"].to_numpy(np.int64) * num_experts + ordered["expert"].to_numpy(np.int64)
    return keys.astype(np.int32)


def _next_use(keys: np.ndarray) -> np.ndarray:
    """For each access, the index of the next access to the same key (n if none)."""
    n = keys.shape[0]
    order = np.lexsort((np.arange(n), keys))
    sorted_keys = keys[order]
    same = np.empty(n, dtype=bool)
    same[:-1] = sorted_keys[:-1] == sorted_keys[1:]
    same[-1] = False

    nxt = np.full(n, n, dtype=np.int64)
    nxt[:-1] = np.where(same[:-1], order[1:], n)

    next_use = np.empty(n, dtype=np.int64)
    next_use[order] = nxt
    return next_use


def _sim_belady(keys: np.ndarray, capacity: int) -> int:
    next_use = _next_use(keys)
    cache: set[int] = set()
    current: dict[int, int] = {}
    heap: list[tuple[int, int]] = []
    hits = 0

    for i, key in enumerate(keys):
        key = int(key)
        if key in cache:
            hits += 1
        else:
            if len(cache) >= capacity:
                while heap:
                    neg_when, victim = heapq.heappop(heap)
                    # Lazy deletion: skip stale entries superseded by a later push.
                    if victim in cache and -neg_when == current.get(victim):
                        cache.discard(victim)
                        current.pop(victim, None)
                        break
            cache.add(key)
        current[key] = int(next_use[i])
        heapq.heappush(heap, (-int(next_use[i]), key))
    return hits


def _sim_lru(keys: np.ndarray, capacity: int) -> int:
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for key in keys:
        key = int(key)
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[key] = None
    return hits


def _sim_lfu(keys: np.ndarray, capacity: int) -> int:
    counts: dict[int, int] = {}
    cache: set[int] = set()
    heap: list[tuple[int, int, int]] = []
    hits = 0
    tick = 0

    for key in keys:
        key = int(key)
        counts[key] = counts.get(key, 0) + 1
        if key in cache:
            hits += 1
        else:
            if len(cache) >= capacity:
                while heap:
                    count, _, victim = heapq.heappop(heap)
                    if victim in cache and count == counts[victim]:
                        cache.discard(victim)
                        break
            cache.add(key)
        tick += 1
        heapq.heappush(heap, (counts[key], tick, key))
    return hits


def _sim_static(keys: np.ndarray, capacity: int) -> int:
    """Pin the globally hottest keys, never evict.

    Uses whole-trace frequencies, so this is the *offline-profiled* version of
    "pin the hot experts in VRAM" — an achievable policy if you profile ahead of
    time, and a fair upper bound on the naive static approach.

    Note this can score slightly *above* Belady, which is not a contradiction:
    Belady is optimal among **demand-paging** policies, which must pay a
    compulsory miss the first time each key is touched. `static` is a
    prefetching policy — it loads at t=0 and skips those compulsory misses
    entirely. The gap is small (roughly cache_size / n_accesses) but it is the
    whole thesis of this project in miniature: prefetching beats the best
    possible reactive policy, because it attacks misses no eviction rule can.
    """
    unique, counts = np.unique(keys, return_counts=True)
    pinned = set(unique[np.argsort(-counts)[:capacity]].tolist())
    return int(np.isin(keys, list(pinned)).sum()) if pinned else 0


_SIMULATORS = {
    "belady": _sim_belady,
    "lru": _sim_lru,
    "lfu": _sim_lfu,
    "static": _sim_static,
}


def sweep(
    keys: np.ndarray,
    capacities: list[int],
    *,
    policies: tuple[str, ...] = POLICIES,
    bytes_per_expert: float | None = None,
    accesses_per_token: int | None = None,
) -> pd.DataFrame:
    """Hit rate for each (policy, capacity).

    bytes_per_expert and accesses_per_token turn the hit rate into the number
    that actually matters — bytes fetched per generated token, which divided by
    your storage bandwidth is seconds per token.
    """
    rows = []
    total = int(keys.shape[0])
    for policy in policies:
        simulate = _SIMULATORS[policy]
        for capacity in capacities:
            hits = simulate(keys, capacity)
            row = {
                "policy": policy,
                "capacity": capacity,
                "hits": hits,
                "accesses": total,
                "hit_rate": hits / total if total else 0.0,
            }
            if bytes_per_expert is not None:
                row["cache_bytes"] = capacity * bytes_per_expert
            if bytes_per_expert is not None and accesses_per_token:
                misses_per_token = (total - hits) / (total / accesses_per_token)
                row["fetch_bytes_per_token"] = misses_per_token * bytes_per_expert
            rows.append(row)
    return pd.DataFrame(rows)


def seconds_per_token(fetch_bytes_per_token: float, bandwidth_bytes_per_s: float) -> float:
    """Convert a simulated fetch volume into a wall-clock estimate.

    Bandwidth is the *effective* rate of whichever tier the misses come from.
    Useful reference points: 7200rpm SATA HDD ~1.2e8, SATA SSD ~5.0e8,
    NVMe Gen3 x4 ~3.0e9, NVMe Gen4 x4 ~6.5e9, DDR4-2133 dual channel ~3.0e10,
    PCIe 3.0 x16 host-to-device ~1.2e10.
    """
    return fetch_bytes_per_token / bandwidth_bytes_per_s
