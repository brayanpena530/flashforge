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

So a single global LRU is the wrong default *at that capacity*. The fixes are
per-layer cache partitioning (each layer gets its own budget, so a layer's
entries are not evicted by later layers in the same token), or a
frequency-biased policy that does not treat the cyclic sweep as recency
information.

Note the scope. On the real OLMoE trace, provisioned at 256 slots — twice one
token's 128-slot sweep — LRU is comfortably the best online policy (54.5% vs
LFU's 42.1%). The cliff is real but lives below any capacity worth
provisioning. Look at where LRU crosses LFU before designing anything.

MEASURE IT ON THE WHOLE CORPUS
------------------------------
That cliff is real below one token's working set, but above it the ranking
turns on something else entirely: how many distinct *documents* the simulation
sees. Inside a handful of topically consistent documents, expert usage is
concentrated and frequency policies look excellent. Across a diverse corpus the
same statistics dilute and recency wins instead. On a 48-sequence OLMoE trace
at 25% capacity, a 300k-access prefix covered 5 documents and put LFU (72.4%)
comfortably above LRU (65.2%); the full 48 reverses it to LRU 54.5%, LFU 42.1%.

Which is why `subsample_sequences` exists: budget by dropping whole sequences,
never by truncating the flattened key stream.

STAGE 1E — SPENDING THE BELADY GAP
----------------------------------
Stage 1d left the fill path at 98% of the card's pinned PCIe rate, so no
reordering or acceleration of transfers can pay any more; only moving fewer
bytes can. At 256 slots the runtime misses 52.5% of 128 lookups per token, and
every avoided miss is 12.58 MB that never crosses the link. Belady's 23.4-point
margin over LRU is therefore a throughput budget, not a curiosity.

The four candidates here are chosen against a specific diagnosis. 256 slots is
exactly two tokens of sweep, so global LRU grants *every* expert the router
touched once a full two tokens of residency, whether or not it is ever touched
again. `lru2` and `slru` both attack that by requiring a second reference
before granting full residency; `layered` attacks the cyclic sweep that makes
late layers evict early ones; `hybrid` splits the pool between a frequency-
pinned half and a recency half, on the theory that `static` and `lru` fail on
different accesses rather than one simply being worse.

All four lose. Ranked on 147,456 decode lookups dumped from the runtime over 18
prompts spanning the corpus's six domains, at the 256 slots it ships at:

    policy        hit rate   vs lru   predicted decode t/s
    belady           77.7%   +21.4      10.87   (unreachable)
    slru p=0.5       58.2%    +1.9       7.94
    layered          56.8%    +0.4       7.79
    lru              56.3%       -       7.74
    lfu              42.3%   -14.1       6.51
    lru2             35.6%   -20.8       6.06

The best online policy anyone here can build is worth **+1.9 points**, which is
+2.6% predicted throughput — under the harness's own ±8% pass-to-pass spread,
so it is not merely small, it is unmeasurable. Belady's 21.4 points are real
and no online policy gets a tenth of them.

AND THE FIRST VERSION OF THAT TABLE SAID THE OPPOSITE
-----------------------------------------------------
Ranked on the *benchmark* prompt — which is the string "The history of
computing is" repeated 64 times — `lru2` scored +16.9 points and `static`
+33.5. Widening the trace to 18 real documents sent `lru2` to -20.8. Same
model, same capacity, same simulator; one prompt versus eighteen.

This is the reversal the section above warns about, reproduced exactly, and it
is worth being precise about the trap. The repeated phrase is the right choice
for a *timing* harness: it fixes the sequence length and makes the work
reproducible. It is the wrong trace to fit a policy to, because a policy fitted
to it is fitted to a workload of one topic. Dump with `--trace-prompts` for
policy work; the flag exists because trusting the narrow table would have
shipped a 22% regression.

WHAT THE GAP IS ACTUALLY WORTH, WHICH IS THE USEFUL PART
---------------------------------------------------------
Belady at 256 slots scores 77.7%. LRU reaches 77.1% at 464 slots. So perfect
prophecy is worth **1.8x the cache** — and it is cheaper to buy the capacity
than to predict the future, because capacity is purchasable and prophecy is
not. Halving the bytes per expert buys 2x the slots for the same VRAM *and*
halves what each remaining miss costs:

    config                              slots   hit    GB/token   pred t/s
    fp16, LRU            (today)          256  56.3%      0.703       7.74
    fp16, Belady         (unreachable)    256  77.7%      0.358      10.87
    int8, LRU            (same 3.0 GB)    512  81.4%      0.149      14.39

Quantised experts under plain LRU beat perfect eviction at fp16 by 32%. That is
the Stage 1e result: the eviction gap is a measurement of how much capacity is
missing, not an invitation to write a smarter policy.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict

import numpy as np
import pandas as pd

POLICIES = ("belady", "lru", "lfu", "static")

# Stage 1e candidates. Kept separate from POLICIES so the Stage 0 report keeps
# printing the four columns it has always printed, and so a policy that loses
# here does not silently become part of the Q5 answer.
CANDIDATES = ("lru2", "slru", "layered", "hybrid")


def build_access_sequence(frame: pd.DataFrame, num_experts: int) -> np.ndarray:
    """Flatten a routing trace into the decode-order (layer, expert) key stream."""
    ordered = frame.sort_values(["seq_id", "pos", "layer", "rank"], kind="stable")
    keys = ordered["layer"].to_numpy(np.int64) * num_experts + ordered["expert"].to_numpy(np.int64)
    return keys.astype(np.int32)


def subsample_sequences(
    frame: pd.DataFrame, max_accesses: int, *, seed: int = 0
) -> tuple[pd.DataFrame, int, int]:
    """Cut a trace to a budget by dropping whole sequences, not by truncating.

    Truncating the flattened key stream to its first N entries looks harmless
    and is not. The stream is ordered by (seq_id, pos, layer), so a prefix is
    the first few *documents* rather than a sample of them — and the policy
    ranking depends on how many documents are in view (see the module
    docstring for the measured reversal).

    Sampling whole sequences keeps each document's internal access pattern
    intact while preserving corpus diversity. Returns the trimmed frame, how
    many sequences were kept, and how many there were to begin with.
    """
    sequences = frame["seq_id"].drop_duplicates().to_numpy()
    total = int(len(sequences))
    if total == 0 or len(frame) <= max_accesses:
        return frame, total, total

    per_sequence = len(frame) / total
    keep = max(1, int(max_accesses // per_sequence))
    if keep >= total:
        return frame, total, total

    rng = np.random.default_rng(seed)
    chosen = rng.choice(sequences, size=keep, replace=False)
    return frame[frame["seq_id"].isin(chosen)], keep, total


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


def _sim_lru2(keys: np.ndarray, capacity: int) -> int:
    """LRU-K with K=2: evict by the *second* most recent reference.

    The policy LRU loses to here is not "keep things longer", it is "tell a
    one-hit wonder from a regular". At 256 slots the cache holds exactly two
    tokens of sweep, so LRU keeps every expert the router touched once for two
    tokens whether or not it will ever be touched again — and Belady's 23.4
    points of headroom is largely those. LRU-2 ranks a key by how long ago it
    was seen *twice*, so a single reference buys much less residency.

    Keys referenced only once have infinite backward-2-distance and are evicted
    first, tie-broken by recency, which is LRU restricted to the unproven set.
    """
    last: dict[int, int] = {}
    second: dict[int, int] = {}
    cache: set[int] = set()
    heap: list[tuple[int, int, int]] = []
    hits = 0

    def priority(key: int) -> tuple[int, int]:
        # Evicting the minimum: unproven keys (rank 0) go before proven ones.
        return (1, second[key]) if key in second else (0, last[key])

    for time, key in enumerate(keys):
        key = int(key)
        if key in cache:
            hits += 1
        else:
            if len(cache) >= capacity:
                while heap:
                    rank, when, victim = heapq.heappop(heap)
                    if victim in cache and (rank, when) == priority(victim):
                        cache.discard(victim)
                        # Forget the history too. A key that comes back after
                        # eviction is a fresh arrival: crediting it with
                        # references from before it was thrown out would let a
                        # long-dead key re-enter straight into the proven set.
                        last.pop(victim, None)
                        second.pop(victim, None)
                        break
            cache.add(key)

        if key in last:
            second[key] = last[key]
        last[key] = time
        rank, when = priority(key)
        heapq.heappush(heap, (rank, when, key))
    return hits


def _sim_slru(keys: np.ndarray, capacity: int, *, protected: float = 0.6) -> int:
    """Segmented LRU: a probationary segment in front of a protected one.

    Same intuition as LRU-2 and a cheaper implementation of it — two
    `OrderedDict`s and no heap, so it is the one that can actually go in the
    runtime's hot path. A miss lands in probation; a second reference promotes
    it to protected. Victims come from probation's LRU end, so the cost of a
    one-hit wonder is bounded by the probation segment rather than the cache.
    """
    # Not clamped away from the ends. protected=0 has to be exactly LRU and
    # protected=1 exactly LRU-with-an-extra-hop, because those degeneracies are
    # what the tests pin the implementation against; clamping to [1, capacity-1]
    # makes both of them almost-but-not-quite right, which is the hardest kind
    # of wrong to notice in a table of hit rates.
    protected_cap = max(0, min(capacity, int(round(capacity * protected))))
    prot: OrderedDict[int, None] = OrderedDict()
    prob: OrderedDict[int, None] = OrderedDict()
    hits = 0

    for key in keys:
        key = int(key)
        if key in prot:
            prot.move_to_end(key)
            hits += 1
            continue
        if key in prob:
            hits += 1
            del prob[key]
            prot[key] = None
            if len(prot) > protected_cap:
                # Demote, do not drop: a protected key that aged out has still
                # been referenced twice, so it outranks a fresh arrival.
                demoted, _ = prot.popitem(last=False)
                prob[demoted] = None
            continue
        if len(prot) + len(prob) >= capacity:
            (prob or prot).popitem(last=False)
        prob[key] = None
    return hits


def _sim_layered(keys: np.ndarray, capacity: int, *, stride: int) -> int:
    """One independent LRU per layer, with the capacity split evenly.

    MoE decode sweeps layer 0..N-1 every token, so under a single global LRU
    the later layers' misses evict the earlier layers' entries on every token.
    Partitioning makes a layer's residency depend only on that layer's routing.
    It also gives up the thing global LRU does well — lending slots to whichever
    layer routes most diffusely — so which wins is a measurement.
    """
    layers = sorted({int(k) // stride for k in np.unique(keys)})
    budgets = {layer: capacity // len(layers) for layer in layers}
    for layer in layers[: capacity % len(layers)]:
        budgets[layer] += 1

    caches: dict[int, OrderedDict[int, None]] = {layer: OrderedDict() for layer in layers}
    hits = 0
    for key in keys:
        key = int(key)
        cache = caches[key // stride]
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            if len(cache) >= budgets[key // stride]:
                cache.popitem(last=False)
            cache[key] = None
    return hits


def _sim_hybrid(keys: np.ndarray, capacity: int, *, pinned: float = 0.5) -> int:
    """Pin the globally hottest keys, run LRU over what is left.

    `static` alone scored 42.2% and LRU 54.5%, and the temptation is to read
    that as "frequency loses". It does not follow: the two policies fail on
    different accesses. Frequency captures the experts every document uses and
    misses the local burst; recency captures the burst and re-fetches the
    perennials after every sweep. Splitting the pool lets each cover its half.

    Like `static`, the pinned set is chosen from whole-trace frequencies, so
    this is an *offline-profiled* policy. That is achievable — profile once at
    build time — but it is not an online result, and it must not be compared
    against LRU as if it were.
    """
    pin_count = max(0, min(capacity, int(round(capacity * pinned))))
    unique, counts = np.unique(keys, return_counts=True)
    hot = set(unique[np.argsort(-counts)[:pin_count]].tolist())

    lru_cap = capacity - pin_count
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for key in keys:
        key = int(key)
        if key in hot:
            hits += 1
            continue
        if lru_cap <= 0:
            continue
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            if len(cache) >= lru_cap:
                cache.popitem(last=False)
            cache[key] = None
    return hits


_SIMULATORS = {
    "belady": _sim_belady,
    "lru": _sim_lru,
    "lfu": _sim_lfu,
    "static": _sim_static,
    "lru2": _sim_lru2,
    "slru": _sim_slru,
    "layered": _sim_layered,
    "hybrid": _sim_hybrid,
}

# Policies that need to decode a layer index out of the flat key.
_NEEDS_STRIDE = frozenset({"layered"})


def sweep(
    keys: np.ndarray,
    capacities: list[int],
    *,
    policies: tuple[str, ...] = POLICIES,
    bytes_per_expert: float | None = None,
    accesses_per_token: int | None = None,
    stride: int | None = None,
    params: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Hit rate for each (policy, capacity).

    bytes_per_expert and accesses_per_token turn the hit rate into the number
    that actually matters — bytes fetched per generated token, which divided by
    your storage bandwidth is seconds per token.

    `stride` is num_experts, needed to recover a layer index from a flat key;
    `params` carries per-policy knobs, e.g. {"slru": {"protected": 0.75}}. The
    label in the output is the policy name plus those knobs, so a parameter
    sweep does not collapse into one row.
    """
    rows = []
    total = int(keys.shape[0])
    for policy in policies:
        base, _, _ = policy.partition(":")
        simulate = _SIMULATORS[base]
        kwargs: dict[str, float | int] = dict((params or {}).get(policy, {}))
        if base in _NEEDS_STRIDE:
            if stride is None:
                raise ValueError(f"policy {policy!r} needs stride=num_experts")
            kwargs["stride"] = stride
        for capacity in capacities:
            hits = simulate(keys, capacity, **kwargs)
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
