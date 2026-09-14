"""GPU-resident expert cache with LRU eviction.

LRU is the measured choice, not the default one. Stage 0's Q5 swept Belady,
LRU, LFU and a static pinned set over a 48-document OLMoE trace at 25%
capacity: LRU 54.5%, LFU 42.1%, static 42.2%, with Belady's 77.9% as the
ceiling. An earlier run said the opposite because it was reading a
single-domain prefix of the trace; see the README. The gap to Belady is 23.4
points, so eviction policy has real headroom left — but the first job is to
have a correct, fast, measurable LRU to improve *on*.

WHY THE SLOTS ARE PREALLOCATED
------------------------------
Every fill writes into a slot that already exists. Allocating a fresh tensor
per fill would hand the work to the caching allocator, whose behaviour under a
steady stream of same-sized alloc/free is exactly the thing that makes offload
benchmarks irreproducible — you end up measuring allocator luck. A fixed pool
also makes the VRAM budget a number you choose rather than one you discover.

WHY THERE IS NO EXPLICIT SYNCHRONIZE
------------------------------------
Fills are issued on the current CUDA stream, and the expert GEMMs that consume
them run on that same stream afterwards. CUDA guarantees ordering within a
stream, so the copy is complete before the GEMM reads it without any
`synchronize()` call. That is worth stating explicitly because it is precisely
the guarantee Stage 1's prefetcher will *give up* when it moves fills to a side
stream, at which point events become mandatory.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from flashforge.runtime.store import PROJECTIONS, ExpertStore

ExpertKey = tuple[int, int]


@dataclass
class CacheStats:
    """Fill counters. `hits + misses` is expert *lookups*, not tokens."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_fetched: int = 0
    # Lookups broken out by layer, so a bad hit rate can be traced to where it
    # happens rather than averaged into a single uninformative number.
    per_layer: dict[int, list[int]] = field(default_factory=dict)

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def record(self, layer: int, hits: int, misses: int) -> None:
        self.hits += hits
        self.misses += misses
        counts = self.per_layer.setdefault(layer, [0, 0])
        counts[0] += hits
        counts[1] += misses

    def reset(self) -> None:
        self.hits = self.misses = self.evictions = self.bytes_fetched = 0
        self.per_layer.clear()

    def describe(self) -> str:
        return (
            f"hit rate {self.hit_rate:.1%} ({self.hits:,} / {self.lookups:,}) | "
            f"{self.evictions:,} evictions | "
            f"{self.bytes_fetched / 1e9:.2f} GB fetched"
        )


class ExpertCache:
    """A fixed pool of GPU expert slots, filled on demand, evicted by LRU."""

    def __init__(
        self,
        store: ExpertStore,
        capacity: int,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.store = store
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.dtype = dtype or store.shape.dtype
        self.stats = CacheStats()

        shape = store.shape
        self._slots = torch.empty(
            (self.capacity, shape.numel), dtype=self.dtype, device=self.device
        )

        # Per-projection views over the whole pool, built once. `unflatten`
        # rather than `view` because a column slice of the pool is not
        # contiguous, and down_proj's (hidden, intermediate) shape is the
        # transpose of the other two — see ExpertStore._flatten_expert.
        m = shape.matrix_numel
        matrix_shapes = {
            "gate_proj": (shape.intermediate_size, shape.hidden_size),
            "up_proj": (shape.intermediate_size, shape.hidden_size),
            "down_proj": (shape.hidden_size, shape.intermediate_size),
        }
        self._views = {
            name: self._slots[:, i * m : (i + 1) * m].unflatten(1, matrix_shapes[name])
            for i, name in enumerate(PROJECTIONS)
        }

        self._slot_of: OrderedDict[ExpertKey, int] = OrderedDict()
        self._free: list[int] = list(range(self.capacity))

    # -- geometry ----------------------------------------------------------

    @property
    def bytes_resident(self) -> int:
        return self.capacity * self.store.shape.nbytes

    def gate_proj(self, slot: int) -> torch.Tensor:
        return self._views["gate_proj"][slot]

    def up_proj(self, slot: int) -> torch.Tensor:
        return self._views["up_proj"][slot]

    def down_proj(self, slot: int) -> torch.Tensor:
        return self._views["down_proj"][slot]

    # -- the hot path ------------------------------------------------------

    def acquire(self, layer: int, experts: list[int]) -> dict[int, int]:
        """Ensure every requested expert is resident; return expert -> slot.

        `experts` must be de-duplicated by the caller — it comes straight from
        the router's unique-expert list, and re-deriving uniqueness here would
        double the work on the hottest path in the model.
        """
        if len(experts) > self.capacity:
            raise ValueError(
                f"Layer {layer} routes to {len(experts)} experts but the cache holds "
                f"{self.capacity}. A cache smaller than one layer's working set "
                "cannot complete a forward pass.\n"
                "Note the working set is the *union* over the tokens in the batch, "
                "not top_k: a single decode token needs top_k slots, but a prefill "
                "batch of more than a few tokens will touch nearly every expert in "
                "the layer. Size the cache for num_experts, or prefill in chunks."
            )

        hits: list[int] = []
        misses: list[int] = []
        for expert in experts:
            key = (layer, expert)
            if key in self._slot_of:
                # Touch before evicting anything, so this request's own hits are
                # at the MRU end and cannot be chosen as victims below.
                self._slot_of.move_to_end(key)
                hits.append(expert)
            else:
                misses.append(expert)

        self.stats.record(layer, len(hits), len(misses))
        if misses:
            self._evict_for(len(misses))
            self._fill(layer, misses)

        return {expert: self._slot_of[(layer, expert)] for expert in experts}

    def _evict_for(self, needed: int) -> None:
        shortfall = needed - len(self._free)
        for _ in range(max(0, shortfall)):
            # popitem(last=False) is the least-recently-used end. Every key
            # belonging to the in-flight request was moved to the other end in
            # acquire(), and acquire() refuses requests larger than capacity,
            # so a victim is never something we are about to read.
            _, slot = self._slot_of.popitem(last=False)
            self._free.append(slot)
            self.stats.evictions += 1

    def _fill(self, layer: int, experts: list[int]) -> None:
        row_bytes = self.store.shape.nbytes
        pinned = self.store.is_pinned(layer)
        for expert in experts:
            slot = self._free.pop()
            # non_blocking only actually overlaps when the source is pinned;
            # on pageable memory CUDA falls back to a synchronous copy, which
            # is correct either way. Ordering with the GEMM that follows is
            # provided by the stream, not by waiting here.
            self._slots[slot].copy_(self.store.row(layer, expert), non_blocking=pinned)
            self._slot_of[(layer, expert)] = slot
            self.stats.bytes_fetched += row_bytes

    # -- introspection -----------------------------------------------------

    def resident(self) -> set[ExpertKey]:
        return set(self._slot_of)

    def clear(self) -> None:
        """Drop every resident expert without freeing the pool."""
        self._slot_of.clear()
        self._free = list(range(self.capacity))

    def describe(self) -> str:
        total_slots = len(self.store.layers) * self.store.num_experts
        share = self.capacity / total_slots if total_slots else 0.0
        return (
            f"{self.capacity:,} slots ({share:.0%} of {total_slots:,} experts) | "
            f"{self.bytes_resident / (1 << 30):.2f} GB on {self.device}"
        )
