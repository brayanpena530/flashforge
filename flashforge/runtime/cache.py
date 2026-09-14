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
Demand fills are issued on the current CUDA stream, and the expert GEMMs that
consume them run on that same stream afterwards. CUDA guarantees ordering
within a stream, so the copy is complete before the GEMM reads it without any
`synchronize()` call.

Stage 1c's `prefetch` gives that guarantee up, exactly as this docstring
predicted it would: it issues fills on a side stream so they overlap the
*previous* layer's GEMMs. Every read and write across the two streams is
therefore ordered by an explicit event. There are two hazards, not one, and
they need different mechanisms — see `prefetch`.
"""

from __future__ import annotations

import contextlib
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
    # Stage 1c. `prefetch_issued` is experts speculatively fetched;
    # `prefetch_used` is how many of those a later `acquire` actually wanted.
    # Their ratio is the predictor's precision measured *in the runtime*, which
    # is the only place it counts — Q3's 0.835 was recall, offline, on a trace.
    prefetch_issued: int = 0
    prefetch_used: int = 0
    prefetch_wasted: int = 0
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

    @property
    def prefetch_precision(self) -> float:
        return self.prefetch_used / self.prefetch_issued if self.prefetch_issued else 0.0

    def reset(self) -> None:
        self.hits = self.misses = self.evictions = self.bytes_fetched = 0
        self.prefetch_issued = self.prefetch_used = self.prefetch_wasted = 0
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

        # Stage 1c instrumentation, off by default because recording two CUDA
        # events per layer is itself a perturbation. See `drain_fill_ms`.
        self.time_fills = False
        self._fill_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        # Side-stream work is timed separately and must never be added to the
        # demand total. Demand fill blocks the GEMMs; speculative fill is
        # supposed not to. Summing them into one "fill ms" column would make a
        # prefetch run look worse the better the overlap got.
        self._spec_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        # Stage 1c prefetch. The stream is created eagerly on CUDA so that
        # enabling prefetch mid-run cannot allocate one inside a timed region.
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        # Experts whose fill is in flight on the side stream, and the event that
        # says it has landed.
        self._pending: dict[ExpertKey, torch.cuda.Event] = {}
        # Slots the layer currently executing holds. The side stream must not
        # choose these as eviction victims; see `prefetch` for why the WAR event
        # alone does not cover them.
        self._protected: set[int] = set()

    @property
    def _timing(self) -> bool:
        """`time_fills`, but only where CUDA events exist to record.

        The CPU-side tests build an ExpertCache on the CPU, where a fill is a
        host memcpy and there is nothing asynchronous to measure. Silently doing
        nothing there is right; raising would make the flag untestable off-GPU.
        """
        return self.time_fills and self.device.type == "cuda"

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

    def gather(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack several slots into batched weights: (E,I,H), (E,I,H), (E,H,I).

        One `index_select` over the flat pool, not three over the per-projection
        views, because the three projections of a slot are adjacent in the row —
        the same argument that made a fill one `copy_` instead of three.

        This *copies* the weights, which is the whole cost of the grouped path:
        E x 12.58 MB read and written on-device per layer. It buys the removal
        of 3E kernel launches, and Stage 1's profile said launches were 95% of
        decode. Which of those two wins is a measurement, not an argument; see
        `ff-serve --grouped/--no-grouped`.
        """
        rows = self._slots.index_select(0, slots)
        shape = self.store.shape
        m = shape.matrix_numel
        # unflatten, not view: a column slice of `rows` is not contiguous. Same
        # reason as the pool views above.
        return (
            rows[:, 0 * m : 1 * m].unflatten(1, (shape.intermediate_size, shape.hidden_size)),
            rows[:, 1 * m : 2 * m].unflatten(1, (shape.intermediate_size, shape.hidden_size)),
            rows[:, 2 * m : 3 * m].unflatten(1, (shape.hidden_size, shape.intermediate_size)),
        )

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
                # A hit that a prefetch put there is still a hit, but the copy
                # may not have landed. Ordering it against this stream is the
                # whole reason the side stream needs events: cache.py's module
                # docstring promised this exact guarantee would be given up.
                self._await(key, used=True)
            else:
                misses.append(expert)

        self.stats.record(layer, len(hits), len(misses))
        if misses:
            self._evict_for(len(misses))
            self._fill(layer, misses)

        slots = {expert: self._slot_of[(layer, expert)] for expert in experts}
        # These are the slots the GEMMs about to run will read. A prefetch
        # issued during those GEMMs must not evict them.
        self._protected = set(slots.values())
        return slots

    def _await(self, key: ExpertKey, *, used: bool) -> None:
        """Make the compute stream wait for an in-flight prefetch of `key`."""
        event = self._pending.pop(key, None)
        if event is None:
            return
        torch.cuda.current_stream(self.device).wait_event(event)
        if used:
            self.stats.prefetch_used += 1

    def _evict_for(self, needed: int) -> None:
        shortfall = needed - len(self._free)
        for _ in range(max(0, shortfall)):
            # popitem(last=False) is the least-recently-used end. Every key
            # belonging to the in-flight request was moved to the other end in
            # acquire(), and acquire() refuses requests larger than capacity,
            # so a victim is never something we are about to read.
            key, slot = self._slot_of.popitem(last=False)
            if key in self._pending:
                # A speculative fetch nobody wanted. The slot is about to be
                # rewritten on the compute stream, so that write has to be
                # ordered after the side stream's write — otherwise the two race
                # and the loser's bytes are what the model reads.
                self._await(key, used=False)
                self.stats.prefetch_wasted += 1
            self._free.append(slot)
            self.stats.evictions += 1

    def prefetch(self, layer: int, experts: list[int]) -> int:
        """Speculatively fill `experts` on the side stream. Returns how many.

        Called from layer L for layer L+1, so the copies overlap L's GEMMs
        instead of stalling L+1's. Nothing here is allowed to block: a
        misprediction must cost bandwidth, never latency, or a predictor at
        Q3's 0.835 recall would be a net loss.

        TWO WRITE HAZARDS, TWO MECHANISMS
        ---------------------------------
        The side stream writes into slots the compute stream reads, so both
        directions have to be ordered explicitly.

        *Later layers' slots* are covered by the event recorded below: it is
        recorded on the compute stream at issue time, so waiting on it means
        every kernel enqueued before now — including every previous layer's
        GEMMs — has finished before a single byte is overwritten.

        *This* layer's slots are not, because its GEMMs have not been enqueued
        yet when the prefetch is issued. They are covered by `_protected`, which
        `acquire` just set, and which eviction below refuses to touch.

        Missing either one produces a model that is correct on most tokens and
        quietly wrong on the ones where the race is lost — the failure mode with
        no error message, which is why it is spelled out rather than commented.
        """
        wanted = [e for e in experts if (layer, e) not in self._slot_of]
        if not wanted:
            return 0

        # Only the unprotected LRU tail may be displaced, and only by as many
        # experts as there is room for. A prefetch that had to evict the layer
        # it is running underneath would be trading a certain cost for a
        # speculative one.
        evictable = [
            key for key, slot in self._slot_of.items() if slot not in self._protected
        ]
        room = len(self._free) + len(evictable)
        wanted = wanted[:room]
        if not wanted:
            return 0

        # On CPU there is no side stream and a fill is a host memcpy, so the
        # copies simply happen here. The bookkeeping below — eviction, slot
        # accounting, the protected set — is identical either way, which is what
        # lets the CPU tests cover it. Only the ordering is GPU-only.
        if self._stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            context = torch.cuda.stream(self._stream)
        else:
            ready = None
            context = contextlib.nullcontext()

        row_bytes = self.store.shape.nbytes
        pinned = self.store.is_pinned(layer)
        with context:
            if self._stream is not None:
                self._stream.wait_event(ready)
            if self._timing:
                spec_start = torch.cuda.Event(enable_timing=True)
                spec_start.record(self._stream)
            for expert in wanted:
                if not self._free:
                    key = evictable.pop(0)
                    if key in self._pending:
                        # Displacing one speculation with another. Order the
                        # writes; the side stream is one stream, so recording
                        # and waiting here is enough.
                        self._stream.wait_event(self._pending.pop(key))
                        self.stats.prefetch_wasted += 1
                    self._free.append(self._slot_of.pop(key))
                    self.stats.evictions += 1

                slot = self._free.pop()
                self._slots[slot].copy_(self.store.row(layer, expert), non_blocking=pinned)
                if self._stream is not None:
                    landed = torch.cuda.Event()
                    landed.record(self._stream)
                    self._pending[(layer, expert)] = landed

                self._slot_of[(layer, expert)] = slot
                self.stats.bytes_fetched += row_bytes
                self.stats.prefetch_issued += 1
            if self._timing:
                spec_end = torch.cuda.Event(enable_timing=True)
                spec_end.record(self._stream)
                self._spec_events.append((spec_start, spec_end))

        return len(wanted)

    def drain_fill_ms(self) -> tuple[float, float]:
        """(demand ms, speculative ms) of GPU fill time since the last drain.

        The first number bounds prefetching. Demand fills are issued on the
        compute stream, so a fill's duration is time the GEMMs that follow it
        are *not* running — it is on the critical path by construction, and a
        perfect prefetcher would recover all of it and nothing more.

        The second is side-stream work, which is *meant* to be hidden. It is
        reported beside the first rather than added to it, because the two
        answer different questions: how much is still blocking, and how much
        bandwidth the speculation spent to get it there.

        This synchronizes, so call it outside a timed region. Events are
        recorded per call (per layer), not per expert, so a layer's misses are
        timed as the one back-to-back burst they are issued as.
        """
        if not (self._fill_events or self._spec_events):
            return (0.0, 0.0)
        torch.cuda.synchronize()
        totals = tuple(
            sum(start.elapsed_time(end) for start, end in events)
            for events in (self._fill_events, self._spec_events)
        )
        self._fill_events.clear()
        self._spec_events.clear()
        return totals

    def _fill(self, layer: int, experts: list[int]) -> None:
        row_bytes = self.store.shape.nbytes
        pinned = self.store.is_pinned(layer)
        if self._timing:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        for expert in experts:
            slot = self._free.pop()
            # non_blocking only actually overlaps when the source is pinned;
            # on pageable memory CUDA falls back to a synchronous copy, which
            # is correct either way. Ordering with the GEMM that follows is
            # provided by the stream, not by waiting here.
            self._slots[slot].copy_(self.store.row(layer, expert), non_blocking=pinned)
            self._slot_of[(layer, expert)] = slot
            self.stats.bytes_fetched += row_bytes
        if self._timing:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._fill_events.append((start_event, end_event))

    # -- introspection -----------------------------------------------------

    def resident(self) -> set[ExpertKey]:
        return set(self._slot_of)

    def clear(self) -> None:
        """Drop every resident expert without freeing the pool."""
        if self._pending:
            # Slots are about to be declared free, so any in-flight write to
            # them has to be finished, not merely ordered.
            torch.cuda.synchronize(self.device)
            self._pending.clear()
        self._slot_of.clear()
        self._protected.clear()
        self._free = list(range(self.capacity))

    def describe(self) -> str:
        total_slots = len(self.store.layers) * self.store.num_experts
        share = self.capacity / total_slots if total_slots else 0.0
        return (
            f"{self.capacity:,} slots ({share:.0%} of {total_slots:,} experts) | "
            f"{self.bytes_resident / (1 << 30):.2f} GB on {self.device}"
        )
