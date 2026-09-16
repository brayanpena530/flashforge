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

from flashforge.runtime.store import PROJECTIONS, ExpertStore, dequantize_matrix

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
        # The pool mirrors a store row byte for byte, so it is int8 once the
        # store is quantised and `self.dtype` is only the dtype the *math*
        # happens in. Capacity rises for free: nothing here knows that a row
        # got smaller, it just allocates `row_numel` of `row_dtype`.
        self._shape = shape
        self._slots = torch.empty(
            (self.capacity, shape.row_numel), dtype=shape.row_dtype, device=self.device
        )

        # Where each projection lives inside a row. Elements when the row is
        # plain fp16, bytes once it is a packed mixed-dtype buffer — the two
        # only coincide when every projection is one byte wide.
        m = shape.matrix_numel
        self._layout: dict[str, tuple[int, int, int | None, int | None, tuple[int, int]]] = {}
        for i, name in enumerate(PROJECTIONS):
            mshape = shape.matrix_shape(name)
            if shape.quant is None:
                self._layout[name] = (i * m, (i + 1) * m, None, None, mshape)
                continue
            wlo, whi = shape.spans[name]
            if shape.is_quantized(name):
                slo, shi = shape.spans[f"{name}.scale"]
            else:
                slo = shi = None
            self._layout[name] = (wlo, whi, slo, shi, mshape)

        # Per-projection views over the whole pool, built once. `unflatten`
        # rather than `view` because a column slice of the pool is not
        # contiguous, and down_proj's (hidden, intermediate) shape is the
        # transpose of the other two — see ExpertStore._flatten_expert.
        self._views = {name: self._unpack(self._slots, name) for name in PROJECTIONS}

        # Stage 1e-2c. `gather` reads the pool with one `index_select`, and
        # PyTorch chooses that kernel by asking whether the tensor's **element**
        # count fits in an int32. A quantised pool is int8, so elements are
        # bytes, and at 8.39 MB per row it crosses INT32_MAX at 256 slots —
        # after which the kernel falls to 64-bit index math and loses its
        # vectorised loads. Measured on this card: 116 GB/s at 255 slots, 49
        # GB/s at 256, flat on both sides. A step, not a slope. That step is the
        # 26 ms/token the int8 capacity ladder could not explain.
        #
        # Selecting through a *wider* view of the same bytes fixes both halves
        # of it. The rows are contiguous and identically sized, so viewing the
        # pool as int64 divides the element count by eight — which puts the
        # boundary out of reach until 2,047 slots — and hands the kernel eight
        # bytes per element to move instead of one. It is faster below the old
        # boundary too: 100 -> 280 GB/s at the shipping 238 slots. Not a
        # workaround for the cliff so much as the gather this always wanted.
        #
        # `view(dtype)` needs the row to divide evenly into the wider type, and
        # a toy row in the tests may not, so the fallback is the pool itself.
        self._gather_pool = self._widen(self._slots)

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

        # Stage 1e. Set to a list to record every lookup as a flat
        # `layer * num_experts + expert` key, in acquire order. This is the
        # stream an eviction policy actually faces, which is not the same object
        # as Q5's offline trace: Q5 replayed a *corpus*, and the policy ranking
        # there was measured to depend on how many documents were in view. A
        # policy tuned on the runtime's own log is tuned on the workload the
        # runtime is graded on.
        self.access_log: list[int] | None = None

    @property
    def _timing(self) -> bool:
        """`time_fills`, but only where CUDA events exist to record.

        The CPU-side tests build an ExpertCache on the CPU, where a fill is a
        host memcpy and there is nothing asynchronous to measure. Silently doing
        nothing there is right; raising would make the flag untestable off-GPU.
        """
        return self.time_fills and self.device.type == "cuda"

    # -- geometry ----------------------------------------------------------

    @staticmethod
    def _widen(pool: torch.Tensor) -> torch.Tensor:
        """The widest reinterpretation of `pool` that `index_select` can use.

        Widest first: int64 divides the element count by eight against an int8
        pool, int32 by four. Both are pure reinterpretations — same bytes, same
        rows, same order — so the selected block views back to `row_dtype`
        exactly. Returns `pool` unchanged when no width divides the row, which
        is the only correctness question here and is why this is a fallback
        rather than an assertion: a toy row in the tests need not be a multiple
        of eight bytes, and the slow gather is still a *right* gather.
        """
        for wider in (torch.int64, torch.int32):
            if pool.element_size() >= wider.itemsize:
                continue
            per = wider.itemsize // pool.element_size()
            if pool.shape[-1] % per:
                continue
            try:
                return pool.view(wider)
            except RuntimeError:
                # Unaligned storage or a non-contiguous last axis. Neither can
                # happen for a freshly allocated pool, and neither is worth
                # crashing over if it ever does.
                continue
        return pool

    @property
    def bytes_resident(self) -> int:
        return self.capacity * self.store.shape.nbytes

    def _unpack(
        self, rows: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Slice one projection out of a `(N, row_numel)` block: (weight, scale).

        `scale` is None when the projection is stored at full width, in which
        case `weight` is already in `self.dtype` and the caller is done. This is
        the *only* place that knows the row layout; both the per-slot accessors
        and `gather` go through it, so an int8 store cannot be half-supported.
        """
        wlo, whi, slo, shi, mshape = self._layout[name]
        if slo is None:
            weight = rows[:, wlo:whi]
            if self._shape.quant is not None:
                # A byte buffer holding an exempt projection: reinterpret, do
                # not convert. `view` is legal here because a column slice keeps
                # stride 1 on the last axis.
                weight = weight.view(self.dtype)
            return weight.unflatten(1, mshape), None
        return (
            rows[:, wlo:whi].unflatten(1, mshape),
            rows[:, slo:shi].view(self.dtype).unflatten(1, (mshape[0], -1)),
        )

    def _projection(self, name: str, slot: int) -> torch.Tensor:
        weight, scale = self._views[name]
        if scale is None:
            return weight[slot]
        # A view when unquantised, a fresh tensor when not. Every caller is
        # read-only, so that difference is invisible — but it is a difference,
        # and the loop path's bit-exactness against the stock block is measured
        # with it in place rather than assumed through it.
        return dequantize_matrix(
            weight[slot].unsqueeze(0), scale[slot].unsqueeze(0), self.dtype
        )[0]

    def gate_proj(self, slot: int) -> torch.Tensor:
        return self._projection("gate_proj", slot)

    def up_proj(self, slot: int) -> torch.Tensor:
        return self._projection("up_proj", slot)

    def down_proj(self, slot: int) -> torch.Tensor:
        return self._projection("down_proj", slot)

    def gather(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack several slots into batched weights: (E,I,H), (E,I,H), (E,H,I).

        One `index_select` over the flat pool, not three over the per-projection
        views, because the three projections of a slot are adjacent in the row —
        the same argument that made a fill one `copy_` instead of three.

        It runs over `_gather_pool`, a wider reinterpretation of the same bytes.
        See `_widen` and the note beside it in `__init__`: the element count, not
        the byte count, is what picks the kernel, and an int8 pool has one
        element per byte. Selecting through int64 moves the same rows at up to
        5.4x the bandwidth and takes a 26 ms/token cliff out of the capacity
        curve. The result is viewed straight back, so nothing below this line
        can tell the difference.

        This *copies* the weights, which is the whole cost of the grouped path:
        E x 12.58 MB read and written on-device per layer. It buys the removal
        of 3E kernel launches, and Stage 1's profile said launches were 95% of
        decode. Which of those two wins is a measurement, not an argument; see
        `ff-serve --grouped/--no-grouped`.

        It is also where int8 becomes fp16. `_forward_grouped` needs no
        knowledge of quantisation for exactly that reason: it was already paying
        for a copy here, and dequantisation rides along inside it. The output is
        the same fp16 (E,I,H) it always was, so the bmms and their transient
        VRAM are unchanged — what changed is that the *input* to this copy is
        half the size, which is the whole point.
        """
        rows = self._gather_pool.index_select(0, slots)
        if rows.dtype is not self._shape.row_dtype:
            rows = rows.view(self._shape.row_dtype)
        out = []
        for name in PROJECTIONS:
            weight, scale = self._unpack(rows, name)
            out.append(weight if scale is None else dequantize_matrix(weight, scale, self.dtype))
        return tuple(out)

    # -- the hot path ------------------------------------------------------

    def acquire(self, layer: int, experts: list[int]) -> dict[int, int]:
        """Ensure every requested expert is resident; return expert -> slot.

        `experts` must be de-duplicated by the caller — it comes straight from
        the router's unique-expert list, and re-deriving uniqueness here would
        double the work on the hottest path in the model.
        """
        if self.store.shape is not self._shape:
            # Quantising the store rebuilds every row at a new width and dtype.
            # A pool allocated against the old geometry would still accept the
            # `copy_` — it is the same number of *elements* only by accident —
            # and the model would read half an expert and half a neighbour.
            # That is a silent wrong-output failure, so it is checked rather
            # than documented.
            raise RuntimeError(
                f"The store's row geometry changed after this cache was built "
                f"({self._shape.row_numel} x {self._shape.row_dtype} -> "
                f"{self.store.shape.row_numel} x {self.store.shape.row_dtype}). "
                "Quantise the store before constructing the cache."
            )

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

        if self.access_log is not None:
            stride = self.store.num_experts
            self.access_log.extend(layer * stride + expert for expert in experts)

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

    def release(self) -> None:
        """Free the slot pool. The cache is unusable afterwards.

        A sweep loads one model per capacity and relies on the previous
        capacity's pool being collected before the next one is allocated. That
        worked until it didn't: three GB of VRAM survived `del model, report`
        into the next iteration and the second arm of the first int8 A/B died
        at `model.to(device)` with 0 bytes free. Whatever the surviving
        reference is, dropping the pool by hand does not depend on finding it —
        `_views` and `_layout` are views over `_slots`, so clearing them first
        is what makes the storage actually reachable for free. `_gather_pool` is
        another one, and is the reason this is a list rather than a line: adding
        a view of the pool anywhere means adding it here, or a sweep silently
        keeps every capacity it has already measured.
        """
        self.clear()
        self._views.clear()
        self._slots = torch.empty(0, dtype=self._shape.row_dtype, device=self.device)
        self._gather_pool = self._slots

    def describe(self) -> str:
        total_slots = len(self.store.layers) * self.store.num_experts
        share = self.capacity / total_slots if total_slots else 0.0
        return (
            f"{self.capacity:,} slots ({share:.0%} of {total_slots:,} experts) | "
            f"{self.bytes_resident / (1 << 30):.2f} GB on {self.device}"
        )
