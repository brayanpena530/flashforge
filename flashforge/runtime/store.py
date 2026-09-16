"""CPU-side expert weight store.

Every expert in the model lives here, in host RAM, and the GPU cache pulls from
it. Two decisions shape the layout.

**One expert is one contiguous block.** A SwiGLU expert is three matrices
(gate_proj and up_proj are `intermediate x hidden`, down_proj is
`hidden x intermediate`), and the obvious layout keeps them as three separate
tensors. That would make every cache fill three transfers instead of one.
Q8 measured the cost of not batching — at 64 KiB reads the drive needs 16
concurrent requests to reach peak and issuing them singly costs 6.5x — and the
same argument applies one tier up. So each expert is flattened into a single
`3 * intermediate * hidden` row and a fill is exactly one `copy_`.

**Pinning is budgeted, not assumed.** Pinned (page-locked) memory is what lets
a host-to-device copy run as a true async DMA, and Q7 measured pinned at
1.21 ms per expert against pageable's slower path. But pinned pages cannot be
swapped, so pinning is a hard claim on physical RAM. OLMoE's full expert set is
16 layers x 64 experts x 12.58 MB = **12.9 GB**, against 15 GB free on the
machine this was built for. Pinning all of it would work right up until it
didn't. `pin_gb` therefore caps it, layers are pinned until the budget runs
out, and `describe()` reports what actually got pinned rather than what was
asked for.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, replace

import torch
from torch import nn

log = logging.getLogger(__name__)

# The three projections of a SwiGLU expert, in the order they are packed into a
# row. Order is arbitrary but must match `ExpertCache.views`, so it lives here
# as the single definition rather than being spelled out in both places.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class QuantSpec:
    """Which projections are int8, and how much weight shares a scale.

    `group_size=0` is one scale per output channel. Anything else splits each
    row into contiguous spans of that many *inputs*, so a scale covers a
    stretch of the dot product rather than a slice across unrelated channels.
    A row that does not divide evenly falls back to per-channel.

    `projections` defaults to two of three. `down_proj` consumes the product of
    two activations, so it sees the widest dynamic range and is the one worth
    exempting first — quantising all three scored 98.32% teacher-forced top-1
    agreement against a pre-committed 99% bar, exempting it scored 99.13%.

    That 99.13% did not replicate, and neither did the instrument that read it.
    Scored on the deterministic loop path against 24,576 positions of the real
    512-token corpus — where the null control reads exactly 100.000%, so the
    harness contributes no noise at all — it comes back at **98.767% +/-
    0.070%**, missing the bar by 0.233 points at 3.3 sigma.

    Raising the scales to fp32 cannot fix it: measured on real expert matrices,
    fp32 scales remove 0.00% of the 0.8886% relative RMS weight error, because
    fp16's 11-bit mantissa is already three bits finer than the 256-level grid a
    scale exists to place. The shortfall is int8 rounding itself.

    The default is kept because it is still the best of the variants measured
    and the throughput win is real (+17%), but it is a *default*, not a
    clearance. See the runtime package docstring before shipping it.
    """

    projections: tuple[str, ...] = ("gate_proj", "up_proj")
    group_size: int = 0

    def __post_init__(self) -> None:
        unknown = set(self.projections) - set(PROJECTIONS)
        if unknown:
            raise ValueError(f"Not projections of a SwiGLU expert: {sorted(unknown)}")
        if not self.projections:
            raise ValueError("A QuantSpec that quantises nothing is not a quantised store.")


@dataclass(frozen=True)
class ExpertShape:
    """Geometry of one expert, and how it is packed into a flat row.

    WHY THE SCALES RIDE ALONG IN THE ROW
    ------------------------------------
    Stage 1e-2's design note said the scales would live in a table resident on
    the GPU for every expert in the model — 8.4 MB, 0.26% of the pool — so that
    they never crossed PCIe. Writing it settled the question the other way. A
    resident table has to be indexed by `(layer, expert)`, but `gather` is
    handed *slots*, so the cache would have to maintain a slot -> key mapping on
    the device; and the alternative, a second small pool written at fill time,
    costs a second `copy_` per miss. Appending the scales to the row instead
    costs 0.1% more bytes on the link (1.5% at group_size=128) and keeps a fill
    at exactly one `copy_`, one pool, and no reverse map. The transfer was never
    the expensive part of a scale; the bookkeeping was.
    """

    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype
    quant: QuantSpec | None = None

    @property
    def matrix_numel(self) -> int:
        """Elements in one projection. All three are the same size."""
        return self.hidden_size * self.intermediate_size

    @property
    def numel(self) -> int:
        """Weight elements in one expert. Unchanged by quantisation."""
        return len(PROJECTIONS) * self.matrix_numel

    # -- per-projection geometry -------------------------------------------

    def matrix_shape(self, name: str) -> tuple[int, int]:
        """(out_features, in_features). down_proj is the transpose of the others."""
        if name == "down_proj":
            return (self.hidden_size, self.intermediate_size)
        return (self.intermediate_size, self.hidden_size)

    def is_quantized(self, name: str) -> bool:
        return self.quant is not None and name in self.quant.projections

    def groups(self, name: str) -> int:
        """Scales per output channel. 0 when the projection is not quantised."""
        if not self.is_quantized(name):
            return 0
        _, in_features = self.matrix_shape(name)
        size = self.quant.group_size
        return in_features // size if size and in_features % size == 0 else 1

    def scale_numel(self, name: str) -> int:
        out_features, _ = self.matrix_shape(name)
        return out_features * self.groups(name)

    @property
    def item_size(self) -> int:
        return torch.empty(0, dtype=self.dtype).element_size()

    @property
    def spans(self) -> dict[str, tuple[int, int]]:
        """Byte ranges within a row: three weights, then the scales.

        Keyed by projection name for the weights and `<name>.scale` for the
        scales. Weights come first and in `PROJECTIONS` order so that an
        unquantised store's layout is byte-for-byte what it was before this
        existed — the scales region is simply empty.
        """
        spans: dict[str, tuple[int, int]] = {}
        offset = 0
        item = self.item_size
        for name in PROJECTIONS:
            width = self.matrix_numel * (1 if self.is_quantized(name) else item)
            spans[name] = (offset, offset + width)
            # Every span starts on an `item`-byte boundary. Reinterpreting a byte
            # slice as fp16 needs its offset divisible by 2, and real expert
            # matrices are large enough that this never pads — but the toy shapes
            # the tests use are not, and an alignment bug there would surface as
            # a shape error in a place that has nothing to do with alignment.
            offset += -(-width // item) * item
        for name in PROJECTIONS:
            width = self.scale_numel(name) * self.item_size
            spans[f"{name}.scale"] = (offset, offset + width)
            offset += width
        return spans

    @property
    def nbytes(self) -> int:
        return self.spans[f"{PROJECTIONS[-1]}.scale"][1]

    # -- how a row is actually stored ---------------------------------------

    @property
    def row_dtype(self) -> torch.dtype:
        """int8 once quantised, because a row is then a mixed-dtype byte buffer."""
        return torch.int8 if self.quant is not None else self.dtype

    @property
    def row_numel(self) -> int:
        return self.nbytes if self.quant is not None else self.numel


def quantize_matrix(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8. `w` is (E, out, in); returns int8 (E, out, in) and scales
    (E, out, groups).

    The ungrouped case is handled by adding a length-1 group axis rather than
    branching, so both paths produce a scale tensor of the same rank. That is
    what lets `dequantize_matrix` be one expression, and it is why `groups()`
    returns 1 rather than 0 for a per-channel quantised projection.
    """
    out_features, in_features = w.shape[1], w.shape[2]
    grouped = bool(group_size) and in_features % group_size == 0
    view = (
        w.reshape(-1, out_features, in_features // group_size, group_size)
        if grouped
        else w.unsqueeze(2)
    )
    scale = view.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = (view / scale).round_().clamp_(-127, 127)
    return q.reshape(w.shape).to(torch.int8), scale.squeeze(-1)


def dequantize_matrix(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of `quantize_matrix`, in `dtype`. (E, out, in)."""
    experts, out_features, in_features = q.shape
    groups = scale.shape[-1]
    wide = q.reshape(experts, out_features, groups, in_features // groups).to(dtype)
    return wide.mul_(scale.unsqueeze(-1).to(dtype)).reshape(q.shape)


class ExpertStore:
    """All expert weights, in host RAM, one contiguous row per expert.

    Indexed by `(layer, expert)` where `layer` is the model's own layer number,
    not a position in the MoE-layer list — the two diverge on models that keep
    some layers dense, and using the model's numbering keeps the store's keys
    the same as the trace's.
    """

    def __init__(self, shape: ExpertShape, num_experts: int):
        self.shape = shape
        self.num_experts = num_experts
        # layer index -> (num_experts, shape.row_numel) host tensor
        self._layers: dict[int, torch.Tensor] = {}
        self._pinned: set[int] = set()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        moe_layers: list[int],
        *,
        pin_gb: float = 0.0,
        detach_from_model: bool = True,
    ) -> "ExpertStore":
        """Pull every expert out of a loaded model and into a flat store.

        `detach_from_model` replaces each layer's `experts` ModuleList with an
        empty one as soon as that layer has been copied. That is not tidiness:
        the store is a second copy of the largest thing in the model, and on a
        machine where the checkpoint is 13.8 GB against 15 GB free, holding
        both at once is the difference between running and swapping. Peak
        overhead with detaching is one layer (~805 MB for OLMoE), not the whole
        expert set.
        """
        blocks = _find_expert_blocks(model, moe_layers)
        if not blocks:
            raise ValueError(
                "No expert ModuleLists found at layers.<i>.mlp.experts. This model "
                "does not use the block layout CachedMoEBlock replaces."
            )

        first_expert = blocks[moe_layers[0]][0]
        shape = _shape_of(first_expert)
        num_experts = len(blocks[moe_layers[0]])
        store = cls(shape, num_experts)

        pin_budget_bytes = int(pin_gb * (1 << 30))
        pinned_bytes = 0
        layer_bytes = num_experts * shape.nbytes

        for layer_idx in moe_layers:
            experts = blocks[layer_idx]
            if len(experts) != num_experts:
                raise ValueError(
                    f"Layer {layer_idx} has {len(experts)} experts but layer "
                    f"{moe_layers[0]} has {num_experts}; the store assumes a "
                    "uniform expert count per layer."
                )

            want_pinned = pinned_bytes + layer_bytes <= pin_budget_bytes
            try:
                rows = torch.empty(
                    (num_experts, shape.row_numel),
                    dtype=shape.row_dtype,
                    pin_memory=want_pinned,
                )
            except RuntimeError as exc:
                # Pinned host memory has its own ceiling, well below free RAM,
                # and it is reported as "CUDA error: out of memory" from a *host*
                # allocation — which reads like a VRAM problem and is not one.
                # Measured here: 11.8 GB pinnable on an idle 32 GB machine
                # against a 12.0 GB store, so asking for all of it fails on the
                # last layer after eleven minutes of work.
                #
                # A partly pinned store is a perfectly good store; the pinned
                # layers still take the async path and `is_pinned` is already
                # per-layer. So degrade rather than discard the load.
                if not want_pinned:
                    raise
                log.warning(
                    "Could not pin layer %d (%.2f GB pinned so far): %s\n"
                    "Falling back to pageable for this and every later layer. "
                    "Lower --pin-gb to keep the budget under the ceiling.",
                    layer_idx, pinned_bytes / (1 << 30), str(exc).splitlines()[0],
                )
                pin_budget_bytes = pinned_bytes  # stop trying
                want_pinned = False
                rows = torch.empty((num_experts, shape.row_numel), dtype=shape.row_dtype)
            for expert_idx, expert in enumerate(experts):
                rows[expert_idx] = _flatten_expert(expert, shape)

            store._layers[layer_idx] = rows
            if want_pinned:
                store._pinned.add(layer_idx)
                pinned_bytes += layer_bytes

            if detach_from_model:
                _detach_experts(model, layer_idx)
                gc.collect()

        if pin_gb > 0 and not store._pinned:
            log.warning(
                "pin_gb=%.1f is smaller than one layer of experts (%.2f GB); "
                "nothing was pinned and transfers will take the pageable path.",
                pin_gb, layer_bytes / (1 << 30),
            )
        return store

    def repin(self, pin_gb: float) -> int:
        """Re-allocate layers so pinned coverage matches a new budget.

        Whether memory is page-locked is fixed when it is allocated, so changing
        it means allocating again and copying. That is worth the cost for one
        reason: pin coverage is the largest remaining lever on a system whose
        link is saturated, and the only honest way to measure a lever is to
        A/B it on one loaded model. Comparing a pinned invocation against a
        pageable one would compare two machines — see troubleshoot.md 1.8, where
        free RAM fell 12.4 -> 5.8 GB across a single sweep and the same capacity
        measured 2.77 against 5.53 tok/s.

        Peak overhead is two layers (~1.6 GB), not the whole store: each layer
        is replaced before the next is touched.

        Returns the number of layers pinned afterwards.
        """
        budget = int(pin_gb * (1 << 30))
        layer_bytes = self.num_experts * self.shape.nbytes
        claimed = 0

        for layer in self.layers:
            want = claimed + layer_bytes <= budget
            if want:
                claimed += layer_bytes
            if want == self.is_pinned(layer):
                continue

            old = self._layers[layer]
            try:
                fresh = torch.empty_like(old, pin_memory=want)
            except RuntimeError as exc:
                # The pinned ceiling is well below free RAM and is reported as a
                # CUDA OOM from a host allocation. Stop climbing, keep what is
                # already pinned, and say so — a partly pinned store is fine.
                log.warning(
                    "Pinned ceiling reached at layer %d (%.2f GB pinned): %s",
                    layer, (claimed - layer_bytes) / (1 << 30), str(exc).splitlines()[0],
                )
                claimed -= layer_bytes
                budget = 0  # every later layer falls to pageable
                continue

            fresh.copy_(old)
            self._layers[layer] = fresh
            del old
            if want:
                self._pinned.add(layer)
            else:
                self._pinned.discard(layer)

        gc.collect()
        return len(self._pinned)

    def quantize_int8(self, spec: QuantSpec | None = None) -> dict[str, float]:
        """Convert every expert to int8 in place, for real this time.

        Rows are rebuilt as mixed-dtype byte buffers: the quantised projections
        as int8, any exempt projection still in `shape.dtype`, and the scales
        appended on the end. `shape.row_dtype` becomes int8 and `shape.nbytes`
        falls, which is the entire point — everything downstream sizes itself
        off those two numbers, so the cache gets more slots and a miss moves
        fewer bytes without any of it being told that quantisation happened.

        **Call this before building the cache.** The pool is allocated from
        `shape.row_numel`, so a cache built against the fp16 shape and then fed
        int8 rows would be silently half-filled with garbage. The ordering is
        enforced in `ExpertCache.__init__`, which rejects a store whose row
        geometry has moved underneath it.

        Peak overhead is one layer of each representation (~1.2 GB for OLMoE),
        not the whole store: each layer is replaced before the next is touched.

        Irreversible, like `fake_quantize_int8`. Returns the same weight-error
        dict, which is a sanity check and *not* the acceptance test — see
        `fake_quantize_int8` for what the acceptance test is and what it cost to
        find out that greedy token ids were not it.
        """
        if self.shape.quant is not None:
            raise RuntimeError(
                f"Store is already quantised ({self.shape.quant}). Quantisation "
                "is lossy and irreversible; quantising twice would compound the "
                "error while reporting only the second round's."
            )
        spec = spec or QuantSpec()
        source = self.shape
        target = replace(source, quant=spec)
        spans = target.spans
        m = source.matrix_numel

        squared_error = 0.0
        squared_weight = 0.0
        worst = 0.0
        chunk = 8  # see fake_quantize_int8 for why the fp32 transient is bounded

        for layer in self.layers:
            old = self._layers[layer]
            want_pinned = self.is_pinned(layer)
            try:
                fresh = torch.empty(
                    (self.num_experts, target.nbytes),
                    dtype=target.row_dtype,
                    pin_memory=want_pinned,
                )
            except RuntimeError as exc:
                # Same ceiling as `from_model` and `repin`, reached from a worse
                # position: quantising a pinned store holds the old pinned layer
                # and the new one at once, so the peak is above both stores'
                # steady state. Degrade rather than lose an hour of model load.
                # `install_expert_cache` avoids this entirely by quantising a
                # pageable store and pinning afterwards.
                if not want_pinned:
                    raise
                log.warning(
                    "Could not pin quantised layer %d: %s\n"
                    "Falling back to pageable for it; call repin() afterwards.",
                    layer, str(exc).splitlines()[0],
                )
                self._pinned.discard(layer)
                want_pinned = False
                fresh = torch.empty(
                    (self.num_experts, target.nbytes), dtype=target.row_dtype
                )
            for i, name in enumerate(PROJECTIONS):
                lo, hi = spans[name]
                slo, shi = spans[f"{name}.scale"]
                for start in range(0, self.num_experts, chunk):
                    stop = min(start + chunk, self.num_experts)
                    flat = old[start:stop, i * m : (i + 1) * m]
                    if not target.is_quantized(name):
                        # Straight through, but into the new row's own offset:
                        # an exempt projection sits after two int8 ones, so its
                        # byte position is not the one it had in the fp16 row.
                        fresh[start:stop, lo:hi].view(source.dtype).copy_(flat)
                        continue

                    w = flat.unflatten(1, target.matrix_shape(name)).to(torch.float32)
                    q, scale = quantize_matrix(w, spec.group_size)
                    fresh[start:stop, lo:hi].copy_(q.flatten(1))
                    fresh[start:stop, slo:shi].view(source.dtype).copy_(
                        scale.flatten(1).to(source.dtype)
                    )

                    # Measured against what the runtime will actually read back,
                    # which is the fp16 round-trip of the scale, not the fp32
                    # scale the quantiser computed.
                    stored = fresh[start:stop, slo:shi].view(source.dtype)
                    deq = dequantize_matrix(
                        q, stored.unflatten(1, (w.shape[1], -1)).float(), torch.float32
                    )
                    error = deq - w
                    squared_error += float(error.pow(2).sum())
                    squared_weight += float(w.pow(2).sum())
                    worst = max(worst, float(error.abs().max() / w.abs().max()))
                    del w, q, scale, deq, error

            self._layers[layer] = fresh
            del old

        self.shape = target
        gc.collect()
        rms = (squared_error / squared_weight) ** 0.5 if squared_weight else 0.0
        return {"rel_rms_error": rms, "worst_channel_rel_error": worst}

    def matrix(self, layer: int, expert: int, name: str) -> torch.Tensor:
        """One projection of one expert, dequantised, as (out, in) in `shape.dtype`.

        Host-side and deliberately slow. This is the reference the tests hold
        the GPU path to, so it reconstructs from the stored bytes by the same
        route the cache does rather than from anything kept on the side.
        """
        shape = self.shape
        row = self.row(layer, expert)
        mshape = shape.matrix_shape(name)
        if shape.quant is None:
            # An unquantised row is indexed in *elements*, not bytes; `spans`
            # describes the packed byte layout and the two only coincide when
            # every projection happens to be one byte wide.
            start = PROJECTIONS.index(name) * shape.matrix_numel
            return row[start : start + shape.matrix_numel].unflatten(0, mshape)
        lo, hi = shape.spans[name]
        if not shape.is_quantized(name):
            return row[lo:hi].view(shape.dtype).unflatten(0, mshape)
        out_features = mshape[0]
        slo, shi = shape.spans[f"{name}.scale"]
        q = row[lo:hi].unflatten(0, mshape).unsqueeze(0)
        scale = row[slo:shi].view(shape.dtype).unflatten(0, (out_features, -1)).unsqueeze(0)
        return dequantize_matrix(q, scale, shape.dtype)[0]

    def fake_quantize_int8(
        self, group_size: int = 0, projections: tuple[str, ...] = PROJECTIONS
    ) -> dict[str, float]:
        """Round-trip every expert through int8, in place.

        This changes the numbers without changing the bytes: rows stay fp16, so
        transfers, cache capacity and throughput are all untouched. That is the
        entire point. Stage 1e-2's case for int8 is a *bytes* argument —
        512 slots instead of 256 and 6.29 MB per miss instead of 12.58 — and its
        only real risk is a *numerics* one. Separating them means the numerics
        question can be answered in one run, before any of the plumbing exists,
        and a failure here kills the stage for 20 minutes rather than two days.

        Symmetric, with `group_size` controlling how much weight shares a scale.

        `group_size=0` is one scale per output channel — for OLMoE that is
        2*intermediate + hidden = 4,096 fp16 values against 6.29 MB, so 0.13%
        overhead, and it is strictly better than the per-tensor variant at
        effectively no cost. It was the first thing tried and it **missed**:
        98.32% teacher-forced top-1 agreement against a pre-committed 99% bar,
        at a mean KL of 0.0024 nats. The distribution barely moves; the argmax
        flips a little too often.

        `group_size=128` splits each row into blocks of 128 inputs that share a
        scale, which is the standard fix and costs 1.6% storage rather than
        0.13%. The scales stay small enough to keep fully resident on the GPU
        either way, so this does not change the design — only the constant.

        Not a tolerance to be relaxed when it fails. The bar is a property of
        the model's output, so missing it means the quantiser is too coarse, and
        the answer is a finer grid.

        Irreversible — the low bits are gone. A caller that wants to compare
        against unquantised has to measure that first.

        Returns the relative error, which is a sanity check on the quantiser
        and *not* the acceptance test. Weight error does not linearly predict
        output error; the acceptance test is greedy token ids on the real model
        (troubleshoot.md 4.6).
        """
        shape = self.shape
        if shape.quant is not None:
            raise RuntimeError(
                "Store is already quantised for real; a fake round-trip on top "
                "would measure the second rounding only."
            )
        m = shape.matrix_numel

        squared_error = 0.0
        squared_weight = 0.0
        worst = 0.0
        # A whole layer's projection promoted to fp32 is ~512 MB for OLMoE, and
        # with the store itself resident this machine has run as low as 1.0 GB
        # free. Chunking keeps the transient under 200 MB; the arithmetic is
        # identical either way because every scale is per output channel and so
        # never spans experts.
        chunk = max(1, 8)
        for layer in self.layers:
            rows = self._layers[layer]
            for i, name in enumerate(PROJECTIONS):
                # Leaving a projection in fp16 costs capacity — two of three
                # quantised is 8.39 MB per expert rather than 6.29, so 384 slots
                # instead of 512 — and buys back accuracy. down_proj is the one
                # worth exempting first: it consumes the product of two
                # activations, so it sees the widest dynamic range of the three.
                if name not in projections:
                    continue
                for start in range(0, self.num_experts, chunk):
                    block = rows[start : start + chunk, i * m : (i + 1) * m]
                    # fp32 for the quantiser's own arithmetic. In fp16 an amax
                    # over 2,048 elements and a division by a small scale both
                    # lose precision, and the measured error would then be the
                    # quantiser's rounding rather than int8's.
                    w = block.unflatten(1, shape.matrix_shape(name)).to(torch.float32)
                    # Group along the *input* dimension, so a scale covers a
                    # contiguous span of the dot product rather than a slice
                    # across unrelated output channels. group_size=0, or a row
                    # that does not divide evenly, falls back to the whole row.
                    q, scale = quantize_matrix(w, group_size)
                    deq = dequantize_matrix(q, scale, torch.float32)

                    error = deq - w
                    squared_error += float(error.pow(2).sum())
                    squared_weight += float(w.pow(2).sum())
                    worst = max(worst, float(error.abs().max() / w.abs().max()))
                    block.copy_(deq.flatten(1).to(shape.dtype))
                    del w, q, scale, deq, error

        rms = (squared_error / squared_weight) ** 0.5 if squared_weight else 0.0
        return {"rel_rms_error": rms, "worst_channel_rel_error": worst}

    # -- access ------------------------------------------------------------

    @property
    def layers(self) -> list[int]:
        return sorted(self._layers)

    @property
    def total_bytes(self) -> int:
        return len(self._layers) * self.num_experts * self.shape.nbytes

    @property
    def pinned_bytes(self) -> int:
        return len(self._pinned) * self.num_experts * self.shape.nbytes

    def is_pinned(self, layer: int) -> bool:
        return layer in self._pinned

    def row(self, layer: int, expert: int) -> torch.Tensor:
        """The flat host row for one expert — the source of a cache fill."""
        try:
            rows = self._layers[layer]
        except KeyError:
            raise KeyError(
                f"Layer {layer} is not in the store (have {self.layers})."
            ) from None
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"Expert {expert} out of range for {self.num_experts} experts.")
        return rows[expert]

    def describe(self) -> str:
        pinned_gb = self.pinned_bytes / (1 << 30)
        total_gb = self.total_bytes / (1 << 30)
        share = f"{pinned_gb / total_gb:.0%}" if total_gb else "0%"
        return (
            f"{len(self._layers)} layers x {self.num_experts} experts | "
            f"{self.shape.nbytes / 1e6:.2f} MB each | {total_gb:.2f} GB total | "
            f"{pinned_gb:.2f} GB pinned ({share} of it, {len(self._pinned)} layers)"
        )


# ==========================================================================
# Model surgery
# ==========================================================================

def _find_expert_blocks(model: nn.Module, moe_layers: list[int]) -> dict[int, nn.ModuleList]:
    """Locate `layers.<i>.mlp.experts` for each MoE layer."""
    wanted = set(moe_layers)
    found: dict[int, nn.ModuleList] = {}
    for layer_idx in sorted(wanted):
        block = _mlp_of(model, layer_idx)
        experts = getattr(block, "experts", None) if block is not None else None
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            found[layer_idx] = experts
    return found


def _mlp_of(model: nn.Module, layer_idx: int) -> nn.Module | None:
    for name, module in model.named_modules():
        if name.endswith(f"layers.{layer_idx}.mlp"):
            return module
    return None


def _detach_experts(model: nn.Module, layer_idx: int) -> None:
    block = _mlp_of(model, layer_idx)
    if block is not None and hasattr(block, "experts"):
        block.experts = nn.ModuleList()


def _shape_of(expert: nn.Module) -> ExpertShape:
    gate = getattr(expert, "gate_proj", None)
    if gate is None:
        raise ValueError(
            f"Expert module {type(expert).__name__} has no gate_proj; expected the "
            "SwiGLU triple (gate_proj, up_proj, down_proj)."
        )
    return ExpertShape(
        hidden_size=int(gate.in_features),
        intermediate_size=int(gate.out_features),
        dtype=gate.weight.dtype,
    )


def _flatten_expert(expert: nn.Module, shape: ExpertShape) -> torch.Tensor:
    """Pack one expert's three projections into a single flat row.

    Note `down_proj` is transposed relative to the other two — (hidden,
    intermediate) rather than (intermediate, hidden). Flattening hides that,
    so `ExpertCache.views` has to restore each matrix's own shape rather than
    assuming a common one. Getting this wrong produces a model that runs and
    returns garbage, which is why the two sides share `PROJECTIONS`.
    """
    parts = []
    for name in PROJECTIONS:
        projection = getattr(expert, name, None)
        if projection is None:
            raise ValueError(f"Expert is missing {name}.")
        weight = projection.weight
        if weight.is_meta:
            raise ValueError(
                f"{name} is a meta tensor — the model was loaded with accelerate "
                "offloading. Load it with device_map=None or 'cpu' so the runtime "
                "owns placement instead of accelerate."
            )
        if weight.numel() != shape.matrix_numel:
            raise ValueError(
                f"{name} has {weight.numel()} elements, expected {shape.matrix_numel}; "
                "experts are not uniformly shaped."
            )
        parts.append(weight.detach().to("cpu", shape.dtype).reshape(-1))
    return torch.cat(parts)
