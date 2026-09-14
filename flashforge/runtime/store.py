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
from dataclasses import dataclass

import torch
from torch import nn

log = logging.getLogger(__name__)

# The three projections of a SwiGLU expert, in the order they are packed into a
# row. Order is arbitrary but must match `ExpertCache.views`, so it lives here
# as the single definition rather than being spelled out in both places.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class ExpertShape:
    """Geometry of one expert, and how it is packed into a flat row."""

    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype

    @property
    def matrix_numel(self) -> int:
        """Elements in one projection. All three are the same size."""
        return self.hidden_size * self.intermediate_size

    @property
    def numel(self) -> int:
        return len(PROJECTIONS) * self.matrix_numel

    @property
    def nbytes(self) -> int:
        return self.numel * torch.empty(0, dtype=self.dtype).element_size()


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
        # layer index -> (num_experts, shape.numel) host tensor
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
            rows = torch.empty(
                (num_experts, shape.numel), dtype=shape.dtype, pin_memory=want_pinned
            )
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
