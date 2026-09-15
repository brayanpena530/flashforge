"""Install the expert cache into an already-loaded model.

The sequence matters and is easy to get subtly wrong, so it lives in one place:

1. Discover the routers (reuses Stage 0's structural `discover_moe`).
2. Copy every expert into the host store, detaching each layer's experts from
   the model as it goes so the two copies never both exist in full.
3. Replace each MoE block with a `CachedMoEBlock` that keeps the original gate.
4. *Then* move the model to the GPU — which now means the non-expert weights
   only, because the experts are no longer attached to it.

Step 4 is the point of the whole exercise. OLMoE-1B-7B is 6.9B parameters,
13.8 GB in fp16, against 6 GB of VRAM. Of that, 6.44B parameters are experts:
strip them out and the residual model is ~0.9 GB, which leaves most of the card
for the expert cache. The model must therefore be loaded to **CPU** first —
`device_map="auto"` would hand placement to accelerate, scatter experts across
devices and meta tensors, and leave the runtime fighting it for control.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from flashforge.models import MoESpec, discover_moe
from flashforge.runtime.block import CachedMoEBlock
from flashforge.runtime.cache import ExpertCache
from flashforge.runtime.store import ExpertStore

log = logging.getLogger(__name__)


@dataclass
class PatchReport:
    """What the install actually did, for printing and for assertions."""

    spec: MoESpec
    store: ExpertStore
    cache: ExpertCache
    blocks: list[CachedMoEBlock]
    resident_bytes: int

    def describe(self) -> str:
        return (
            f"patched {len(self.blocks)} MoE layers\n"
            f"  store: {self.store.describe()}\n"
            f"  cache: {self.cache.describe()}\n"
            f"  resident on device: {self.resident_bytes / (1 << 30):.2f} GB "
            f"(non-expert weights + cache)"
        )


def install_expert_cache(
    model: nn.Module,
    *,
    capacity: int,
    device: torch.device | str = "cuda",
    pin_gb: float = 0.0,
    grouped: bool = True,
    prefetch: bool = False,
) -> PatchReport:
    """Move experts to host RAM, front them with a GPU cache, return the wiring.

    `capacity` is in experts, not bytes — it is the same unit Q5's sweep used,
    so a capacity chosen from the hit-rate curve transfers directly.

    `grouped` selects Stage 1b's batched-GEMM path: decode 4.61 -> 5.93 tok/s
    (+29%, against a 14% within-path spread over nine passes), prefill
    unresolvable. It defaults **on** despite not being bit-exact, because on the
    real model in fp16 the two paths produced identical greedy output for 65
    tokens — the divergence is below the argmax. Pass `grouped=False` for the
    bit-exact loop path, which is what the parity tests hold to zero.

    `prefetch` selects Stage 1c's side-stream speculative fill. It defaults
    **off**, and unlike `grouped` that is not caution — it is a measurement.
    Over nine passes it is 14% *slower* than the demand path (p=0.010), and
    truncating it to the four most confident predictions only gets it back to
    indistinguishable (p=0.09).

    The mechanism is not what failed. Blocking fill drops 116.1 -> 36.0 ms/token
    and the decode hit rate goes 47.5% -> 82.4%. But the link is saturated at
    ~5.8 GB/s, so decode speed is bandwidth / bytes-per-token, and speculation
    at 78.7% precision adds 23% more bytes. Overlap cannot pay for them.

    It is kept because it is correct, cheap to re-test, and the conclusion is
    operating-point-specific: a machine with bandwidth to spare, or a store
    pinned well past this one's 9-of-16 layers, changes the arithmetic. Run
    `ff-serve --path grouped,prefetch` and read the sustained-GB/s column before
    turning it on. It requires pinned host memory to do anything at all — on a
    pageable store `copy_(non_blocking=True)` is synchronous, the side stream
    cannot overlap, and the speculation is pure loss. Pass `pin_gb`.
    """
    device = torch.device(device)
    spec = discover_moe(model)
    act_fn = _activation_of(model)

    store = ExpertStore.from_model(model, spec.moe_layers, pin_gb=pin_gb)
    gc.collect()

    cache = ExpertCache(store, capacity, device=device, dtype=store.shape.dtype)

    blocks: list[CachedMoEBlock] = []
    for layer_idx, gate in spec.gates:
        block = CachedMoEBlock(
            layer_idx,
            gate,
            cache,
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            norm_topk_prob=spec.norm_topk_prob,
            act_fn=act_fn,
            grouped=grouped,
        )
        _replace_module(model, f"layers.{layer_idx}.mlp", block)
        blocks.append(block)

    # Stage 1c: each block gets a handle on the next one's router, so it can run
    # Q3's `stale_router` predictor on its own hidden state. The last MoE layer
    # keeps `next_gate = None` — there is no layer after it to prefetch for, and
    # guessing into the next *token* is a different predictor with a different
    # (much longer) horizon.
    for block, following in zip(blocks, blocks[1:]):
        block.next_gate = following.gate
        block.next_layer_idx = following.layer_idx
        block.prefetch = prefetch

    model.to(device)
    gc.collect()

    resident = sum(p.numel() * p.element_size() for p in model.parameters())
    return PatchReport(
        spec=spec,
        store=store,
        cache=cache,
        blocks=blocks,
        resident_bytes=resident + cache.bytes_resident,
    )


def _activation_of(model: nn.Module):
    """The expert FFN's activation, read from config rather than assumed.

    Every model on the dev ladder uses SiLU, but reading it keeps a future
    model with a different `hidden_act` from silently producing wrong numbers
    instead of an error.
    """
    name = getattr(model.config, "hidden_act", None)
    if name is None:
        return F.silu
    try:
        from transformers.activations import ACT2FN

        return ACT2FN[name]
    except (ImportError, KeyError):
        log.warning("Unknown hidden_act %r; falling back to SiLU.", name)
        return F.silu


def _replace_module(model: nn.Module, suffix: str, replacement: nn.Module) -> None:
    """Swap the module whose qualified name ends with `suffix`."""
    for name, module in list(model.named_modules()):
        if not name.endswith(suffix):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, replacement)
        return
    raise KeyError(f"No module matching *{suffix} in {type(model).__name__}.")
