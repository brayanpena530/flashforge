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
from flashforge.runtime.store import ExpertStore, QuantSpec

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
    stream_prefill: bool = False,
    quant: QuantSpec | None = None,
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

    `stream_prefill` selects Stage 1e-4. It is the same side-stream machinery
    as `prefetch` with the predictor removed, because during prefill there is
    nothing to predict: a batch past a few dozen tokens routes to nearly every
    expert in the next layer, so fetching all of them is right by construction.

    It defaults **off because it was measured and lost**: prefill 119.8 -> 90.4
    tok/s, **-25% at p=0.011**. It was predicted to win on the grounds that
    `prefetch`'s arithmetic did not apply here — speculation lost because it
    added 23% more bytes to a saturated link, and this was supposed to add ~7%.
    It adds **58%**, because a 128-token prefill layer routes to 41.4 of 64
    experts rather than the "nearly all" the design assumed. Prefill is
    transfer-bound at 78%, so bytes set the clock and the overlap cannot pay.

    The mechanism is not what failed, and the counters say so loudly: prefill
    hit rate 7.1% -> **95.0%**, blocking fill 78.1% -> **3.0%** of prefill wall
    clock. It removed nearly all of the stall it was built to remove.

    It cannot change the model's output — a prefetch only moves where a weight
    already is, never which weight runs — so unlike `quant` there is no
    numerics risk here, and `tests/runtime_check.py` holds it to a bit-exact
    difference of zero. It needs room for two layers of experts at once
    (`capacity >= 2 * num_experts`); below that `prefetch` truncates the fetch
    to whatever is free and the overlap degrades smoothly rather than failing.

    `quant` selects Stage 1e-2's int8 experts. It is applied to the store before
    the cache is built, because the cache sizes its pool from the store's row
    geometry and `capacity` is in experts — so the *same* `capacity` costs 33%
    less VRAM with the default two-of-three spec, and the caller is expected to
    spend that by asking for more. `None` keeps fp16.

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

    # Loaded pageable when quantising, then pinned afterwards. Pinning first
    # would page-lock 12 GB of fp16 rows that are about to be discarded, and
    # then ask for the int8 copies *on top* — which is how the first quantised
    # sweep died on the pinned ceiling at layer 6. Quantise, then pin what is
    # left; `repin` already knows how to degrade when the ceiling is reached.
    store = ExpertStore.from_model(
        model, spec.moe_layers, pin_gb=0.0 if quant is not None else pin_gb
    )
    gc.collect()

    if quant is not None:
        error = store.quantize_int8(quant)
        log.info(
            "int8 experts (%s, group %d): %.2f%% relative RMS weight error, "
            "%.2f MB per expert",
            "+".join(quant.projections), quant.group_size,
            100 * error["rel_rms_error"], store.shape.nbytes / 1e6,
        )
        if pin_gb:
            store.repin(pin_gb)
            gc.collect()

    # Built after quantisation, never before: the pool's width and dtype come
    # from the store's row geometry, and `acquire` refuses a cache whose store
    # moved underneath it rather than reading half an expert.
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
        block.stream_prefill = stream_prefill

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
