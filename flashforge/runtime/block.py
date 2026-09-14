"""A sparse MoE block whose experts come from the cache instead of VRAM.

This is a drop-in replacement for `OlmoeSparseMoeBlock` (and the identically
shaped Qwen3-MoE block). The router half is copied from the original on
purpose: any difference there changes which experts are selected, and a cache
that serves the wrong experts quickly is not an optimisation. The only
behavioural change is where the weights come from.

There is one incidental win. The stock block loops over **all** experts and
runs an empty GEMM for every one that no token routed to — 64 Python
iterations per layer to do 8 experts' worth of work. Looping over the routed
set only is exactly equivalent, because `index_add_` of an empty slice is a
no-op, and it removes 56 of those iterations. That is a real speedup but it has
nothing to do with caching, so `ff-serve` reports the uncached dense-VRAM
baseline separately rather than folding the two together.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from flashforge.runtime.cache import ExpertCache


class CachedMoEBlock(nn.Module):
    """Routes like the stock block; fetches experts through an `ExpertCache`."""

    def __init__(
        self,
        layer_idx: int,
        gate: nn.Linear,
        cache: ExpertCache,
        *,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        act_fn=F.silu,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = gate
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.act_fn = act_fn
        # One cache is shared by every layer. It is deliberately not an
        # nn.Module, so `model.to(...)` and `state_dict()` leave the slot pool
        # alone rather than trying to move or serialise it once per layer.
        self.cache = cache

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # `unique` returns ascending order, which is the stock loop's order with
        # the unrouted experts removed. Keeping the order matters: index_add_
        # accumulates in fp16 here and floating-point addition is not
        # associative, so a different visit order would give a different (still
        # correct, but not bit-identical) result and make the parity test
        # against the stock block impossible to write tightly.
        routed = torch.unique(selected_experts).tolist()

        # One cache call per layer rather than per expert, so the fills for a
        # layer are issued back to back. Q8's lesson one tier down was that
        # request batching, not queue depth alone, is what saturates a device.
        slots = self.cache.acquire(self.layer_idx, routed)

        for expert_idx in routed:
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            expert_out = self._run_expert(slots[expert_idx], current_state)
            current_hidden_states = expert_out * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def _run_expert(self, slot: int, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU against cache slot views — `F.linear` is `x @ W.T`, as nn.Linear."""
        gate = self.cache.gate_proj(slot)
        up = self.cache.up_proj(slot)
        down = self.cache.down_proj(slot)
        return F.linear(self.act_fn(F.linear(x, gate)) * F.linear(x, up), down)
