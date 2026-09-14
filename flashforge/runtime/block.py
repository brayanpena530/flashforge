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

TWO EXECUTION PATHS
-------------------
`grouped=False` is the reference: one Python iteration per routed expert, three
`F.linear` calls each, experts visited in ascending order so that fp16
`index_add_` accumulates in exactly the stock block's order. It is bit-exact
against `OlmoeSparseMoeBlock` and that is its job.

`grouped=True` is Stage 1b. It sorts the (token, expert) pairs by expert, pads
each expert's token group to a common length, gathers the routed experts'
weights into one batched tensor and runs three `bmm` calls for the whole layer.
Dispatches per layer go from 3E + 5E bookkeeping ops to a fixed handful.

It is **not** bit-exact, and cannot be: the accumulation happens in one
`index_add_` over all experts at once, whose atomic ordering is not defined.
The parity test therefore holds the loop path to zero difference and the
grouped path to a tolerance. Treat the loop path as the oracle.
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
        grouped: bool = False,
        group_bytes: int = 256 << 20,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = gate
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.act_fn = act_fn
        self.grouped = grouped
        # A prefill batch routes to every expert in the layer, and gathering all
        # 64 of OLMoE's costs 805 MB of transient VRAM — on a 6 GB card holding a
        # 3.9 GB slot pool, that is the difference between running and OOM. The
        # gather is therefore chunked to this budget.
        self.group_bytes = group_bytes
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

        # `unique` returns ascending order, which is the stock loop's order with
        # the unrouted experts removed. Keeping the order matters: index_add_
        # accumulates in fp16 here and floating-point addition is not
        # associative, so a different visit order would give a different (still
        # correct, but not bit-identical) result and make the parity test
        # against the stock block impossible to write tightly.
        #
        # `return_counts` costs nothing extra and the grouped path needs the
        # group sizes. Both paths pay one device->host sync here either way.
        unique_experts, counts = torch.unique(selected_experts, return_counts=True)
        routed = unique_experts.tolist()

        # One cache call per layer rather than per expert, so the fills for a
        # layer are issued back to back. Q8's lesson one tier down was that
        # request batching, not queue depth alone, is what saturates a device.
        slots = self.cache.acquire(self.layer_idx, routed)

        if self.grouped:
            self._forward_grouped(
                hidden_states, routing_weights, selected_experts,
                routed, counts, slots, final_hidden_states,
            )
        else:
            self._forward_loop(
                hidden_states, routing_weights, selected_experts,
                routed, slots, final_hidden_states,
            )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    # -- reference path ----------------------------------------------------

    def _forward_loop(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        routed: list[int],
        slots: dict[int, int],
        final_hidden_states: torch.Tensor,
    ) -> None:
        hidden_dim = hidden_states.shape[1]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in routed:
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            expert_out = self._run_expert(slots[expert_idx], current_state)
            current_hidden_states = expert_out * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    # -- Stage 1b: grouped path --------------------------------------------

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        routed: list[int],
        counts: torch.Tensor,
        slots: dict[int, int],
        final_hidden_states: torch.Tensor,
    ) -> None:
        """Sort tokens by expert, pad to a common group size, run three bmms."""
        device = hidden_states.device
        hidden_dim = hidden_states.shape[1]
        top_k = selected_experts.shape[1]
        pairs = selected_experts.numel()

        # Sort the (token, expert) pairs by expert. `stable` keeps tokens in
        # their original order inside a group, which is not required for
        # correctness but makes the padding pattern reproducible run to run.
        order = torch.argsort(selected_experts.reshape(-1), stable=True)
        token_of = torch.div(order, top_k, rounding_mode="floor")
        weight_of = routing_weights.reshape(-1)[order]

        # Group sizes are already on the host from the `unique` above; the
        # offsets follow from them, so no second sync is needed.
        size_list = counts.tolist()
        offset_list = [0]
        for size in size_list[:-1]:
            offset_list.append(offset_list[-1] + size)

        # The gather is the only large transient: one full expert row per expert
        # in the chunk.
        chunk = max(1, self.group_bytes // self.cache.store.shape.nbytes)

        for start in range(0, len(routed), chunk):
            stop = min(start + chunk, len(routed))
            group_sizes = size_list[start:stop]
            group_offsets = offset_list[start:stop]
            # Pad to the longest group *in this chunk*, not overall: chunking
            # for memory happens to bound the padding waste as well.
            width = max(group_sizes)
            n = stop - start

            sizes = torch.tensor(group_sizes, device=device)
            offsets = torch.tensor(group_offsets, device=device)
            ar = torch.arange(width, device=device)
            live = ar.unsqueeze(0) < sizes.unsqueeze(1)                    # (n, width)
            # Padding slots point at a real row (clamped) and carry weight 0, so
            # they contribute an exact zero to the accumulation rather than
            # needing to be masked out of the scatter.
            pos = (offsets.unsqueeze(1) + ar.unsqueeze(0)).clamp_(max=pairs - 1)
            tokens = token_of[pos]                                        # (n, width)
            weights = (weight_of[pos] * live).to(hidden_states.dtype)

            flat_tokens = tokens.reshape(-1)
            x = hidden_states.index_select(0, flat_tokens).view(n, width, hidden_dim)

            slot_ids = torch.tensor(
                [slots[e] for e in routed[start:stop]], device=device, dtype=torch.long
            )
            gate_w, up_w, down_w = self.cache.gather(slot_ids)

            # F.linear is x @ W.T; bmm needs that spelled out.
            act = self.act_fn(torch.bmm(x, gate_w.transpose(1, 2)))
            out = torch.bmm(act * torch.bmm(x, up_w.transpose(1, 2)), down_w.transpose(1, 2))
            out = out * weights.unsqueeze(-1)

            final_hidden_states.index_add_(0, flat_tokens, out.view(-1, hidden_dim))

    def _run_expert(self, slot: int, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU against cache slot views — `F.linear` is `x @ W.T`, as nn.Linear."""
        gate = self.cache.gate_proj(slot)
        up = self.cache.up_proj(slot)
        down = self.cache.down_proj(slot)
        return F.linear(self.act_fn(F.linear(x, gate)) * F.linear(x, up), down)
