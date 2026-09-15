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
        # Stage 1c. `next_gate` is the *next* MoE layer's real router module,
        # wired up by install_expert_cache. Running it on this layer's hidden
        # state is Q3's `stale_router` predictor, which recalled 0.835 of the
        # true top-k one layer ahead — the best of the predictors swept, and
        # nearly free: one (tokens x hidden) @ (hidden x num_experts) matmul
        # against the 3 x top_k expert GEMMs it is trying to overlap.
        self.prefetch = False
        self.next_gate: nn.Linear | None = None
        self.next_layer_idx: int | None = None
        # How many of the predicted experts to actually fetch. `None` means
        # top_k — every expert the prediction names, which is what the first
        # version did and what made it move 24% more bytes than the demand path
        # for a 78% precision return. Fetching fewer spends the speculation
        # budget on the confident end of the prediction only.
        self.prefetch_k: int | None = None
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

        # ONE device->host sync per layer, and it is worth saying why this is
        # written the way it is rather than the obvious way.
        #
        # The obvious way is `torch.unique(selected_experts, return_counts=True)`
        # and then `.tolist()` on each — two transfers. Add prefetch's predicted
        # set and it is three, sixteen times per token. Stage 1c measured what
        # that costs: the first prefetcher took 68 ms/token of transfer stall
        # off the critical path and returned 4% more tokens, because each sync
        # it added handed the stall straight back.
        #
        # `bincount` gives the routed set *and* the group sizes as one
        # fixed-size vector, so the prediction concatenates onto it and the
        # whole layer costs a single 64- or 128-element copy. Walking it
        # host-side yields ascending order, which is what `unique` gave and what
        # fp16 `index_add_` needs to stay bit-exact against the stock block.
        counts_all = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts)

        predicted: list[int] | None = None
        if self.prefetch and self.next_gate is not None and not self._is_prefill(hidden_states):
            fused = torch.cat([counts_all, self._predict_next(hidden_states)]).tolist()
            predicted = [e for e in range(self.num_experts) if fused[self.num_experts + e]]
        else:
            fused = counts_all.tolist()

        routed = [e for e in range(self.num_experts) if fused[e]]
        group_sizes = [fused[e] for e in routed]

        # One cache call per layer rather than per expert, so the fills for a
        # layer are issued back to back. Q8's lesson one tier down was that
        # request batching, not queue depth alone, is what saturates a device.
        slots = self.cache.acquire(self.layer_idx, routed)

        # Issued after this layer's own experts are resident and before its
        # GEMMs are enqueued, which is the only window where the copies have
        # something to hide behind: they run on the side stream while the bmms
        # below run on the compute stream.
        if predicted:
            self.cache.prefetch(self.next_layer_idx, predicted)

        if self.grouped:
            self._forward_grouped(
                hidden_states, routing_weights, selected_experts,
                routed, group_sizes, slots, final_hidden_states,
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
        size_list: list[int],
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

        # Group sizes arrive as a host list from `forward`'s single sync; the
        # offsets follow from them, so nothing here touches the device.
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

    # -- Stage 1c: speculative prefetch -------------------------------------

    def _is_prefill(self, hidden_states: torch.Tensor) -> bool:
        """More tokens than experts, so the batch routes to essentially all of them.

        Prefetching that is pure loss: there is nothing left to predict, and the
        speculative fetch would be a second whole-layer transfer stacked on top
        of the demand one it cannot avoid.
        """
        return hidden_states.shape[0] > self.num_experts

    def _predict_next(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Q3's `stale_router`: the next layer's real gate, on this layer's input.

        Returns a `num_experts` count vector, on the device and deliberately
        *not* synced — `forward` concatenates it onto this layer's own counts so
        both cross to the host in one copy.

        The guess cannot affect the output. It only decides which weights are
        already in VRAM when the next layer asks, and the next layer asks its
        own router regardless — so a wrong prediction costs PCIe bandwidth and a
        cache slot, never a wrong expert. That is what makes a predictor at
        0.835 recall usable at all, and it is why this is allowed to be cheap
        and approximate when the thing it feeds is not.

        `prefetch_k` truncates the guess. `topk` returns logits in descending
        order, so taking fewer keeps the router's most confident predictions and
        drops the marginal ones — which are also the ones most likely to be
        wrong. The point is not to predict better but to spend less: the first
        version fetched all `top_k` and paid full freight for a 22% error rate
        on the link that was already the bottleneck.
        """
        k = min(self.prefetch_k or self.top_k, self.num_experts)
        with torch.no_grad():
            _, predicted = torch.topk(self.next_gate(hidden_states), k, dim=-1)
            return torch.bincount(predicted.reshape(-1), minlength=self.num_experts)

    def _run_expert(self, slot: int, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU against cache slot views — `F.linear` is `x @ W.T`, as nn.Linear."""
        gate = self.cache.gate_proj(slot)
        up = self.cache.up_proj(slot)
        down = self.cache.down_proj(slot)
        return F.linear(self.act_fn(F.linear(x, gate)) * F.linear(x, up), down)
