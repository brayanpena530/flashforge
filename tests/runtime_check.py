"""Correctness checks for the Stage 1 offload runtime.

Runs on CPU against a randomly initialised OLMoE block — no model download, no
GPU, a few seconds:

    uv run python tests/runtime_check.py

The load-bearing check is **parity**: `CachedMoEBlock` must produce the same
output as the stock `OlmoeSparseMoeBlock` it replaces. Everything else here is
about the cache being a correct cache; parity is about the runtime still being
the same model. A fast wrong answer is the failure mode this whole file exists
to catch, because nothing downstream would notice — generation keeps working,
the text just quietly gets worse.

The cache checks lean on one property that is easy to lose in a refactor: a
victim must never be an expert the in-flight request is about to read. That
bug does not throw. It silently serves whatever weights last landed in the
reused slot, which is exactly the kind of corruption parity would only catch by
luck.
"""
import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flashforge.cli import _force_utf8_stdout  # noqa: E402
from flashforge.runtime.cache import ExpertCache  # noqa: E402
from flashforge.runtime.patch import install_expert_cache  # noqa: E402
from flashforge.runtime.store import ExpertStore  # noqa: E402

_force_utf8_stdout()
torch.manual_seed(0)

H, I, E, K, L = 64, 32, 16, 4, 3
TOKENS = 12

failures: list[str] = []


def check(name, condition, detail):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}: {detail}")
    if not condition:
        failures.append(name)


# ==========================================================================
# A small real OLMoE stack, on CPU, with random weights
# ==========================================================================

from transformers.models.olmoe.configuration_olmoe import OlmoeConfig  # noqa: E402
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock  # noqa: E402

config = OlmoeConfig(
    hidden_size=H,
    intermediate_size=I,
    num_experts=E,
    num_experts_per_tok=K,
    num_hidden_layers=L,
    norm_topk_prob=True,
    hidden_act="silu",
)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = OlmoeSparseMoeBlock(config)


class _Stack(nn.Module):
    """Just enough model for `discover_moe` and the store to walk.

    The module path has to be `layers.<i>.mlp.gate`, because that is the
    structural pattern Stage 0's discovery keys on — a fake with a different
    layout would test the runtime against a model shape no real model has.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(L)])
        self.config = config

    def forward(self, hidden):  # exercised only through the blocks
        for layer in self.layers:
            hidden = layer.mlp(hidden)[0]
        return hidden


model = _Stack().eval().to(torch.float32)
for parameter in model.parameters():
    nn.init.normal_(parameter, std=0.05)

reference = copy.deepcopy(model)
hidden_in = torch.randn(2, TOKENS, H)

with torch.no_grad():
    expected = reference(hidden_in)

# ==========================================================================
# Store
# ==========================================================================

print("Expert store")
probe_store = ExpertStore.from_model(
    copy.deepcopy(model), list(range(L)), detach_from_model=False
)
check("store holds every layer", probe_store.layers == list(range(L)),
      f"layers {probe_store.layers}, {probe_store.num_experts} experts each")
check("expert row is the packed SwiGLU triple",
      probe_store.shape.numel == 3 * H * I,
      f"{probe_store.shape.numel} elements = 3 x {H} x {I}")

source_expert = reference.layers[0].mlp.experts[5]
row = probe_store.row(0, 5)
m = H * I
check("gate_proj round-trips",
      torch.equal(row[:m].reshape(I, H), source_expert.gate_proj.weight),
      "flattened row matches the source weight")
check("down_proj keeps its own (hidden, intermediate) shape",
      torch.equal(row[2 * m:].reshape(H, I), source_expert.down_proj.weight),
      f"({H}, {I}) not ({I}, {H}) — the transpose the other two do not have")

check("nothing is pinned without a budget", not probe_store.is_pinned(0),
      f"pinned {probe_store.pinned_bytes / 1e6:.1f} MB of "
      f"{probe_store.total_bytes / 1e6:.1f} MB")

# Detaching is what keeps peak RAM to one layer rather than the whole model;
# if it silently stopped happening, the runtime would still work and would
# start OOMing only on a model large enough to matter.
detachable = copy.deepcopy(model)
ExpertStore.from_model(detachable, list(range(L)), detach_from_model=True)
check("experts are detached from the model",
      all(len(layer.mlp.experts) == 0 for layer in detachable.layers),
      "every layer's ModuleList is empty, so the weights exist once")

# ==========================================================================
# Cache
# ==========================================================================

print("\nExpert cache")
cache = ExpertCache(probe_store, capacity=8, device="cpu")
slots = cache.acquire(0, [1, 2, 3])
check("first touch is all misses", cache.stats.misses == 3 and cache.stats.hits == 0,
      cache.stats.describe())
check("slots are distinct", len(set(slots.values())) == 3, f"slots {sorted(slots.values())}")

cache.acquire(0, [1, 2, 3])
check("second touch is all hits", cache.stats.hits == 3, cache.stats.describe())

check("cached weights match the store",
      torch.equal(cache.gate_proj(slots[2]), probe_store.row(0, 2)[:m].reshape(I, H)),
      "the slot holds expert 2's real gate_proj")

# (0, 1) and (1, 1) are different tensors and must not share a slot.
same_expert_other_layer = cache.acquire(1, [1])
check("keys are (layer, expert), not expert",
      same_expert_other_layer[1] != slots[1],
      f"layer 0 expert 1 in slot {slots[1]}, layer 1 expert 1 in "
      f"slot {same_expert_other_layer[1]}")

print("\nEviction")
tight = ExpertCache(probe_store, capacity=4, device="cpu")
tight.acquire(0, [0, 1, 2, 3])
tight.acquire(0, [4, 5])
check("evicts to make room", tight.stats.evictions == 2,
      f"{tight.stats.evictions} evictions to fit 6 experts in 4 slots")
check("LRU keeps the most recent", {(0, 2), (0, 3), (0, 4), (0, 5)} == tight.resident(),
      f"resident {sorted(tight.resident())} — 0 and 1 were the oldest")

# The bug this guards: a request whose hits sit at the LRU end could have those
# very entries chosen as victims to make room for its own misses, and then read
# back the wrong weights out of the reused slot. Nothing raises.
protect = ExpertCache(probe_store, capacity=4, device="cpu")
protect.acquire(0, [0, 1, 2, 3])
got = protect.acquire(0, [0, 1, 8, 9])
check("in-flight hits are never evicted",
      torch.equal(protect.gate_proj(got[0]), probe_store.row(0, 0)[:m].reshape(I, H))
      and torch.equal(protect.gate_proj(got[1]), probe_store.row(0, 1)[:m].reshape(I, H)),
      "experts 0 and 1 still hold their own weights after 8 and 9 displaced 2 and 3")

undersized = ExpertCache(probe_store, capacity=2, device="cpu")
try:
    undersized.acquire(0, [0, 1, 2])
    raised = False
except ValueError as exc:
    raised = "cannot complete a forward pass" in str(exc)
check("a cache smaller than one request is an error, not corruption", raised,
      "acquire() refuses rather than thrashing a slot mid-layer")

# ==========================================================================
# Parity — the check everything else is in service of
# ==========================================================================

print("\nBlock parity against stock OlmoeSparseMoeBlock")

for capacity, label in [(L * E, "everything resident"), (E, "thrashing")]:
    patched = copy.deepcopy(model)
    # grouped=False explicitly: this is the bit-exact oracle, and it is the only
    # assertion in the file that would silently weaken if the default flipped.
    report = install_expert_cache(patched, capacity=capacity, device="cpu", grouped=False)
    with torch.no_grad():
        actual = patched(hidden_in)
    delta = (actual - expected).abs().max().item()
    check(f"output matches stock ({label}, capacity {capacity})", delta == 0.0,
          f"max abs difference {delta:.3e} over {actual.numel()} activations")

check("every MoE layer was patched", len(report.blocks) == L,
      report.describe().splitlines()[0])

# One layer's union exactly. Every layer therefore evicts the whole of the
# previous layer's working set before it can run, so almost nothing survives to
# be reused. Parity holding here is the real test: it means eviction and refill
# are correct, not merely untouched.
check("thrashing capacity really did thrash", report.cache.stats.hit_rate < 0.30,
      report.cache.stats.describe())

# The batch union, not top_k, is what the cache has to hold. A single decode
# token needs K slots; this 24-token batch touches every expert in the layer.
# Sizing a cache from top_k would deadlock the first prefill it ever saw.
too_small = copy.deepcopy(model)
try:
    install_expert_cache(too_small, capacity=K + 1, device="cpu")
    with torch.no_grad():
        too_small(hidden_in)
    refused = False
except ValueError as exc:
    refused = "union" in str(exc)
check("a top_k-sized cache is refused on a prefill batch", refused,
      f"{2 * TOKENS} tokens x top-{K} covers all {E} experts, so {K + 1} slots cannot work")

warm = copy.deepcopy(model)
warm_report = install_expert_cache(warm, capacity=L * E, device="cpu")
with torch.no_grad():
    warm(hidden_in)
    warm_report.cache.stats.reset()
    warm(hidden_in)
check("a cache large enough to hold the model never misses twice",
      warm_report.cache.stats.hit_rate == 1.0,
      warm_report.cache.stats.describe())

# ==========================================================================
# Stage 1b — the grouped path
# ==========================================================================

print("\nGrouped path (Stage 1b)")

# Not bit-exact by construction: one index_add_ over every expert at once has
# no defined accumulation order. The tolerance is what this path costs, so it
# is asserted rather than waved at. In fp32 the loop path's own rounding is
# ~1e-7 relative, and these activations are O(1).
for capacity, label in [(L * E, "everything resident"), (E, "thrashing")]:
    grouped_model = copy.deepcopy(model)
    grouped_report = install_expert_cache(
        grouped_model, capacity=capacity, device="cpu", grouped=True
    )
    with torch.no_grad():
        grouped_out = grouped_model(hidden_in)
    delta = (grouped_out - expected).abs().max().item()
    check(f"grouped matches stock within tolerance ({label})", delta < 1e-5,
          f"max abs difference {delta:.3e} (loop path is exactly 0)")

# Padding is the part most likely to be wrong and least likely to announce it:
# a group shorter than the chunk's width is padded with rows that point at a
# real token and must carry weight zero. An imbalanced batch is what exercises
# it — one token routed alone alongside a full batch makes the group sizes
# differ by an order of magnitude.
lopsided = torch.randn(1, 1, H)
solo_ref, solo_grouped = copy.deepcopy(model), copy.deepcopy(model)
install_expert_cache(solo_ref, capacity=L * E, device="cpu", grouped=False)
install_expert_cache(solo_grouped, capacity=L * E, device="cpu", grouped=True)
with torch.no_grad():
    delta = (solo_grouped(lopsided) - solo_ref(lopsided)).abs().max().item()
check("single-token decode agrees with the loop path", delta < 1e-5,
      f"max abs difference {delta:.3e} — every group is width 1, all padding")

# Chunking for VRAM must not change the answer. Forcing one expert per chunk is
# the pathological end of that: maximum chunks, minimum padding.
chunked = copy.deepcopy(model)
chunk_report = install_expert_cache(chunked, capacity=L * E, device="cpu", grouped=True)
for block in chunk_report.blocks:
    block.group_bytes = 1  # -> chunk size clamps to 1 expert
with torch.no_grad():
    delta = (chunked(hidden_in) - grouped_out).abs().max().item()
check("chunking the gather does not change the result", delta < 1e-5,
      f"max abs difference {delta:.3e} at one expert per chunk")

# The gather is where a shape mistake would show as garbage rather than a raise,
# because all three projections have the same element count in this config.
gather_cache = ExpertCache(probe_store, capacity=8, device="cpu")
gather_slots = gather_cache.acquire(0, [3, 7])
slot_ids = torch.tensor([gather_slots[3], gather_slots[7]])
check("the grouped path is the installed default",
      all(block.grouped for block in warm_report.blocks),
      "install_expert_cache defaults to grouped=True; the parity tests above opt "
      "out explicitly so the bit-exact assertion cannot weaken silently")

gate_w, up_w, down_w = gather_cache.gather(slot_ids)
check("gather stacks the right experts in the right shapes",
      gate_w.shape == (2, I, H) and down_w.shape == (2, H, I)
      and torch.equal(gate_w[1], reference.layers[0].mlp.experts[7].gate_proj.weight)
      and torch.equal(up_w[0], reference.layers[0].mlp.experts[3].up_proj.weight)
      and torch.equal(down_w[1], reference.layers[0].mlp.experts[7].down_proj.weight),
      f"gate {tuple(gate_w.shape)}, down {tuple(down_w.shape)} — down keeps its transpose")

print("\nRouting is untouched")
patched = copy.deepcopy(model)
install_expert_cache(patched, capacity=L * E, device="cpu")
with torch.no_grad():
    _, ref_logits = reference.layers[0].mlp(hidden_in)
    _, new_logits = patched.layers[0].mlp(hidden_in)
check("router logits are identical", torch.equal(ref_logits, new_logits),
      "the gate is the original module, so selection cannot drift")

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("All checks passed.")
