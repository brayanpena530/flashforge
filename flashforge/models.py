"""Model loading and MoE router discovery.

The discovery logic is deliberately structural rather than model-specific: it
looks for `model.layers.<i>.mlp.gate` modules that are `nn.Linear` projecting
into `num_experts`. That pattern covers OLMoE and Qwen3-MoE unchanged, which is
the whole dev ladder for Stage 0.

DeepSeek's `MoEGate` is a custom module rather than a bare Linear and will need
its own branch here later; `discover_moe` raises a clear error instead of
silently finding nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import torch
from torch import nn

log = logging.getLogger(__name__)

# Layer index is captured from the module path so we keep the model's own
# numbering even when only some layers are MoE (many models keep layer 0 dense).
_GATE_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"


@dataclass
class MoESpec:
    """Everything the tracer needs to know about a model's routing."""

    num_experts: int
    top_k: int
    hidden_size: int
    norm_topk_prob: bool
    # (model layer index, the nn.Linear that produces router logits)
    gates: list[tuple[int, nn.Module]] = field(repr=False, default_factory=list)

    @property
    def moe_layers(self) -> list[int]:
        return [idx for idx, _ in self.gates]

    @property
    def n_moe_layers(self) -> int:
        return len(self.gates)

    def describe(self) -> str:
        return (
            f"{self.n_moe_layers} MoE layers | {self.num_experts} experts/layer | "
            f"top-{self.top_k} | hidden {self.hidden_size} | "
            f"{self.num_experts * self.n_moe_layers} distinct (layer, expert) slots"
        )


def _cfg_attr(config, names: tuple[str, ...], default=None):
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def discover_moe(model: nn.Module) -> MoESpec:
    """Locate router gates and read the routing hyperparameters off the config."""
    config = model.config
    num_experts = _cfg_attr(config, ("num_experts", "n_routed_experts", "num_local_experts"))
    top_k = _cfg_attr(config, ("num_experts_per_tok", "top_k", "moe_top_k"))
    hidden_size = _cfg_attr(config, ("hidden_size", "d_model"))
    norm_topk_prob = bool(_cfg_attr(config, ("norm_topk_prob",), default=False))

    if num_experts is None or top_k is None:
        raise ValueError(
            f"Could not read MoE config from {type(config).__name__}. "
            "Expected num_experts / num_experts_per_tok (or the n_routed_experts alias)."
        )

    gates: list[tuple[int, nn.Module]] = []
    for name, module in model.named_modules():
        match = _GATE_PATH.search(name)
        if not match:
            continue
        if not isinstance(module, nn.Linear):
            log.warning("Skipping %s: expected nn.Linear gate, got %s", name, type(module).__name__)
            continue
        if module.out_features != num_experts:
            log.warning(
                "Skipping %s: out_features=%d but config says %d experts",
                name, module.out_features, num_experts,
            )
            continue
        gates.append((int(match.group(1)), module))

    if not gates:
        raise ValueError(
            "No router gates found. This model's routing module is not a bare "
            "nn.Linear at `layers.<i>.mlp.gate` (DeepSeek's MoEGate is one such "
            "case) — add a branch to discover_moe() for it."
        )

    gates.sort(key=lambda pair: pair[0])
    return MoESpec(
        num_experts=int(num_experts),
        top_k=int(top_k),
        hidden_size=int(hidden_size),
        norm_topk_prob=norm_topk_prob,
        gates=gates,
    )


def gate_weight_matrices(spec: MoESpec) -> dict[str, "torch.Tensor"]:
    """Router weight matrices keyed by layer index, as float32 CPU tensors.

    Saved alongside the traces so the "stale router" predictor in the Q3
    analysis can run layer N+k's router on layer N's hidden state without
    reloading the model.
    """
    return {
        str(layer_idx): gate.weight.detach().to("cpu", torch.float32)
        for layer_idx, gate in spec.gates
    }


def load_model(
    model_id: str = DEFAULT_MODEL,
    *,
    dtype: str = "float16",
    device_map: str | dict | None = "auto",
    gpu_memory: str = "4.5GiB",
    cpu_memory: str = "22GiB",
    load_4bit: bool = False,
    cache_dir: str | None = None,
):
    """Load a causal LM sized for a 6GB card with CPU spillover.

    OLMoE-1B-7B is 6.9B total parameters — about 13.8GB in fp16, so it does not
    fit in 6GB of VRAM. The default device_map="auto" plus max_memory splits it
    across GPU and system RAM. That is slow for generation but entirely fine for
    Stage 0, where we only need forward passes to read routing decisions.

    load_4bit shrinks it to roughly 4GB (fits VRAM, much faster to iterate) at
    the cost of perturbing hidden states and therefore routing. Fine for
    exploratory runs; use fp16 for numbers you intend to trust.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    kwargs: dict = {"torch_dtype": torch_dtype, "cache_dir": cache_dir}

    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = device_map or "auto"
    elif device_map is not None:
        kwargs["device_map"] = device_map
        if torch.cuda.is_available():
            kwargs["max_memory"] = {0: gpu_memory, "cpu": cpu_memory}

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer
