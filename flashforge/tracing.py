"""Router hooks and trace collection.

What gets captured, per (sequence, layer):

  * full router logits  [T, num_experts] float16  — small, and everything else
    in the analysis can be derived from them (top-k sets, gate weights, the
    "stale router" predictor). Cheap enough to always keep.
  * hidden states       [T, hidden_size] float16  — the gate's *input*. Needed
    for the Q3 probe (can layer N's hidden state predict layer N+k's routing?).
    Bigger; toggled with save_hidden.

Positions are set explicitly by the caller rather than inferred, so prefill
(one forward, T tokens) and decode (T forwards of 1 token) both land in the
same coordinate system.

Batch size must be 1. The gate sees a flattened [B*T, H] tensor, so with B>1
we could not recover which row belongs to which sequence without replicating
the model's own view logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import MoESpec

# Long-format trace schema. One row per (sequence, position, layer, rank).
#
# is_decode marks which phase produced the row: 0 for prefill, 1 for a real
# greedy decode step. Prefill sees every position in one forward pass; decode
# sees one token at a time against a growing KV cache. Same model, but the
# access pattern a cache actually faces at serving time is the second one.
# Deriving the split from `pos >= prefill_len` only works while every sequence
# truncates to the same length, so it is recorded rather than inferred.
TRACE_DTYPES = {
    "seq_id": "int32",
    "pos": "int32",
    "layer": "int16",
    "rank": "int8",
    "expert": "int16",
    "weight": "float32",
    "is_decode": "int8",
}


@dataclass
class SequenceBuffer:
    """Per-sequence accumulation, flushed to disk when the sequence ends."""

    seq_id: int
    logits: dict[int, list[np.ndarray]] = field(default_factory=dict)
    hidden: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def add(self, layer: int, logits: np.ndarray, hidden: np.ndarray | None) -> None:
        self.logits.setdefault(layer, []).append(logits)
        if hidden is not None:
            self.hidden.setdefault(layer, []).append(hidden)

    def stack(self, store: dict[int, list[np.ndarray]]) -> dict[str, np.ndarray]:
        return {str(layer): np.concatenate(chunks, axis=0) for layer, chunks in store.items()}


class RouterTracer:
    """Forward-hook based capture of MoE routing decisions.

    Usage:
        with RouterTracer(spec, out_dir) as tracer:
            tracer.begin_sequence(0)
            tracer.set_window(0, input_ids.shape[1])
            model(input_ids)
            tracer.end_sequence()
    """

    def __init__(
        self,
        spec: MoESpec,
        out_dir: str | Path,
        *,
        save_hidden: bool = True,
    ) -> None:
        self.spec = spec
        self.out_dir = Path(out_dir)
        self.save_hidden = save_hidden
        self.records: list[dict[str, np.ndarray]] = []

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._buffer: SequenceBuffer | None = None
        self._pos_start = 0
        self._pos_len = 0

        (self.out_dir / "logits").mkdir(parents=True, exist_ok=True)
        if save_hidden:
            (self.out_dir / "hidden").mkdir(parents=True, exist_ok=True)

    # ---- lifecycle -----------------------------------------------------

    def __enter__(self) -> "RouterTracer":
        for layer_idx, gate in self.spec.gates:
            self._handles.append(gate.register_forward_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_sequence(self, seq_id: int) -> None:
        self._buffer = SequenceBuffer(seq_id=seq_id)

    def set_window(self, start: int, length: int, *, is_decode: bool = False) -> None:
        """Declare the absolute token positions the next forward will cover.

        `is_decode` tags the rows so the two phases can be analysed apart.
        """
        self._pos_start = start
        self._pos_len = length
        self._is_decode = 1 if is_decode else 0

    def end_sequence(self) -> None:
        """Flush the current sequence's dense arrays to disk."""
        if self._buffer is None:
            return
        buf = self._buffer
        np.savez_compressed(
            self.out_dir / "logits" / f"seq_{buf.seq_id:05d}.npz", **buf.stack(buf.logits)
        )
        if self.save_hidden and buf.hidden:
            np.savez(
                self.out_dir / "hidden" / f"seq_{buf.seq_id:05d}.npz", **buf.stack(buf.hidden)
            )
        self._buffer = None

    # ---- the hook ------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            if self._buffer is None:
                return
            hidden = args[0].detach()
            logits = output.detach()

            if hidden.dim() == 3:  # [B, T, H] — some implementations skip the flatten
                if hidden.shape[0] != 1:
                    raise RuntimeError("RouterTracer requires batch size 1")
                hidden = hidden[0]
                logits = logits[0]
            if hidden.shape[0] != self._pos_len:
                raise RuntimeError(
                    f"set_window declared {self._pos_len} tokens but the gate saw "
                    f"{hidden.shape[0]}. Call set_window() before every forward."
                )

            # Mirror the model's own routing math: softmax over all experts,
            # then top-k. Weights are recorded pre-renormalisation so the raw
            # gate mass is preserved; norm_topk_prob is on the spec if the
            # analysis wants to reapply it.
            probs = torch.softmax(logits.float(), dim=-1)
            weights, experts = torch.topk(probs, self.spec.top_k, dim=-1)

            self._buffer.add(
                layer_idx,
                logits.to("cpu", torch.float16).numpy(),
                hidden.to("cpu", torch.float16).numpy() if self.save_hidden else None,
            )

            n_tokens = hidden.shape[0]
            positions = np.arange(
                self._pos_start, self._pos_start + n_tokens, dtype=np.int32
            )
            self.records.append(
                {
                    "seq_id": np.full(n_tokens * self.spec.top_k, self._buffer.seq_id, np.int32),
                    "pos": np.repeat(positions, self.spec.top_k),
                    "layer": np.full(n_tokens * self.spec.top_k, layer_idx, np.int16),
                    "rank": np.tile(
                        np.arange(self.spec.top_k, dtype=np.int8), n_tokens
                    ),
                    "expert": experts.to("cpu").numpy().astype(np.int16).ravel(),
                    "weight": weights.to("cpu").numpy().astype(np.float32).ravel(),
                    "is_decode": np.full(
                        n_tokens * self.spec.top_k, getattr(self, "_is_decode", 0), np.int8
                    ),
                }
            )

        return hook

    # ---- output --------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame({name: np.empty(0, dt) for name, dt in TRACE_DTYPES.items()})
        columns = {
            name: np.concatenate([rec[name] for rec in self.records]).astype(dt)
            for name, dt in TRACE_DTYPES.items()
        }
        frame = pd.DataFrame(columns)
        return frame.sort_values(["seq_id", "pos", "layer", "rank"], ignore_index=True)

    def write_parquet(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.out_dir / "routing.parquet"
        self.to_frame().to_parquet(path, engine="pyarrow", index=False)
        return path


@torch.no_grad()
def trace_prompt(
    model,
    tokenizer,
    tracer: RouterTracer,
    text: str,
    seq_id: int,
    *,
    max_length: int = 512,
    gen_tokens: int = 0,
) -> int:
    """Run one prompt through the model, capturing routing. Returns token count.

    Prefill gives routing for every position in one forward, which is enough to
    answer all five Stage 0 questions. gen_tokens additionally captures real
    decode steps (greedy), which is the access pattern a cache would actually
    see at serving time.
    """
    device = next(model.parameters()).device
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = encoded["input_ids"].to(device)
    prefill_len = int(input_ids.shape[1])

    tracer.begin_sequence(seq_id)
    tracer.set_window(0, prefill_len)
    outputs = model(input_ids, use_cache=True)

    total = prefill_len
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for step in range(gen_tokens):
        tracer.set_window(prefill_len + step, 1, is_decode=True)
        outputs = model(next_token, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        total += 1
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    tracer.end_sequence()
    return total
