"""Qualify fp32 scales before building them.

    uv run python tools/scale_precision.py

Stage 1e-2 shipped int8 experts with the per-output-channel scales stored fp16,
riding on the end of the row. The README lists fp32 scales as "the one untried
lever that could recover the bar" — a 0.233-point shortfall on teacher-forced
top-1 agreement, now measured at 0.070% standard error by
`tools/int8_accuracy.py`.

That lever has a ceiling, and the ceiling is computable without touching the
store, the cache, or the GPU. Read real expert matrices straight out of the
safetensors shards, quantise each one three ways, and compare:

    fp32 scale   — the floor. Only int8 rounding remains.
    fp16 scale   — what ships.
    fp32 vs fp16 — the entire budget an upgrade can spend.

This is the Stage 1e-3 move applied to the last open item in Stage 1: the stage
rests on one number nobody has measured, so measure that number first. It costs
no checkpoint load and no model — `safe_open` reads individual tensors.

THE PRE-COMMITTED BAR
---------------------
`int8_accuracy.py` resolves agreement to 0.070%. Weight error and agreement
error are not the same quantity, but they move together monotonically and the
observed mapping is 0.88% relative RMS weight error -> 0.233 points of
agreement. Treating that as locally linear, recovering the bar needs the weight
error to fall by roughly the full 0.233/0.233 = 100%, and *detecting* any
change at all needs it to fall by about 30% (0.070/0.233).

So: **fp32 scales are worth building only if they cut the relative RMS weight
error by 30% or more.** Below that the improvement is real and unmeasurable,
which is the same condition Stage 1e-1 closed on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# gate_proj and up_proj are what QuantSpec quantises by default; down_proj stays
# fp16, so its scales do not exist and cannot be upgraded.
PROJECTIONS = ("gate_proj", "up_proj")
# Four layers spread across the depth, eight experts each. The first and last
# layers of an MoE stack routinely have different weight statistics from the
# middle, and a qualifier that samples only layer 0 is measuring a special case.
LAYERS = (0, 5, 10, 15)
EXPERTS = 8
DETECTABLE = 0.30


def _quantize(w: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    """Symmetric per-output-channel int8, dequantised back. `w` is (out, in).

    Deliberately a re-implementation rather than a call into
    `store.quantize_matrix`: the store has no fp32-scale mode, and adding one to
    find out whether it is worth adding is the build-before-qualifying order
    that Stage 1e-2 was written to stop repeating.
    """
    amax = w.abs().amax(dim=1, keepdim=True).float()
    scale = (amax / 127.0).clamp_min(torch.finfo(torch.float16).tiny)
    # The round trip through storage is the whole experiment. An fp32 scale is
    # stored exactly; an fp16 one is not, and the quantised integers are chosen
    # against the *stored* value, because that is what the dequantiser will use.
    scale = scale.to(scale_dtype).float()
    q = (w.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q.float() * scale


def _rel_rms(w: torch.Tensor, approx: torch.Tensor) -> float:
    w = w.float()
    return float(((approx - w) ** 2).mean().sqrt() / (w**2).mean().sqrt())


def main() -> int:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flashforge.models import DEFAULT_MODEL

    root = Path(snapshot_download(DEFAULT_MODEL, allow_patterns=["*.json"]))
    import json

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    print(f"{'layer.expert.proj':>26} {'fp32 scale':>11} {'fp16 scale':>11} {'fp16 cost':>10}")
    fp32_errors, fp16_errors = [], []
    handles: dict[str, object] = {}
    for layer in LAYERS:
        for expert in range(EXPERTS):
            for proj in PROJECTIONS:
                key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                shard = index[key]
                if shard not in handles:
                    path = snapshot_download(DEFAULT_MODEL, allow_patterns=[shard])
                    handles[shard] = safe_open(Path(path) / shard, framework="pt")
                w = handles[shard].get_tensor(key)

                e32 = _rel_rms(w, _quantize(w, torch.float32))
                e16 = _rel_rms(w, _quantize(w, torch.float16))
                fp32_errors.append(e32)
                fp16_errors.append(e16)
                if expert == 0:
                    print(f"{f'{layer}.{expert}.{proj}':>26} "
                          f"{100 * e32:>10.4f}% {100 * e16:>10.4f}% "
                          f"{100 * (e16 / e32 - 1):>9.2f}%")

    mean32 = sum(fp32_errors) / len(fp32_errors)
    mean16 = sum(fp16_errors) / len(fp16_errors)
    reduction = 1 - mean32 / mean16

    print("\n" + "=" * 78)
    print(f"over {len(fp16_errors)} real expert matrices "
          f"({len(LAYERS)} layers x {EXPERTS} experts x {len(PROJECTIONS)} projections)")
    print(f"  fp16 scales (shipping): {100 * mean16:.4f}% relative RMS weight error")
    print(f"  fp32 scales (proposed): {100 * mean32:.4f}%")
    print(f"  upgrading removes     : {100 * reduction:.2f}% of the error")
    print(f"  detectable needs      : {100 * DETECTABLE:.0f}%")
    print()
    if reduction >= DETECTABLE:
        print("BUILD IT. The upgrade clears the detection threshold, so the "
              "agreement re-measurement can distinguish its effect from zero.")
    else:
        print("DO NOT BUILD IT. The upgrade is real and unmeasurable: it moves "
              f"agreement by roughly {0.233 * reduction:.4f} points against a "
              "0.070-point standard error. The 0.233-point shortfall to the 99% "
              "bar is int8 rounding itself, not the precision of the scales.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
