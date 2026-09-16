"""Locate the 26 ms/token cliff in `index_select`, with no model and no checkpoint.

    uv run python tools/gather_cliff.py

WHAT THIS IS EXPLAINING
-----------------------
Stage 1e-2's int8 capacity ladder is not a curve. Split each arm's token time
into the part spent blocked on fill and the part that is not:

    slots   ms/token   fill   non-fill
      200      130.5   64.4       66.1
      238      120.2   54.2       66.0
      270      145.8   53.4       92.4
      300      137.0   44.1       92.9
      330      133.3   40.7       92.6
      357      130.9   38.2       92.7

Non-fill time is flat at 66.0 ms, steps once, and is flat again at 92.6 ms. A
**step function of +26.5 ms**, not a pressure effect — pressure would grow with
the pool, and 357 slots pays exactly what 270 pays while holding 1.2 GB less
free VRAM. `tools/slot_cliff.py` then crossed capacity with sweep position and
confirmed the step is capacity: 238 vs 270 is -17.4% at p<0.0001, reproducible
to within 0.7% across three alternating visits, with zero allocator retries.

THE HYPOTHESIS
--------------
`ExpertCache` allocates its pool as `(capacity, row_numel)` of `row_dtype`, and
once the store is quantised `row_dtype` is **int8** — so the tensor's element
count equals its byte count. `gather` reads it with one `index_select`.

PyTorch picks a kernel for `index_select` by asking `canUse32BitIndexMath`,
which fails once a tensor exceeds `INT32_MAX` **elements**. Past that it
switches to 64-bit index arithmetic and loses the vectorised path with it. For
an int8 pool of 8,392,704-byte rows that boundary is:

    2,147,483,647 / 8,392,704 = 255.87  ->  capacity 256

which is exactly the interval the measured step falls in: 238 is below it, 270
is above it. The prediction is sharp enough to be wrong: the step must land
**between 255 and 256 slots**, and nowhere else.

WHY THIS TOOL HAS NO MODEL IN IT
--------------------------------
The claim is about one kernel on one tensor shape. A checkpoint, a router, and
an LRU policy are all confounds here, and each costs minutes per arm. Allocating
the pool directly and timing the `index_select` isolates the mechanism and puts
the boundary search — which needs many capacities — inside a two-minute budget
instead of an hour. Same move as `tools/cpu_channel.py` in Stage 1e-3.
"""
from __future__ import annotations

import torch

# Matches OLMoE-1B-7B under the shipping QuantSpec: gate_proj and up_proj int8,
# down_proj left fp16, per-channel scales packed on the end. 8,392,704 bytes.
HIDDEN, INTERMEDIATE = 2048, 1024
ROW_NUMEL = 2 * HIDDEN * INTERMEDIATE + 2 * HIDDEN * INTERMEDIATE + 2 * INTERMEDIATE
INT32_MAX = 2**31 - 1
BOUNDARY = INT32_MAX // ROW_NUMEL  # the last capacity that fits in 32-bit math

# Decode routes one token to top_k = 8 experts per layer, so a gather is eight
# rows. That is the operating point the 26 ms was measured at; a wider gather
# would amortise any per-call overhead and hide the thing being looked for.
GATHERED = 8
LAYERS = 16
REPEATS = 50
WARMUP = 5


def _time_gather(capacity: int, device: torch.device) -> tuple[float, float]:
    """Milliseconds per gather of `GATHERED` rows, and the achieved GB/s."""
    pool = torch.empty((capacity, ROW_NUMEL), dtype=torch.int8, device=device)
    # Spread across the pool rather than clustered: a gather of adjacent rows
    # would be a contiguous read and would not exercise the indexing path.
    idx = torch.linspace(0, capacity - 1, GATHERED).long().to(device)

    for _ in range(WARMUP):
        pool.index_select(0, idx)
    torch.cuda.synchronize()

    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(REPEATS):
        pool.index_select(0, idx)
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / REPEATS

    moved = 2 * GATHERED * ROW_NUMEL  # read the rows, write the copy
    del pool, idx
    torch.cuda.empty_cache()
    return ms, moved / (ms / 1000) / 1e9


def main() -> int:
    if not torch.cuda.is_available():
        print("This needs a CUDA device; the claim is about a CUDA kernel.")
        return 1
    device = torch.device("cuda")
    free, total = (x / (1 << 30) for x in torch.cuda.mem_get_info())
    print(f"{torch.cuda.get_device_name(0)}, {free:.2f} of {total:.2f} GB free, "
          f"torch {torch.__version__}")
    print(f"row = {ROW_NUMEL:,} bytes; int8 pool crosses INT32_MAX elements "
          f"above capacity {BOUNDARY}\n")

    # Tight around the predicted boundary, plus the two capacities the ladder
    # actually measured so the end-to-end numbers have something to attach to.
    plan = sorted({120, 238, BOUNDARY - 1, BOUNDARY, BOUNDARY + 1, BOUNDARY + 2, 270})
    print(f"{'slots':>6} {'pool GB':>8} {'elements':>16} {'32-bit':>7} "
          f"{'gather ms':>10} {'GB/s':>7} {'x16 layers':>11}")
    rows = []
    for capacity in plan:
        numel = capacity * ROW_NUMEL
        if numel / (1 << 30) > free - 0.6:
            print(f"{capacity:>6}  skipped — pool would not fit beside the harness")
            continue
        ms, gbs = _time_gather(capacity, device)
        rows.append((capacity, ms))
        print(f"{capacity:>6} {numel / (1 << 30):>8.3f} {numel:>16,} "
              f"{'yes' if numel <= INT32_MAX else 'NO':>7} "
              f"{ms:>10.3f} {gbs:>7.1f} {ms * LAYERS:>10.1f}ms")

    below = [ms for cap, ms in rows if cap * ROW_NUMEL <= INT32_MAX]
    above = [ms for cap, ms in rows if cap * ROW_NUMEL > INT32_MAX]
    print("\n" + "=" * 84)
    if not below or not above:
        print("Could not bracket the boundary on this card's free VRAM.")
        return 1

    step = (max(above) + min(above)) / 2 - (max(below) + min(below)) / 2
    print(f"  32-bit indexable: {min(below):.3f}-{max(below):.3f} ms per gather")
    print(f"  64-bit indexable: {min(above):.3f}-{max(above):.3f} ms per gather")
    print(f"  step at capacity {BOUNDARY} -> {BOUNDARY + 1}: "
          f"{step:+.3f} ms per gather, {step * LAYERS:+.1f} ms per token "
          f"across {LAYERS} layers")
    print(f"  measured end-to-end, 238 -> 270 slots:  +25.7 ms per token")
    print()
    if abs(step * LAYERS - 25.7) < 8.0:
        print("CONFIRMED. The pool crosses INT32_MAX elements, `index_select` "
              "drops to 64-bit index math, and the cost of that is the whole "
              "26 ms. It is a tensor-shape boundary, not a capacity effect — "
              f"so the usable ceiling is {BOUNDARY} int8 slots, and splitting "
              "the pool in two would move it.")
    else:
        print("NOT CONFIRMED by this margin. The boundary is real and in the "
              "right place, but it does not account for the measured 26 ms on "
              "its own — something else is riding with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
