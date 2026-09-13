"""Q7 and Q8 — the hardware constants every scheduling decision rests on.

The five original questions are all about *routing behaviour*: they measure the
model. These two measure the machine, and neither needs a trace or a model
download.

  Q7 cost model   what does one expert cost on CPU, on GPU, and over PCIe?
                  -> the break-even token count that decides placement
  Q8 storage      how fast can the disk hand you an expert, and at what queue
                  depth? -> how many layers of lookahead a disk tier needs

Why they matter, in one paragraph each.

Q7. A hybrid CPU/GPU scheduler chooses, per expert, between shipping it across
PCIe and running it in place on the CPU. GPU cost is roughly flat in token
count (transfer dominates); CPU cost is linear, `t_c = beta * m + C`. Those two
lines cross at some m*, and m* is the single number the whole placement policy
turns on. It is a property of *your* CPU and *your* bus, not of the model, so it
has to be measured rather than assumed.

Q8. Hiding a transfer requires lead time. One layer of compute hides one PCIe
transfer, which is why cross-layer prefetch with a one-layer horizon works. A
disk read is several times slower, so it needs a proportionally deeper horizon
-- and prediction accuracy decays with depth. Measuring `t_disk / t_pcie` turns
"can we stream from disk?" into an arithmetic question that Q3 can answer.

One structural warning that Q8 exists to surface: PCIe saturates at a queue
depth of one, which is why serial-I/O cost models work for it. NVMe does not.
A single outstanding read leaves an SSD mostly idle, so a disk tier wants
batched, deep-queue requests -- a different scheduling shape, not the same one
scaled down.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Above this, you are certainly timing the OS page cache rather than the device:
# even PCIe 5.0 x4 NVMe tops out around 14 GB/s. This is only a backstop — a
# cached read can easily land *below* it and look like a plausible NVMe number,
# which is why the free-RAM check below is the real guard.
CACHE_SUSPICION_GBPS = 16.0

DEFAULT_TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_READ_SIZES = (64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20)
DEFAULT_QUEUE_DEPTHS = (1, 2, 4, 8, 16, 32)


# ==========================================================================
# Q7 — cost model calibration
# ==========================================================================

@dataclass
class LinearCost:
    """A fitted `t = beta * m + const` in milliseconds."""

    beta_ms_per_token: float
    const_ms: float
    r_squared: float

    def predict(self, tokens: float) -> float:
        return self.beta_ms_per_token * tokens + self.const_ms


def build_expert(hidden_size: int, intermediate_size: int, *, dtype, device):
    """One SwiGLU expert — the gate/up/down triple every dev-ladder model uses.

    Built from raw Linear layers rather than pulled off a checkpoint on
    purpose: the cost we want is a function of shape and hardware only, and
    this way Q7 runs with no model download.
    """
    import torch
    from torch import nn

    class Expert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.act = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    expert = Expert().to(device=device, dtype=dtype)
    expert.eval()
    for param in expert.parameters():
        param.requires_grad_(False)
    return expert


def expert_bytes(hidden_size: int, intermediate_size: int, bytes_per_param: float = 2.0) -> float:
    """Weight bytes for one SwiGLU expert."""
    return 3.0 * hidden_size * intermediate_size * bytes_per_param


def _time_call(fn, *, repeats: int, warmup: int) -> float:
    """Median wall-clock milliseconds over `repeats` calls.

    Median, not mean: on a desktop OS a background process will occasionally
    steal a whole scheduling quantum, and one such outlier moves a mean far
    more than it moves the thing we are trying to measure.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(samples))


def cpu_expert_curve(
    hidden_size: int,
    intermediate_size: int,
    *,
    token_counts: tuple[int, ...] = DEFAULT_TOKEN_COUNTS,
    dtype: str = "float32",
    threads: int | None = None,
    repeats: int = 9,
    warmup: int = 3,
) -> pd.DataFrame:
    """Time one expert on the CPU across token counts.

    float32 by default because CPU float16 GEMM is emulated in most builds and
    would measure the emulation, not the hardware. A production CPU path would
    use a quantised kernel and beat this, so treat the fitted beta as an upper
    bound on CPU cost -- which makes any "CPU is worth it" conclusion drawn
    from it conservative.
    """
    import torch

    if threads is not None:
        torch.set_num_threads(threads)
    used_threads = torch.get_num_threads()

    torch_dtype = getattr(torch, dtype)
    expert = build_expert(hidden_size, intermediate_size, dtype=torch_dtype, device="cpu")

    rows = []
    with torch.inference_mode():
        for tokens in token_counts:
            batch = torch.randn(tokens, hidden_size, dtype=torch_dtype)
            ms = _time_call(lambda: expert(batch), repeats=repeats, warmup=warmup)
            rows.append(
                {
                    "tokens": int(tokens),
                    "ms": ms,
                    "ms_per_token": ms / tokens,
                    "threads": used_threads,
                    "dtype": dtype,
                }
            )
            log.debug("CPU expert, m=%d: %.3f ms", tokens, ms)
    return pd.DataFrame(rows)


def fit_linear_cost(curve: pd.DataFrame) -> LinearCost:
    """Least-squares fit of `ms = beta * tokens + const`.

    r_squared is reported because the linear model is an assumption, not a
    law. If it comes back low, the CPU path has a regime change inside the
    token range -- cache spill is the usual cause -- and a single beta is the
    wrong abstraction for the scheduler.
    """
    x = curve["tokens"].to_numpy(np.float64)
    y = curve["ms"].to_numpy(np.float64)
    if x.size < 2:
        raise ValueError("Need at least two token counts to fit a line.")

    design = np.column_stack([x, np.ones_like(x)])
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - (slope * x + intercept)
    total = y - y.mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total) if total.any() else 1.0
    return LinearCost(float(slope), float(intercept), r2)


def gpu_expert_cost(
    hidden_size: int,
    intermediate_size: int,
    *,
    token_counts: tuple[int, ...] = DEFAULT_TOKEN_COUNTS,
    dtype: str = "float16",
    repeats: int = 20,
    warmup: int = 5,
) -> pd.DataFrame:
    """Time the same expert on the GPU. Expected to be near-flat in token count."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")

    torch_dtype = getattr(torch, dtype)
    expert = build_expert(hidden_size, intermediate_size, dtype=torch_dtype, device="cuda")

    rows = []
    with torch.inference_mode():
        for tokens in token_counts:
            batch = torch.randn(tokens, hidden_size, dtype=torch_dtype, device="cuda")

            def run() -> None:
                expert(batch)
                torch.cuda.synchronize()

            ms = _time_call(run, repeats=repeats, warmup=warmup)
            rows.append({"tokens": int(tokens), "ms": ms, "dtype": dtype})
    return pd.DataFrame(rows)


def pcie_transfer_cost(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype: str = "float16",
    repeats: int = 15,
    warmup: int = 3,
) -> pd.DataFrame:
    """Host-to-device transfer time for one expert, three ways.

    The three modes are the ladder a transfer engine climbs:

      pageable      what you get by default. The driver has to stage through
                    its own bounce buffer, so this is the floor.
      pinned        page-locked host memory, async copy. Usually a large win.
      pinned_split  gate/up/down issued concurrently on three CUDA streams --
                    the fine-grained scheme LayerScope's AsyncIO uses to keep
                    the bus busy through the gaps a single large copy leaves.

    Achieved bandwidth matters as much as the time: if `pinned` is already
    near the bus ceiling, splitting buys nothing and the extra streams are
    complexity for its own sake.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")

    torch_dtype = getattr(torch, dtype)
    itemsize = torch.empty(0, dtype=torch_dtype).element_size()
    total_bytes = expert_bytes(hidden_size, intermediate_size, itemsize)

    # The real three tensors, so the split case has honest shapes and the
    # whole-expert case moves exactly the same bytes.
    shapes = [
        (intermediate_size, hidden_size),  # gate_proj
        (intermediate_size, hidden_size),  # up_proj
        (hidden_size, intermediate_size),  # down_proj
    ]

    def make_host(pinned: bool):
        return [
            torch.empty(shape, dtype=torch_dtype, pin_memory=pinned) for shape in shapes
        ]

    def make_device():
        return [torch.empty(shape, dtype=torch_dtype, device="cuda") for shape in shapes]

    device_tensors = make_device()
    rows = []

    def record(mode: str, ms: float) -> None:
        rows.append(
            {
                "mode": mode,
                "ms": ms,
                "bytes": total_bytes,
                "gbps": total_bytes / (ms / 1e3) / 1e9,
                "dtype": dtype,
            }
        )

    for mode, pinned in (("pageable", False), ("pinned", True)):
        host = make_host(pinned)

        def run() -> None:
            for dst, src in zip(device_tensors, host):
                dst.copy_(src, non_blocking=pinned)
            torch.cuda.synchronize()

        record(mode, _time_call(run, repeats=repeats, warmup=warmup))

    # Three streams, one per sub-tensor.
    host = make_host(True)
    streams = [torch.cuda.Stream() for _ in shapes]

    def run_split() -> None:
        torch.cuda.synchronize()
        for stream, dst, src in zip(streams, device_tensors, host):
            with torch.cuda.stream(stream):
                dst.copy_(src, non_blocking=True)
        for stream in streams:
            stream.synchronize()

    record("pinned_split3", _time_call(run_split, repeats=repeats, warmup=warmup))
    return pd.DataFrame(rows)


def break_even_tokens(fit: LinearCost, gpu_path_ms: float) -> float:
    """Token count at which shipping an expert to the GPU starts to win.

    `gpu_path_ms` is the whole GPU-side cost of one expert: transfer plus
    compute. Below the returned m*, the expert is cold enough that running it
    in place on the CPU is cheaper *and* leaves its PCIe slot free for a
    prefetch -- which is the move that makes cross-layer scheduling pay.

    Returns 0.0 when the GPU path is cheaper even for a single token (the
    usual outcome on a fast bus with a slow CPU), and inf when the CPU wins
    across the board.
    """
    if fit.beta_ms_per_token <= 0:
        return math.inf
    crossing = (gpu_path_ms - fit.const_ms) / fit.beta_ms_per_token
    return float(max(0.0, crossing))


# ==========================================================================
# Q8 — storage read curve
# ==========================================================================

def total_ram_bytes() -> int | None:
    """Physical RAM, or None where we cannot determine it without a dependency.

    Used only to warn when the scratch file is small enough to sit entirely in
    the page cache. Getting this wrong is the single easiest way to produce a
    confident, wrong Q8 — a cached read reports DRAM bandwidth, which on a
    laptop lands squarely inside the plausible range for a good NVMe drive.
    """
    try:  # Windows
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except (AttributeError, OSError, ValueError):
        pass

    try:  # Linux / macOS
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def warn_if_cacheable(file_size_bytes: int) -> None:
    """Warn when the scratch file is small enough to be served from RAM."""
    ram = total_ram_bytes()
    if ram is None:
        log.warning(
            "Could not determine physical RAM. Make sure --file-size-gb exceeds it, "
            "or Q8 will measure the page cache instead of the device."
        )
        return
    if file_size_bytes < ram:
        log.warning(
            "Scratch file is %.1f GiB but this machine has %.1f GiB of RAM. The OS will "
            "cache the whole file, so these numbers are DRAM bandwidth, not disk. They "
            "are fine for checking the harness and useless for design — pass "
            "--file-size-gb %.0f or more for numbers you intend to trust.",
            file_size_bytes / (1 << 30), ram / (1 << 30), (ram / (1 << 30)) * 1.5,
        )


def ensure_scratch_file(path: Path, size_bytes: int, *, chunk: int = 32 << 20) -> Path:
    """Create (or reuse) a scratch file of at least `size_bytes`.

    Content is pseudo-random rather than zeros: filesystems with compression
    or sparse-file support will happily make a zero-filled file cost nothing
    to read, which would turn the whole measurement into a fiction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= size_bytes:
        log.info("Reusing scratch file %s (%.1f GiB)", path, path.stat().st_size / (1 << 30))
        return path

    log.info("Creating scratch file %s (%.1f GiB) ...", path, size_bytes / (1 << 30))
    rng = np.random.default_rng(0)
    written = 0
    last_logged = 0
    with open(path, "wb") as handle:
        while written < size_bytes:
            block = min(chunk, size_bytes - written)
            # rng.bytes, not rng.integers(...).tobytes(): the file has to exceed
            # RAM to be useful, so this loop runs tens of gigabytes and the
            # integer path spends most of that time in the generator rather
            # than the disk.
            handle.write(rng.bytes(block))
            written += block
            if written - last_logged >= (4 << 30):
                log.info("  ... %.0f/%.0f GiB", written / (1 << 30), size_bytes / (1 << 30))
                last_logged = written
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _read_batch(path: Path, offsets: list[int], read_size: int, latencies: list[float]) -> None:
    """Issue a thread's share of the reads. One handle per thread, reused."""
    buffer = memoryview(bytearray(read_size))
    local: list[float] = []
    with open(path, "rb", buffering=0) as handle:
        for offset in offsets:
            start = time.perf_counter()
            handle.seek(offset)
            filled = 0
            while filled < read_size:
                got = handle.readinto(buffer[filled:])
                if not got:
                    break
                filled += got
            local.append((time.perf_counter() - start) * 1e3)
    latencies.extend(local)


def storage_read_curve(
    path: str | Path,
    *,
    file_size_bytes: int = 2 << 30,
    read_sizes: tuple[int, ...] = DEFAULT_READ_SIZES,
    queue_depths: tuple[int, ...] = DEFAULT_QUEUE_DEPTHS,
    target_bytes_per_point: int = 192 << 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Random-offset read bandwidth and latency vs request size and queue depth.

    Queue depth is emulated with threads: `os.pread` is Unix-only, so each
    worker keeps its own handle and does seek+readinto, which releases the GIL
    for the duration of the read.

    The number that matters most here is not peak bandwidth -- it is the queue
    depth at which you *reach* peak. That depth is the minimum number of expert
    reads a disk-tier prefetcher has to have in flight before the device is
    being used properly, and it sets the minimum useful lookahead.

    Read the page-cache warning seriously. On a machine with plenty of free
    RAM a 2 GiB scratch file will sit entirely in cache and report DRAM
    bandwidth. Push `file_size_bytes` past free RAM for numbers you intend to
    design against.
    """
    warn_if_cacheable(file_size_bytes)
    path = ensure_scratch_file(Path(path), file_size_bytes)
    actual_size = path.stat().st_size
    rng = random.Random(seed)

    rows = []
    for read_size in read_sizes:
        if read_size > actual_size:
            log.warning("Skipping %.1f MiB reads: larger than the scratch file",
                        read_size / (1 << 20))
            continue
        span = actual_size - read_size

        for depth in queue_depths:
            n_reads = max(depth * 4, target_bytes_per_point // read_size)
            n_reads = int(math.ceil(n_reads / depth) * depth)
            offsets = [rng.randrange(0, span + 1) for _ in range(n_reads)]

            shards = [offsets[i::depth] for i in range(depth)]
            # One bucket per thread, joined before it is read: no shared
            # mutable state, so no lock.
            collected: list[list[float]] = [[] for _ in range(depth)]

            threads = [
                threading.Thread(target=_read_batch, args=(path, shard, read_size, bucket))
                for shard, bucket in zip(shards, collected)
            ]
            start = time.perf_counter()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            wall = time.perf_counter() - start

            moved = n_reads * read_size
            samples = np.asarray([v for bucket in collected for v in bucket], dtype=np.float64)
            rows.append(
                {
                    "read_bytes": int(read_size),
                    "read_mib": read_size / (1 << 20),
                    "queue_depth": int(depth),
                    "n_reads": n_reads,
                    "gbps": moved / wall / 1e9,
                    "mean_ms": float(samples.mean()),
                    "p50_ms": float(np.percentile(samples, 50)),
                    "p99_ms": float(np.percentile(samples, 99)),
                }
            )
            log.debug("read %.2f MiB qd=%d: %.2f GB/s", read_size / (1 << 20), depth, rows[-1]["gbps"])

    frame = pd.DataFrame(rows)
    peak = frame["gbps"].max() if not frame.empty else 0.0
    if peak > CACHE_SUSPICION_GBPS:
        log.warning(
            "Peak read bandwidth %.1f GB/s exceeds any consumer NVMe device. You are "
            "almost certainly measuring the OS page cache -- rerun with "
            "--file-size-gb larger than your free RAM.", peak,
        )
    return frame


def saturation_knee(curve: pd.DataFrame, *, fraction: float = 0.90) -> pd.DataFrame:
    """Smallest queue depth reaching `fraction` of peak bandwidth, per read size.

    This is the headline of Q8. A knee at depth 1 means the device behaves like
    PCIe and a serial cost model is fine. A knee at 8 or 16 means a disk-tier
    prefetcher must batch its requests, and a one-expert-at-a-time scheduler
    will leave most of the device idle no matter how good its predictions are.
    """
    rows = []
    for read_size, group in curve.groupby("read_bytes"):
        ordered = group.sort_values("queue_depth")
        peak = ordered["gbps"].max()
        reached = ordered[ordered["gbps"] >= fraction * peak]
        knee = int(reached["queue_depth"].iloc[0]) if not reached.empty else int(
            ordered["queue_depth"].iloc[-1]
        )
        best = ordered.loc[ordered["gbps"].idxmax()]
        rows.append(
            {
                "read_bytes": int(read_size),
                "read_mib": read_size / (1 << 20),
                "peak_gbps": float(peak),
                "knee_queue_depth": knee,
                "latency_at_peak_ms": float(best["mean_ms"]),
                "qd1_gbps": float(ordered["gbps"].iloc[0]),
                "qd1_penalty": float(peak / ordered["gbps"].iloc[0]) if ordered["gbps"].iloc[0] else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("read_bytes").reset_index(drop=True)


def disk_time_for_expert(curve: pd.DataFrame, nbytes: float) -> tuple[float, dict]:
    """Estimated time to pull one expert off disk, and the row it came from.

    Picks the measured read size closest to the expert's actual size at that
    size's best queue depth, then scales by the size ratio. Interpolating a
    latency curve is not free of sin, but the alternative -- assuming peak
    sequential bandwidth applies to a 12 MB random read -- is worse.
    """
    if curve.empty:
        raise ValueError("Empty storage curve.")
    sizes = curve["read_bytes"].unique()
    nearest = int(min(sizes, key=lambda s: abs(math.log(s) - math.log(max(nbytes, 1)))))
    group = curve[curve["read_bytes"] == nearest]
    best = group.loc[group["gbps"].idxmax()]
    scaled_ms = float(best["mean_ms"]) * (nbytes / nearest)
    return scaled_ms, {
        "measured_read_bytes": nearest,
        "measured_mean_ms": float(best["mean_ms"]),
        "queue_depth": int(best["queue_depth"]),
        "gbps": float(best["gbps"]),
    }


def required_lookahead(transfer_ms: float, layer_time_ms: float) -> int:
    """How many layers of compute it takes to hide a transfer of `transfer_ms`.

    Feed this straight into Q3: run the predictor sweep out to at least this
    many layers of lookahead and see whether accuracy survives. If it does not,
    the disk tier cannot be hidden behind per-layer prediction and needs a
    longer-horizon signal -- a draft pass, or domain-level cache warming.
    """
    if layer_time_ms <= 0:
        return 0
    return int(math.ceil(transfer_ms / layer_time_ms))


# ==========================================================================
# report assembly
# ==========================================================================

@dataclass
class HardwareReport:
    """Everything ff-analyze needs from a bench run, in one serialisable blob."""

    hidden_size: int
    intermediate_size: int
    expert_bytes: float
    cpu_beta_ms_per_token: float | None = None
    cpu_const_ms: float | None = None
    cpu_r_squared: float | None = None
    cpu_threads: int | None = None
    gpu_expert_ms: float | None = None
    pcie_best_ms: float | None = None
    pcie_best_mode: str | None = None
    pcie_gbps: float | None = None
    break_even_tokens: float | None = None
    disk_expert_ms: float | None = None
    disk_peak_gbps: float | None = None
    disk_knee_queue_depth: int | None = None
    # False when the scratch file fit in RAM, i.e. the disk numbers above are
    # page-cache bandwidth. Recorded rather than inferred later, because a
    # cached read looks entirely plausible once it is sitting in a JSON file.
    disk_measurement_trusted: bool | None = None
    disk_to_pcie_ratio: float | None = None
    layer_time_ms: float | None = None
    required_lookahead: int | None = None

    def to_json(self, path: Path) -> None:
        import json

        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @staticmethod
    def from_json(path: Path) -> "HardwareReport":
        import json

        # utf-8-sig for the same reason ff-analyze uses it: a BOM from a
        # Windows shell would otherwise make a valid file unreadable.
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            return HardwareReport(**json.load(handle))
