"""Stage 1e-3, step 1: is the CPU a second channel, or the same channel twice?

The CPU path is sold as free parallelism. On a cache miss you can ship the
expert across PCIe, or you can run it where it already lives. Q7 says a single
expert at m=1 costs 1.11 ms on the CPU against 1.212 ms on the link, so the two
are near enough to parity that neither replaces the other -- the win, if there
is one, comes from running them *at the same time* and splitting the missed
experts between them.

That argument has an assumption underneath it, and the assumption is the whole
stage:

    the DMA engine reads expert weights out of host DRAM
    the CPU cores read expert weights out of host DRAM

They are only independent channels if host memory bandwidth can feed both at
once. If it cannot, "parallel" means two processes taking turns on one wire and
the combined throughput is the same number it always was. Nothing in this repo
has ever measured that, and Q7 could not have: it timed each path alone.

So this script measures the overlap efficiency

    eta = (link_rate_together + cpu_rate_together)
          / (link_rate_alone   + cpu_rate_alone)

eta = 1.0 means two real channels and the ceiling arithmetic holds. eta = 0.5
means one channel wearing a hat, and Stage 1e-3 closes negative here for the
price of two minutes instead of two days of integration.

PRE-COMMITTED, before running it
--------------------------------
At 238 int8 slots the runtime misses ~58 experts per decode token and the link
clears them in 54.2 ms, so 0.93 ms/expert. Against a CPU expert at c ms, an
optimal split of n misses across both channels takes n / (1/0.93 + 1/c) ms.
At Q7's c = 1.11 that is 29.3 ms, saving 25 ms off a 120 ms token: +26%.

Build the runtime path only if the gain predicted at the *measured* eta and the
*measured* c clears +15%, which is about twice the harness's pass-to-pass
spread. Below that it is another 1e-1: real, and smaller than this project can
measure.

No checkpoint, no model download, no runtime changes. Shapes are OLMoE's.
"""

from __future__ import annotations

import time

import torch

HIDDEN = 2048
INTERMEDIATE = 1024
EXPERTS = 64          # enough distinct weight to defeat any L3 -- see _cpu_rate
TARGET_SECONDS = 2.0  # how long each channel should run, alone and contended
THREAD_SWEEP = (4, 8, 16)


def _expert_bytes(dtype_size: int) -> int:
    return 3 * HIDDEN * INTERMEDIATE * dtype_size


def _median(samples: list[float]) -> float:
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# -- the two channels -------------------------------------------------------
#
# Both are written as "do `count` experts' worth of work" so the contended run
# can size each side to finish at roughly the same moment. A channel that
# finishes early would spend its tail unloaded and report a rate that is part
# contended and part not, which is exactly the confound this is trying to find.


class Link:
    """Host-to-device copies of whole expert rows, pinned, on a side stream."""

    def __init__(self, device: torch.device) -> None:
        self.row_numel = 3 * HIDDEN * INTERMEDIATE
        self.host = torch.empty((EXPERTS, self.row_numel), dtype=torch.float16).pin_memory()
        self.host.normal_()
        # A small ring of destinations: the runtime writes into a slot pool it
        # reuses, and a fresh allocation per copy would time the allocator.
        self.dev = torch.empty((4, self.row_numel), dtype=torch.float16, device=device)
        self.stream = torch.cuda.Stream(device=device)
        self.bytes_per_expert = _expert_bytes(2)

    def run(self, count: int) -> None:
        with torch.cuda.stream(self.stream):
            for i in range(count):
                self.dev[i & 3].copy_(self.host[i % EXPERTS], non_blocking=True)

    def wait(self) -> None:
        self.stream.synchronize()


class CpuExperts:
    """One SwiGLU expert at m=1, the decode case, over a rotating weight set.

    Rotating matters. A single expert's fp32 weights are 25 MB and would sit in
    L3 on a machine with a large enough cache, which would time a cached GEMM
    rather than the memory-bound one the runtime would actually run. Cycling
    EXPERTS of them guarantees every call reaches DRAM, which is also the whole
    point -- DRAM is the resource this script suspects is contended.
    """

    def __init__(self, quantised: bool) -> None:
        self.quantised = quantised
        if quantised:
            # int8 weights plus fp32 per-channel scales, matching what the store
            # holds after Stage 1e-2. Dequantised per call, because that is what
            # a CPU kernel without an int8 GEMM would have to do.
            self.w = torch.randint(
                -127, 128, (EXPERTS, 3, INTERMEDIATE, HIDDEN), dtype=torch.int8
            )
            self.scale = torch.rand(EXPERTS, 3, INTERMEDIATE, 1, dtype=torch.float32) * 0.01
            self.bytes_per_expert = 3 * HIDDEN * INTERMEDIATE
        else:
            self.w = torch.randn(EXPERTS, 3, INTERMEDIATE, HIDDEN, dtype=torch.float32) * 0.02
            self.scale = None
            self.bytes_per_expert = _expert_bytes(4)
        self.x = torch.randn(1, HIDDEN, dtype=torch.float32)

    def _weights(self, e: int):
        w = self.w[e]
        if not self.quantised:
            return w[0], w[1], w[2]
        wide = w.to(torch.float32).mul_(self.scale[e])
        return wide[0], wide[1], wide[2]

    def run(self, count: int) -> None:
        with torch.inference_mode():
            for i in range(count):
                gate, up, down = self._weights(i % EXPERTS)
                # All three are stored (intermediate, hidden). gate and up are
                # used transposed, down is used as-is -- which is the same FLOP
                # count and the same bytes as the real expert, laid out so one
                # stacked tensor holds all three.
                h = torch.nn.functional.silu(self.x @ gate.T) * (self.x @ up.T)
                h @ down

    def wait(self) -> None:
        pass


# -- measurement ------------------------------------------------------------


def _rate_alone(channel, *, repeats: int = 3) -> tuple[float, int]:
    """Experts per second, and a count sized to run for about TARGET_SECONDS."""
    channel.run(4)
    channel.wait()

    probe = 8
    while True:
        start = time.perf_counter()
        channel.run(probe)
        channel.wait()
        elapsed = time.perf_counter() - start
        if elapsed > 0.15 or probe > 4096:
            break
        probe *= 4

    count = max(8, int(probe * TARGET_SECONDS / elapsed))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        channel.run(count)
        channel.wait()
        samples.append(count / (time.perf_counter() - start))
    return _median(samples), count


def _rate_together(link: Link, cpu: CpuExperts, link_count: int, cpu_count: int):
    """Both channels at once. The copies are enqueued first so the DMA is
    already in flight before the CPU loop starts competing for DRAM."""
    start = time.perf_counter()
    link.run(link_count)          # async: returns as soon as it is enqueued
    cpu.run(cpu_count)
    cpu_done = time.perf_counter()
    link.wait()
    link_done = time.perf_counter()
    return link_count / (link_done - start), cpu_count / (cpu_done - start)


def _report(label, link_alone, cpu_alone, link_both, cpu_both, link_gbs, cpu_gbs):
    eta = (link_both + cpu_both) / (link_alone + cpu_alone)
    print(f"  {label}")
    print(f"    link  {link_alone:8.1f} -> {link_both:8.1f} experts/s "
          f"({100 * link_both / link_alone:5.1f}%)  {link_gbs:5.2f} GB/s alone")
    print(f"    cpu   {cpu_alone:8.1f} -> {cpu_both:8.1f} experts/s "
          f"({100 * cpu_both / cpu_alone:5.1f}%)  {cpu_gbs:5.2f} GB/s alone")
    print(f"    eta   {eta:.3f}")
    return eta


def _predicted_gain(link_ms: float, cpu_ms: float) -> float:
    """Percent decode gain from splitting 58 missed experts across both.

    link_ms/cpu_ms are per-expert costs *under contention*. The 54.2 ms and
    120 ms are the measured int8 operating point at 238 slots.
    """
    misses, token_ms, fill_ms = 58.0, 120.0, 54.2
    if cpu_ms <= 0:
        return 0.0
    split_ms = misses / (1.0 / link_ms + 1.0 / cpu_ms)
    return 100.0 * (token_ms / (token_ms - (fill_ms - split_ms)) - 1.0)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    device = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(device)}, {torch.get_num_threads()} default threads\n")

    link = Link(device)
    link_alone, link_count = _rate_alone(link)
    link_gbs = link_alone * link.bytes_per_expert / 1e9
    print(f"link alone: {link_alone:.1f} experts/s, {link_gbs:.2f} GB/s, "
          f"{1000 / link_alone:.3f} ms/expert\n")

    best = None
    for quantised in (False, True):
        kind = "int8+dequant" if quantised else "fp32"
        print(f"{kind} CPU experts")
        cpu = CpuExperts(quantised)
        for threads in THREAD_SWEEP:
            torch.set_num_threads(threads)
            cpu_alone, cpu_count = _rate_alone(cpu)
            cpu_gbs = cpu_alone * cpu.bytes_per_expert / 1e9
            link_both, cpu_both = _rate_together(link, cpu, link_count, cpu_count)
            eta = _report(f"{threads:2d} threads", link_alone, cpu_alone,
                          link_both, cpu_both, link_gbs, cpu_gbs)
            gain = _predicted_gain(1000 / link_both, 1000 / cpu_both)
            print(f"    gain  {gain:+.1f}% decode at this operating point\n")
            if best is None or gain > best[0]:
                best = (gain, kind, threads, eta)
        del cpu

    gain, kind, threads, eta = best
    print("=" * 62)
    print(f"best: {kind}, {threads} threads, eta {eta:.3f}, predicted {gain:+.1f}%")
    print(f"pre-committed bar: +15%  ->  {'BUILD' if gain >= 15 else 'CLOSE NEGATIVE'}")


if __name__ == "__main__":
    main()
