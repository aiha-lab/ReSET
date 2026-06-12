"""Shared benchmark helpers (CUDA-event timer, common shapes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchResult:
    name: str
    shape: tuple[int, ...]
    median_ms: float
    p10_ms: float
    p90_ms: float


def cuda_time_ms(
    fn: Callable[[], None],
    *,
    warmup: int = 10,
    iters: int = 100,
) -> tuple[float, float, float]:
    """Time `fn` using CUDA events. Returns (median, p10, p90) in ms.

    NOTE: this measures end-to-end time between two CUDA events on the default
    stream, which means *host-side launch overhead* (Python, dispatcher,
    custom-op schema check, lazy imports inside ``fn``) is folded into the
    elapsed time whenever the host can't keep up with the GPU. For the FP4
    micro-bench this matters: ``flashinfer_scaled_fp4_mm`` is wrapped in a
    ``torch.library.custom_op`` plus a ``from flashinfer import mm_fp4`` lazy
    import inside the wrapper, so each eager call carries tens of us of host
    work that this function will (correctly) attribute to the leg.

    Use ``cuda_time_ms_graph`` below to measure pure GPU time (host overhead
    elided by CUDA-graph replay).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for s, e in zip(start, end):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()

    samples = sorted(s.elapsed_time(e) for s, e in zip(start, end))
    median = samples[len(samples) // 2]
    p10 = samples[max(0, len(samples) // 10)]
    p90 = samples[min(len(samples) - 1, (9 * len(samples)) // 10)]
    return median, p10, p90


def cuda_time_ms_graph(
    fn: Callable[[], None],
    *,
    warmup: int = 5,
    capture_iters: int = 10,
    replays: int = 20,
) -> tuple[float, float, float]:
    """Time `fn` by capturing it in a CUDA graph then replaying.

    The captured graph contains ``capture_iters`` invocations of ``fn`` (so
    the per-call cost is graph_time / capture_iters). Replaying a graph elides
    *all* host-side overhead (Python, custom-op dispatcher, schema check,
    lazy imports) -- only the GPU kernels actually launched are timed.

    Returns (median, p10, p90) per-call time in ms.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(capture_iters):
            fn()

    g.replay()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(replays):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        g.replay()
        ev_e.record()
        torch.cuda.synchronize()
        samples_ms.append(ev_s.elapsed_time(ev_e) / capture_iters)

    samples_ms.sort()
    median = samples_ms[len(samples_ms) // 2]
    p10 = samples_ms[max(0, len(samples_ms) // 10)]
    p90 = samples_ms[min(len(samples_ms) - 1, (9 * len(samples_ms)) // 10)]
    return median, p10, p90
