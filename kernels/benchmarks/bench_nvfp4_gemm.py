"""Latency comparison: nvfp4r.gemm vs vLLM cutlass_scaled_fp4_mm vs BF16 matmul.

Goal: confirm our kernel is at parity with (or faster than) the
production NVFP4 GEMM that vLLM ships. We use ``vllm.scaled_fp4_quant`` to
produce activations + weights in the standard 128x4-swizzled scale layout,
which is exactly what ``nvfp4r.gemm`` expects internally, so both ops run
on the same byte-level inputs.

The BF16 ``torch.matmul`` row is provided as a sanity reference; FP4 is
expected to be markedly faster on B200 because the tensor cores chew through
half the data and use the extra FLOPs/cycle.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Iterable

import torch

import nvfp4r  # noqa: F401  (registers torch.ops.nvfp4r.*)
import vllm._custom_ops as vops


_GEMM_SHAPES: list[tuple[int, int, int]] = [
    # ----- specialized K (legacy leaderboard) -----
    (128, 4096, 2048),   # K=2048 specialized
    (256, 4096, 2048),
    (128, 4096, 7168),   # K=7168 specialized
    (256, 4096, 7168),

    # ----- our target Qwen prefill shapes (K not in special table) -----
    # DR-7B / Qwen2.5-7B (hidden=3584, inter=18944)
    (128, 4608,  3584),  # qkv_proj
    (128, 3584,  3584),  # o_proj
    (128, 18944, 3584),  # gate/up_proj (each)
    (128, 3584, 18944),  # down_proj
    # Qwen3-8B (hidden=4096, inter=12288)
    (128, 6144,  4096),  # qkv_proj
    (128, 4096,  4096),  # o_proj
    (128, 12288, 4096),  # gate/up_proj (each)
    (128, 4096, 12288),  # down_proj
]


def cuda_event_time_us(fn, n_iters: int = 50, n_warmup: int = 5) -> float:
    """Time ``fn`` using CUDA events. Returns mean microseconds per call."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.mean(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))


def bench_one(M: int, N: int, K: int, dtype: torch.dtype, dev: torch.device) -> dict:
    A_bf = torch.randn(M, K, dtype=dtype, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=dtype, device=dev) * 0.05

    # Per-tensor amax used as the global scale; vLLM packs FP8 block scales as
    # ``round(true_block_amax / global_scale)`` which keeps the FP4 dynamic
    # range usable without saturating the FP8 E4M3 holder.
    one = torch.ones((), dtype=torch.float32, device=A_bf.device)

    A_fp4, A_scales = vops.scaled_fp4_quant(A_bf, one)
    B_fp4, B_scales = vops.scaled_fp4_quant(B_bf, one)
    alpha = one

    # vLLM's FP4 mm: produces a [M, N] BF16/FP16 tensor.
    def run_vllm_fp4():
        return vops.cutlass_scaled_fp4_mm(A_fp4, B_fp4, A_scales, B_scales,
                                          alpha, out_dtype=dtype)

    # nvfp4r.gemm: in-place into a pre-allocated FP16 buffer.
    C_buf = torch.empty(M, N, dtype=torch.float16, device=dev)

    def run_nvfp4r_gemm():
        return nvfp4r.gemm(A_fp4, B_fp4, A_scales, B_scales, C_buf)

    # BF16 dense reference (pure compute baseline).
    def run_bf16():
        return torch.matmul(A_bf, B_bf.t())

    return {
        "shape": (M, N, K),
        "vllm_fp4_us":  cuda_event_time_us(run_vllm_fp4),
        "nvfp4r_us":    cuda_event_time_us(run_nvfp4r_gemm),
        "bf16_us":      cuda_event_time_us(run_bf16),
    }


def main(shapes: Iterable[tuple[int, int, int]], dtype: torch.dtype) -> None:
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(dev)}")
    print(f"dtype (BF16 reference): {dtype}")
    print()
    print(f"{'M':>5} {'N':>6} {'K':>6} | "
          f"{'nvfp4r [us]':>13} {'vllm fp4 [us]':>15} {'bf16 [us]':>11} | "
          f"{'nvfp4r/vllm':>13} {'nvfp4r/bf16':>13}")
    print("-" * 120)
    for M, N, K in shapes:
        try:
            r = bench_one(M, N, K, dtype, dev)
        except Exception as e:
            print(f"{M:>5} {N:>6} {K:>6} | ERROR: {e}")
            continue
        nvr = r["nvfp4r_us"]
        vll = r["vllm_fp4_us"]
        bf = r["bf16_us"]
        print(
            f"{M:>5} {N:>6} {K:>6} | "
            f"{nvr:>13.2f} {vll:>15.2f} {bf:>11.2f} | "
            f"{nvr / vll:>13.2f} {nvr / bf:>13.2f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    main(_GEMM_SHAPES, dt)
