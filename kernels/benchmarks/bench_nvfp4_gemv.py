"""Latency comparison: nvfp4r.gemv vs vLLM cutlass_scaled_fp4_mm @ M=1.

vLLM does NOT ship a dedicated NVFP4 GEMV path — `apply_nvfp4_linear`
forwards every shape (including M==1 decode) into `cutlass_scaled_fp4_mm`,
which is a CUTLASS GEMM tuned for tile sizes that don't fit a single-row
problem well. Our custom gemv targets exactly this hole: a true matvec for
B200 W4A4 decode.

We measure three things per shape:
  * nvfp4r.gemv               : our specialized matvec (this PR)
  * vllm.cutlass_scaled_fp4_mm: what vLLM ModelOpt-NVFP4 actually calls today
  * bf16 torch.matmul          : sanity reference (lower bound for FP4)

Shapes are pulled from the actual nn.Linear projections of the models we run
(DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B/Math-7B, Qwen3-8B). gate/up_proj is
left out (the gate/up projections are benchmarked via the GEMM path).
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch

import nvfp4r  # noqa: F401  (registers torch.ops.nvfp4r.*)
import vllm._custom_ops as vops

from common import cuda_time_ms


# (model_tag, layer_tag, N, K) -- N is nn.Linear out_features, K is in_features.
# For decode we always have M (= input batch token count) == 1.
_DECODE_SHAPES: list[tuple[str, str, int, int]] = [
    # DR-7B / Qwen2.5-7B / Qwen2.5-Math-7B (hidden=3584, inter=18944,
    # 28 q-heads / 4 kv-heads, head_dim=128 -> qkv N = 28*128 + 2*4*128 = 4608).
    ("dr7b",      "qkv_proj",   4608, 3584),
    ("dr7b",      "o_proj",     3584, 3584),
    ("dr7b",      "gate_proj", 18944, 3584),  # treated separately below if you want
    ("dr7b",      "up_proj",   18944, 3584),
    ("dr7b",      "down_proj",  3584, 18944),

    # Qwen3-8B (hidden=4096, inter=12288, 32 q-heads / 8 kv-heads, head_dim=128
    # -> qkv N = 32*128 + 2*8*128 = 6144).
    ("qwen3-8b",  "qkv_proj",   6144, 4096),
    ("qwen3-8b",  "o_proj",     4096, 4096),
    ("qwen3-8b",  "gate_proj", 12288, 4096),
    ("qwen3-8b",  "up_proj",   12288, 4096),
    ("qwen3-8b",  "down_proj",  4096, 12288),
]


def _quantize_pair(
    A_bf: torch.Tensor,  # [M, K]
    B_bf: torch.Tensor,  # [N, K]
):
    """Produce both vLLM-swizzled and nvfp4r-flat NVFP4 representations.

    We use ``input_global_scale = 1.0`` for both paths so that the
    FP8 block scales already encode the original BF16 magnitude (block_amax/6).
    Our nvfp4r.gemv kernel does not apply a final ``alpha`` rescale (it just
    accumulates ``W_fp4 * X_fp4 * sf_W * sf_X``), so any non-unity global scale
    would explode the FP16 accumulator. For vLLM we pass ``alpha = 1.0`` to
    keep the two paths apples-to-apples.
    """
    one = torch.ones((), dtype=torch.float32, device=A_bf.device)

    # Swizzled (vLLM CUTLASS layout).
    A_fp4_sw, A_sf_sw = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=True)
    B_fp4_sw, B_sf_sw = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=True)

    # Flat (our gemv layout: [..., K/16] uint8 in row-major).
    A_fp4, A_sf = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=False)
    B_fp4, B_sf = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=False)

    alpha = one
    return {
        "swizzled": (A_fp4_sw, B_fp4_sw, A_sf_sw, B_sf_sw, alpha),
        "flat":     (A_fp4,   B_fp4,   A_sf,   B_sf,   alpha),
    }


def bench_one(N: int, K: int, dtype: torch.dtype, dev: torch.device) -> dict:
    M = 1
    A_bf = torch.randn(M, K, dtype=dtype, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=dtype, device=dev) * 0.05

    q = _quantize_pair(A_bf, B_bf)
    A_sw, B_sw, Asf_sw, Bsf_sw, alpha = q["swizzled"]
    A_flat, B_flat, Asf, Bsf, _ = q["flat"]

    # ---- vLLM cutlass FP4 GEMM @ M=1 (today's baseline for decode) ----
    def run_vllm():
        return vops.cutlass_scaled_fp4_mm(A_sw, B_sw, Asf_sw, Bsf_sw, alpha,
                                          out_dtype=dtype)

    # ---- nvfp4r.gemv ([L, M=N_out, K/2] for weight; [L, N_pad=1, K/2] for x) ----
    weight = B_flat.unsqueeze(0).contiguous()                      # [1, N, K/2]
    weight_scale = Bsf.view(torch.uint8).unsqueeze(0).contiguous() # [1, N, K/16]
    x = A_flat.view(1, 1, -1).contiguous()                          # [1, 1, K/2]
    x_scale = Asf.view(torch.uint8).view(1, 1, -1).contiguous()     # [1, 1, K/16]
    out_buf = torch.empty(1, N, dtype=torch.float16, device=dev)

    def run_nvfp4r():
        torch.ops.nvfp4r.gemv(weight, weight_scale, x, x_scale, out_buf)
        return out_buf

    # ---- BF16 dense reference ----
    def run_bf16():
        return torch.matmul(A_bf, B_bf.t())

    # Sanity: numerical agreement (loose tolerance, FP16 vs FP32 vs FP4 paths).
    with torch.no_grad():
        y_nv = run_nvfp4r().float()
        y_vl = run_vllm().float()
    rel = (y_nv - y_vl).abs() / (y_vl.abs() + 1e-3)
    max_rel = rel.max().item()

    nvr_med, nvr_p10, nvr_p90 = cuda_time_ms(run_nvfp4r)
    vll_med, vll_p10, vll_p90 = cuda_time_ms(run_vllm)
    bf_med,  _,       _       = cuda_time_ms(run_bf16)

    return {
        "shape":        (M, N, K),
        "nvfp4r_us":    nvr_med * 1000.0,
        "vllm_us":      vll_med * 1000.0,
        "bf16_us":      bf_med  * 1000.0,
        "max_rel_err":  max_rel,
    }


def main(shapes: Iterable[tuple[str, str, int, int]], dtype: torch.dtype) -> None:
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(dev)}")
    print(f"BF16 reference dtype: {dtype}")
    print()
    print(f"{'model':>10} {'layer':>10} {'N':>6} {'K':>6} | "
          f"{'nvfp4r [us]':>12} {'vllm fp4 [us]':>14} {'bf16 [us]':>11} | "
          f"{'speedup vs vllm':>16} {'max rel err':>11}")
    print("-" * 130)
    for tag, layer, N, K in shapes:
        try:
            r = bench_one(N, K, dtype, dev)
        except Exception as e:
            print(f"{tag:>10} {layer:>10} {N:>6} {K:>6} | ERROR: {e}")
            continue
        print(
            f"{tag:>10} {layer:>10} {N:>6} {K:>6} | "
            f"{r['nvfp4r_us']:>12.2f} {r['vllm_us']:>14.2f} {r['bf16_us']:>11.2f} | "
            f"{r['vllm_us'] / r['nvfp4r_us']:>16.2f}x {r['max_rel_err']:>11.4f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                    help="dtype for the activation/weight before NVFP4 quant")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    main(_DECODE_SHAPES, dt)
