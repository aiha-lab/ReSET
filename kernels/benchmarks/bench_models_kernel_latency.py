"""Kernel-level latency tables across the target reasoning models.

Produces three tables (one per nvfp4r op) over five models:
  * DeepSeek-R1-Distill-Qwen-7B  (Qwen2.5-7B arch)
  * DeepSeek-R1-Distill-Qwen-14B (Qwen2.5-14B arch)
  * Qwen3-8B
  * Qwen3-14B
  * Qwen3-32B

For each layer in each model we time:
  * nvfp4r kernel
  * vLLM-CUTLASS    (vops.cutlass_scaled_fp4_mm)
  * FlashInfer-CUTLASS (vllm.utils.flashinfer.flashinfer_scaled_fp4_mm,
    backend="cutlass") -- this is what stock vLLM auto-selects on B200,
    so it's the most realistic baseline for "how fast is production today".
  * BF16 dense reference (torch.matmul / silu_and_mul)

GEMV is benchmarked at M=1 (decode), GEMM at M=128 (small prefill chunk).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import nvfp4r  # noqa: F401  (registers torch.ops.nvfp4r.*)
import nvfp4r.vllm_integration  # noqa: F401  (registers torch.ops.vllm.nvfp4r_*)
import vllm._custom_ops as vops
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

from common import cuda_time_ms, cuda_time_ms_graph

_HAS_FLASHINFER = has_flashinfer()

# Module-level switch flipped by ``main`` -- when True, every leg below is
# measured via CUDA-graph replay (pure GPU time, no Python/dispatcher
# overhead). When False (default), eager event timing is used.
_USE_GRAPH = False


def _time(fn) -> float:
    """Median time in ms; honors the global ``_USE_GRAPH`` switch."""
    if _USE_GRAPH:
        med, _, _ = cuda_time_ms_graph(fn)
    else:
        med, _, _ = cuda_time_ms(fn)
    return med


# ----------------------------------------------------------------------------
# Model dimensions (verified from HuggingFace config.json files).
# qkv_N        = (q_heads + 2 * kv_heads) * head_dim
# inter_N      = intermediate_size (single proj)
# o_N          = q_heads * head_dim    (== hidden for these models)
# down_K       = intermediate_size
# All head_dim = 128 for the family.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int
    inter:  int
    q_heads: int
    kv_heads: int
    layers: int
    head_dim: int = 128

    @property
    def qkv_n(self) -> int:
        return (self.q_heads + 2 * self.kv_heads) * self.head_dim

    @property
    def o_n(self) -> int:
        return self.q_heads * self.head_dim

MODELS: list[ModelSpec] = [
    ModelSpec("DR-Qwen-7B",  hidden=3584, inter=18944, q_heads=28, kv_heads=4,  layers=28),
    ModelSpec("DR-Qwen-14B", hidden=5120, inter=13824, q_heads=40, kv_heads=8,  layers=48),
    ModelSpec("Qwen3-8B",    hidden=4096, inter=12288, q_heads=32, kv_heads=8,  layers=36),
    ModelSpec("Qwen3-14B",   hidden=5120, inter=17408, q_heads=40, kv_heads=8,  layers=40),
    ModelSpec("Qwen3-32B",   hidden=5120, inter=25600, q_heads=64, kv_heads=8,  layers=64),
]


# ----------------------------------------------------------------------------
# Quantization helpers (input_global_scale=1.0 so FP8 block scales already
# encode the original BF16 magnitude; nvfp4r does not apply alpha).
# ----------------------------------------------------------------------------
def _quant_pair_swizzled_and_flat(A_bf, B_bf):
    one = torch.ones((), dtype=torch.float32, device=A_bf.device)
    A_fp4_sw, A_sf_sw = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=True)
    B_fp4_sw, B_sf_sw = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=True)
    A_fp4,    A_sf    = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=False)
    B_fp4,    B_sf    = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=False)
    return (A_fp4_sw, B_fp4_sw, A_sf_sw, B_sf_sw,
            A_fp4,    B_fp4,    A_sf,    B_sf,    one)


def _quant_swizzled(A_bf):
    one = torch.ones((), dtype=torch.float32, device=A_bf.device)
    A_fp4, A_sf = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=True)
    return A_fp4, A_sf, one


# ----------------------------------------------------------------------------
# 1. GEMV table (M=1 decode).
# ----------------------------------------------------------------------------
def bench_gemv_one(N: int, K: int, dtype, dev) -> dict:
    M = 1
    A_bf = torch.randn(M, K, dtype=dtype, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=dtype, device=dev) * 0.05

    (A_sw, B_sw, Asf_sw, Bsf_sw,
     A_flat, B_flat, Asf, Bsf, alpha) = _quant_pair_swizzled_and_flat(A_bf, B_bf)

    def run_vllm():
        return vops.cutlass_scaled_fp4_mm(A_sw, B_sw, Asf_sw, Bsf_sw, alpha,
                                          out_dtype=dtype)

    def run_flashinfer():
        return flashinfer_scaled_fp4_mm(A_sw, B_sw, Asf_sw, Bsf_sw, alpha,
                                        out_dtype=dtype, backend="cutlass")

    weight = B_flat.unsqueeze(0).contiguous()
    weight_scale = Bsf.view(torch.uint8).unsqueeze(0).contiguous()
    x = A_flat.view(1, 1, -1).contiguous()
    x_scale = Asf.view(torch.uint8).view(1, 1, -1).contiguous()
    out_buf = torch.empty(1, N, dtype=torch.float16, device=dev)

    def run_nvfp4r():
        torch.ops.nvfp4r.gemv(weight, weight_scale, x, x_scale, out_buf)
        return out_buf

    # [SHLEE/Approach-B v3] "linear" leg: routes through the new
    # ``vllm::nvfp4r_linear`` op which owns the M-based branching
    # internally. M=1 here so it always goes down the gemv branch.
    weight_2d   = B_flat                                            # [N, K/2] uint8
    weight_sw   = B_sw                                              # [N_pad, K/16_pad] FP8 (swizzled, for fallback)
    ws_flat_3d  = Bsf.view(torch.uint8).unsqueeze(0).contiguous()   # [1, N, K/16]
    out_dtype_int = 0 if dtype is torch.bfloat16 else 1
    one_alpha   = torch.ones((), dtype=torch.float32, device=dev)
    A_2d_bf     = A_bf                                              # [M, K]
    def run_nvfp4r_linear():
        return torch.ops.vllm.nvfp4r_linear(
            A_2d_bf, weight_2d, weight_sw, ws_flat_3d,
            one_alpha, one_alpha,
            1.0,                              # alpha_host
            N,                                # output_size
            0,                                # weights_padding_cols
            32,                               # gemv_max_m
            out_dtype_int,
            False,                            # enable_gemm
        )

    def run_bf16():
        return torch.matmul(A_bf, B_bf.t())

    nvr_med    = _time(run_nvfp4r)
    nvrlin_med = _time(run_nvfp4r_linear)
    vll_med    = _time(run_vllm)
    bf_med     = _time(run_bf16)
    fi_us = float("nan")
    if _HAS_FLASHINFER:
        try:
            fi_us = _time(run_flashinfer) * 1000
        except Exception:
            pass
    return {"nvfp4r_us":     nvr_med    * 1000,
            "nvfp4r_lin_us": nvrlin_med * 1000,
            "vllm_us":       vll_med    * 1000,
            "fi_us":         fi_us,
            "bf16_us":       bf_med     * 1000}


def gemv_layers(m: ModelSpec):
    # nn.Linear convention: weight is [out, in]; we report (N=out, K=in).
    #   qkv_proj : Linear(hidden,           qkv_n)
    #   o_proj   : Linear(q_heads*head_dim, hidden)   (GQA: K differs from hidden on Qwen3-32B)
    #   gate/up  : Linear(hidden,           inter)
    #   down     : Linear(inter,            hidden)
    yield ("qkv_proj",      m.qkv_n,  m.hidden)
    yield ("o_proj",        m.hidden, m.o_n)
    yield ("gate/up_proj",  m.inter,  m.hidden)
    yield ("down_proj",     m.hidden, m.inter)


def print_gemv_table(dtype, dev):
    print("=" * 160)
    print("Table 1.  GEMV (M=1 decode)  -  nvfp4r.gemv (raw kernel) vs nvfp4r_decode_linear (Approach B) vs vLLM-CUTLASS vs FlashInfer-CUTLASS vs BF16")
    print("=" * 160)
    print(f"{'model':>12} {'layer':>13} {'N':>6} {'K':>6} | "
          f"{'raw [us]':>10} {'apprB [us]':>11} {'vllm [us]':>10} {'flashinfer [us]':>16} {'bf16 [us]':>10} | "
          f"{'B vs vllm':>10} {'B vs fi':>9} {'B/raw':>7}")
    print("-" * 160)
    for m in MODELS:
        for layer, N, K in gemv_layers(m):
            r = bench_gemv_one(N, K, dtype, dev)
            fi_us = r['fi_us']
            fi_str = f"{fi_us:>16.2f}" if fi_us == fi_us else f"{'n/a':>16}"
            spd_b_vs_fi = (fi_us / r['nvfp4r_lin_us']) if fi_us == fi_us else float('nan')
            spd_b_vs_fi_str = f"{spd_b_vs_fi:>8.2f}x" if spd_b_vs_fi == spd_b_vs_fi else f"{'n/a':>9}"
            print(f"{m.name:>12} {layer:>13} {N:>6} {K:>6} | "
                  f"{r['nvfp4r_us']:>10.2f} {r['nvfp4r_lin_us']:>11.2f} "
                  f"{r['vllm_us']:>10.2f} {fi_str} {r['bf16_us']:>10.2f} | "
                  f"{r['vllm_us']/r['nvfp4r_lin_us']:>9.2f}x {spd_b_vs_fi_str} "
                  f"{r['nvfp4r_lin_us']/r['nvfp4r_us']:>6.2f}x")
        print("-" * 160)


# ----------------------------------------------------------------------------
# 2. GEMM table (prefill chunk, M=128).
# ----------------------------------------------------------------------------
def bench_gemm_one(M: int, N: int, K: int, dtype, dev) -> dict:
    A_bf = torch.randn(M, K, dtype=dtype, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=dtype, device=dev) * 0.05
    (A_sw, B_sw, Asf_sw, Bsf_sw,
     A_flat, B_flat, Asf, Bsf, alpha) = _quant_pair_swizzled_and_flat(A_bf, B_bf)

    def run_vllm():
        return vops.cutlass_scaled_fp4_mm(A_sw, B_sw, Asf_sw, Bsf_sw, alpha,
                                          out_dtype=dtype)

    def run_flashinfer():
        return flashinfer_scaled_fp4_mm(A_sw, B_sw, Asf_sw, Bsf_sw, alpha,
                                        out_dtype=dtype, backend="cutlass")

    C_buf = torch.empty(M, N, dtype=torch.float16, device=dev)

    def run_nvfp4r():
        return nvfp4r.gemm(A_flat, B_flat, Asf, Bsf, C_buf)

    def run_bf16():
        return torch.matmul(A_bf, B_bf.t())

    nvr_med = _time(run_nvfp4r)
    vll_med = _time(run_vllm)
    bf_med  = _time(run_bf16)
    fi_us = float("nan")
    if _HAS_FLASHINFER:
        try:
            fi_us = _time(run_flashinfer) * 1000
        except Exception:
            pass
    return {"nvfp4r_us": nvr_med * 1000, "vllm_us": vll_med * 1000,
            "fi_us":     fi_us,
            "bf16_us":   bf_med  * 1000}


def gemm_layers(m: ModelSpec):
    # Same (N, K) convention as gemv_layers: N == out_features, K == in_features.
    yield ("qkv_proj",      m.qkv_n,  m.hidden)
    yield ("o_proj",        m.hidden, m.o_n)
    yield ("gate/up_proj",  m.inter,  m.hidden)
    yield ("down_proj",     m.hidden, m.inter)


def print_gemm_table(dtype, dev, M: int = 128):
    print()
    print("=" * 140)
    print(f"Table 2.  GEMM (prefill, M={M})  -  nvfp4r.gemm vs vLLM-CUTLASS vs FlashInfer-CUTLASS vs BF16")
    print("=" * 140)
    print(f"{'model':>12} {'layer':>13} {'N':>6} {'K':>6} | "
          f"{'nvfp4r [us]':>12} {'vllm [us]':>10} {'flashinfer [us]':>16} {'bf16 [us]':>10} | "
          f"{'vs vllm':>9} {'vs fi':>9} {'vs bf16':>9}")
    print("-" * 140)
    for m in MODELS:
        for layer, N, K in gemm_layers(m):
            r = bench_gemm_one(M, N, K, dtype, dev)
            fi_us = r['fi_us']
            fi_str = f"{fi_us:>16.2f}" if fi_us == fi_us else f"{'n/a':>16}"
            spd_fi = (fi_us / r['nvfp4r_us']) if fi_us == fi_us else float('nan')
            spd_fi_str = f"{spd_fi:>8.2f}x" if spd_fi == spd_fi else f"{'n/a':>9}"
            print(f"{m.name:>12} {layer:>13} {N:>6} {K:>6} | "
                  f"{r['nvfp4r_us']:>12.2f} {r['vllm_us']:>10.2f} {fi_str} {r['bf16_us']:>10.2f} | "
                  f"{r['vllm_us']/r['nvfp4r_us']:>8.2f}x {spd_fi_str} "
                  f"{r['bf16_us']/r['nvfp4r_us']:>8.2f}x")
        print("-" * 140)


def bench_batched_gemv_one(M_decode: int, N: int, K: int, dtype, dev) -> dict:
    """Batched-GEMV scaling: L = M_decode tokens against a *broadcast*
    weight (stride_l = 0). This is the path Approach B's
    ``vllm::nvfp4r_linear`` takes during decode when batch_size > 1.

    The single-M GEMV table above only measures M_decode=1; this measures
    M_decode in {1, 4, 8, 16, 32}. We additionally compare against the
    cost of M_decode separate cutlass-cutlass-mm calls (the "stock"
    serialised baseline) to put nvfp4r's batched scaling into context.
    """
    A_bf = torch.randn(M_decode, K, dtype=dtype, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=dtype, device=dev) * 0.05

    one = torch.ones((), dtype=torch.float32, device=dev)
    A_sw, A_sf_sw = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=True)
    B_sw, B_sf_sw = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=True)
    A_flat, A_sf  = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=False)
    B_flat, B_sf  = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=False)

    # Stock: one cutlass / flashinfer mm per batched call (the kernels
    # themselves handle multi-row M internally; we just need the
    # apples-to-apples linear-equivalent time).
    def run_vllm():
        return vops.cutlass_scaled_fp4_mm(A_sw, B_sw, A_sf_sw, B_sf_sw, one,
                                          out_dtype=dtype)

    def run_flashinfer():
        return flashinfer_scaled_fp4_mm(A_sw, B_sw, A_sf_sw, B_sf_sw, one,
                                        out_dtype=dtype, backend="cutlass")

    # nvfp4r batched-GEMV: weight broadcast via stride-0 expand on L axis.
    # Mirrors apply_nvfp4r_linear's gemv branch byte-for-byte.
    weight_2d   = B_flat
    ws_flat_3d  = B_sf.view(torch.uint8).unsqueeze(0).contiguous()
    out_dtype_int = 0 if dtype is torch.bfloat16 else 1

    w_b  = weight_2d.unsqueeze(0).expand(M_decode, -1, -1)    # [M_decode, N, K/2]
    ws_b = ws_flat_3d.expand(M_decode, -1, -1)                # [M_decode, N, K/16]
    x_b  = A_flat.view(M_decode, 1, -1)                       # [M_decode, 1, K/2]
    xs_b = A_sf.view(torch.uint8).view(M_decode, 1, -1)
    out_buf = torch.empty(M_decode, N, dtype=dtype, device=dev)

    def run_nvfp4r_raw():
        torch.ops.nvfp4r.gemv(w_b, ws_b, x_b, xs_b, out_buf, 1.0)
        return out_buf

    def run_nvfp4r_linear():
        return torch.ops.vllm.nvfp4r_linear(
            A_bf, weight_2d, B_sw, ws_flat_3d,
            one, one, 1.0, N, 0, 32, out_dtype_int, False,
        )

    def run_bf16():
        return torch.matmul(A_bf, B_bf.t())

    nvr_med    = _time(run_nvfp4r_raw)
    nvrlin_med = _time(run_nvfp4r_linear)
    vll_med    = _time(run_vllm)
    bf_med     = _time(run_bf16)
    fi_us = float("nan")
    if _HAS_FLASHINFER:
        try:
            fi_us = _time(run_flashinfer) * 1000
        except Exception:
            pass
    return {"nvfp4r_us":     nvr_med    * 1000,
            "nvfp4r_lin_us": nvrlin_med * 1000,
            "vllm_us":       vll_med    * 1000,
            "fi_us":         fi_us,
            "bf16_us":       bf_med     * 1000}


def print_batched_gemv_table(dtype, dev, M_decodes: list[int]):
    """One row per (model, layer, M_decode) -- shows whether nvfp4r's
    batched-GEMV path stays competitive as the decode batch grows. The
    e2e v3 sweep showed catastrophic regression at M_decode >= 8; this
    tells us whether the regression is at the kernel level or in the
    vLLM integration above it.
    """
    print()
    print("=" * 175)
    print(f"Table 1b. BATCHED-GEMV scaling  (M_decode ∈ {M_decodes})")
    print(f"          weight broadcast via stride-0 expand on L axis (mirrors apply_nvfp4r_linear gemv branch)")
    print("=" * 175)
    print(f"{'model':>12} {'layer':>13} {'N':>6} {'K':>6} {'M':>3} | "
          f"{'raw [us]':>10} {'apprB [us]':>11} {'vllm [us]':>10} {'flashinfer [us]':>16} {'bf16 [us]':>10} | "
          f"{'B vs vllm':>10} {'B vs fi':>9} {'us/tok B':>9}")
    print("-" * 175)
    for m in MODELS:
        for layer, N, K in gemv_layers(m):
            for M_decode in M_decodes:
                r = bench_batched_gemv_one(M_decode, N, K, dtype, dev)
                fi_us = r['fi_us']
                fi_str = f"{fi_us:>16.2f}" if fi_us == fi_us else f"{'n/a':>16}"
                spd_b_vs_fi = (fi_us / r['nvfp4r_lin_us']) if fi_us == fi_us else float('nan')
                spd_b_vs_fi_str = f"{spd_b_vs_fi:>8.2f}x" if spd_b_vs_fi == spd_b_vs_fi else f"{'n/a':>9}"
                per_tok = r['nvfp4r_lin_us'] / max(M_decode, 1)
                print(f"{m.name:>12} {layer:>13} {N:>6} {K:>6} {M_decode:>3} | "
                      f"{r['nvfp4r_us']:>10.2f} {r['nvfp4r_lin_us']:>11.2f} "
                      f"{r['vllm_us']:>10.2f} {fi_str} {r['bf16_us']:>10.2f} | "
                      f"{r['vllm_us']/r['nvfp4r_lin_us']:>9.2f}x {spd_b_vs_fi_str} "
                      f"{per_tok:>8.2f}")
            print("-" * 175)


def bench_linear_m_one(M: int, N: int, K: int, dtype, dev) -> dict:
    """Linear-level micro-bench at arbitrary M (decode/prefill mix).

    Routes nvfp4r through the production single-dispatch op
    ``vllm::nvfp4r_linear`` so the M-branching the kernel sees matches
    what vLLM's compiled forward actually invokes (gemv for M=1,
    gemm for M%128==0, otherwise a CUTLASS fallback inlined inside the
    same op). Lets us tell apart "kernel wins/losses" from
    "dispatcher overhead" at every M the e2e sweep cares about.

    Compared baselines:
      * bf16 nn.Linear (``torch.matmul``) — explains why eager-mode
        e2e showed bf16 beating nvfp4 at small M (no quantize +
        ~3-line dispatcher path vs ~8 lines + scaled_fp4_quant for
        nvfp4 backends).
      * vLLM-CUTLASS — vLLM's manually selectable backend.
      * FlashInfer-CUTLASS — vLLM's auto-pick on B200 (production
        default).
    """
    A_bf = torch.randn(M, K, dtype=dtype, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=dtype, device=dev) * 0.05

    one = torch.ones((), dtype=torch.float32, device=dev)
    A_sw,   A_sf_sw   = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=True)
    B_sw,   B_sf_sw   = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=True)
    A_flat, A_sf      = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=False)
    B_flat, B_sf      = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=False)

    def run_bf16():
        return torch.matmul(A_bf, B_bf.t())

    def run_vllm():
        return vops.cutlass_scaled_fp4_mm(A_sw, B_sw, A_sf_sw, B_sf_sw, one,
                                          out_dtype=dtype)

    def run_flashinfer():
        return flashinfer_scaled_fp4_mm(A_sw, B_sw, A_sf_sw, B_sf_sw, one,
                                        out_dtype=dtype, backend="cutlass")

    weight_packed = B_sw                                                # [N, K/2] uint8
    weight_scale_swizzled = B_sf_sw                                     # FP8 swizzled SF
    ws_flat_3d    = B_sf.view(torch.uint8).unsqueeze(0).contiguous()    # [1, N, K/16] uint8 (gemv only)
    out_dtype_int = 0 if dtype is torch.bfloat16 else 1

    def run_nvfp4r_linear():
        return torch.ops.vllm.nvfp4r_linear(
            A_bf, weight_packed, weight_scale_swizzled, ws_flat_3d,
            one, one, 1.0, N, 0, 1, out_dtype_int, True,
        )

    if M <= 1 and N % 8 == 0:
        path = "gemv"
    elif M % 128 == 0 and N % 128 == 0:
        path = "gemm"
    else:
        path = "fallback"

    bf_us  = _time(run_bf16)            * 1000
    vll_us = _time(run_vllm)            * 1000
    nvr_us = _time(run_nvfp4r_linear)   * 1000
    fi_us  = float("nan")
    if _HAS_FLASHINFER:
        try:
            fi_us = _time(run_flashinfer) * 1000
        except Exception:
            pass

    return {"bf16_us":   bf_us,
            "vllm_us":   vll_us,
            "fi_us":     fi_us,
            "nvfp4r_us": nvr_us,
            "path":      path}


def print_linear_m_sweep_table(dtype, dev, Ms: list[int]):
    """One row per (model, layer, M) — comparable to e2e bs in
    ``sweep_e2e_latency.py``."""
    print()
    print("=" * 175)
    print(f"Table M-sweep.  Linear-level micro-bench across M ∈ {Ms}")
    print(f"          BF16 nn.Linear  /  vLLM-CUTLASS  /  FlashInfer-CUTLASS  /  nvfp4r (vllm::nvfp4r_linear, gemv_max_m=1, enable_gemm=1)")
    print("=" * 175)
    print(f"{'model':>12} {'layer':>13} {'N':>6} {'K':>6} {'M':>4} {'path':>9} | "
          f"{'bf16 [us]':>10} {'vllm [us]':>10} {'fi [us]':>10} {'nvfp4r [us]':>12} | "
          f"{'B vs vllm':>10} {'B vs fi':>9} {'B vs bf16':>10}")
    print("-" * 175)
    for m in MODELS:
        for layer, N, K in gemm_layers(m):
            for M in Ms:
                r = bench_linear_m_one(M, N, K, dtype, dev)
                fi_us  = r['fi_us']
                fi_str = f"{fi_us:>10.2f}" if fi_us == fi_us else f"{'n/a':>10}"
                spd_fi = (fi_us / r['nvfp4r_us']) if fi_us == fi_us else float('nan')
                spd_fi_str = f"{spd_fi:>8.2f}x" if spd_fi == spd_fi else f"{'n/a':>9}"
                print(f"{m.name:>12} {layer:>13} {N:>6} {K:>6} {M:>4} {r['path']:>9} | "
                      f"{r['bf16_us']:>10.2f} {r['vllm_us']:>10.2f} {fi_str} {r['nvfp4r_us']:>12.2f} | "
                      f"{r['vllm_us']/r['nvfp4r_us']:>9.2f}x {spd_fi_str} "
                      f"{r['bf16_us']/r['nvfp4r_us']:>9.2f}x")
            print("-" * 175)


# ----------------------------------------------------------------------------
# Qwen3-8B kernel-only sweep — raw __global__ kernel comparison.
#
# Unlike ``bench_linear_m_one`` (which routes nvfp4r through the
# ``vllm::nvfp4r_linear`` op so wrapper GPU work like
# ``scaled_fp4_quant`` + ``transpose().contiguous()`` + ``mul/cast``
# is included), this benches just the raw ``torch.ops.nvfp4r.{gemv,gemm}``
# kernel. The intent: answer the apples-to-apples question "are our
# Blackwell ``__global__`` kernels actually faster than the production
# CUTLASS kernels?", separated from any integration overhead.
#
# nvfp4r path mapping (depends on M):
#   M == 1                  -> torch.ops.nvfp4r.gemv (single-token decode)
#   1 < M <= 64             -> torch.ops.nvfp4r.gemv (batched, weight stride-0 broadcast)
#   M >= 128 and M%128 == 0 -> torch.ops.nvfp4r.gemm
# ----------------------------------------------------------------------------
def bench_qwen3_8b_global_one(M: int, N: int, K: int, dtype, dev) -> dict:
    A_bf = torch.randn(M, K, dtype=dtype, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=dtype, device=dev) * 0.05

    one = torch.ones((), dtype=torch.float32, device=dev)
    A_sw,   A_sf_sw = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=True)
    B_sw,   B_sf_sw = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=True)
    A_flat, A_sf    = vops.scaled_fp4_quant(A_bf, one, is_sf_swizzled_layout=False)
    B_flat, B_sf    = vops.scaled_fp4_quant(B_bf, one, is_sf_swizzled_layout=False)

    def run_bf16():
        return torch.matmul(A_bf, B_bf.t())

    def run_vllm():
        return vops.cutlass_scaled_fp4_mm(A_sw, B_sw, A_sf_sw, B_sf_sw, one,
                                          out_dtype=dtype)

    def run_flashinfer():
        return flashinfer_scaled_fp4_mm(A_sw, B_sw, A_sf_sw, B_sf_sw, one,
                                        out_dtype=dtype, backend="cutlass")

    # --- nvfp4r raw kernel (pre-quantised inputs, no wrapper) ---
    if M == 1:
        # GEMV (M=1 decode). [N, K/2] / [N, K/16] / [1, 1, K/2] / [1, 1, K/16]
        weight       = B_flat.unsqueeze(0).contiguous()
        weight_scale = B_sf.view(torch.uint8).unsqueeze(0).contiguous()
        x_b          = A_flat.view(1, 1, -1).contiguous()
        x_sf_b       = A_sf.view(torch.uint8).view(1, 1, -1).contiguous()
        out_buf      = torch.empty(1, N, dtype=torch.float16, device=dev)

        def run_nvfp4r():
            torch.ops.nvfp4r.gemv(weight, weight_scale, x_b, x_sf_b, out_buf, 1.0)
            return out_buf

        path = "gemv"

    elif M <= 64:
        # Batched-GEMV: weight broadcast via stride-0 expand on L axis.
        w_b   = B_flat.unsqueeze(0).expand(M, -1, -1)               # [M, N, K/2]
        ws_b  = (B_sf.view(torch.uint8).unsqueeze(0)                # [M, N, K/16]
                       .expand(M, -1, -1))
        x_b   = A_flat.view(M, 1, -1)                                # [M, 1, K/2]
        xs_b  = A_sf.view(torch.uint8).view(M, 1, -1)                # [M, 1, K/16]
        out_buf = torch.empty(M, N, dtype=dtype, device=dev)

        def run_nvfp4r():
            torch.ops.nvfp4r.gemv(w_b, ws_b, x_b, xs_b, out_buf, 1.0)
            return out_buf

        path = "batched-gemv"

    else:
        # GEMM (prefill / large-batch decode). M must be a multiple of 128 for
        # current launch table (BLOCK_M=128 in cuda/gemm/gemm_nvfp4.cu).
        assert M % 128 == 0, f"GEMM path requires M % 128 == 0, got M={M}"
        C_buf = torch.empty(M, N, dtype=torch.float16, device=dev)

        def run_nvfp4r():
            return nvfp4r.gemm(A_flat, B_flat, A_sf, B_sf, C_buf)

        path = "gemm"

    bf_us  = _time(run_bf16)        * 1000
    vll_us = _time(run_vllm)        * 1000
    nvr_us = _time(run_nvfp4r)      * 1000
    fi_us  = float("nan")
    if _HAS_FLASHINFER:
        try:
            fi_us = _time(run_flashinfer) * 1000
        except Exception:
            pass

    return {"bf16_us":   bf_us,
            "vllm_us":   vll_us,
            "fi_us":     fi_us,
            "nvfp4r_us": nvr_us,
            "path":      path}


def print_qwen3_8b_global_table(dtype, dev, Ms: list[int]):
    """One row per (Qwen3-8B layer, M). Raw __global__ kernels only."""
    QWEN3_8B = next(m for m in MODELS if m.name == "Qwen3-8B")
    layers = list(gemm_layers(QWEN3_8B))

    print()
    print("=" * 145)
    print(f"Qwen3-8B __global__ kernel comparison  (M ∈ {Ms})")
    print(f"  bf16 = torch.matmul   |   vllm = vops.cutlass_scaled_fp4_mm   |   "
          f"flashinfer = flashinfer_scaled_fp4_mm(backend='cutlass')   |   nvfp4r = torch.ops.nvfp4r.{{gemv,gemm}}")
    print("=" * 145)
    # Plain text header (kept aligned for terminal viewing).
    print(f"{'layer':>13} {'N':>6} {'K':>6} {'M':>4} {'path':>13} | "
          f"{'bf16 [us]':>10} {'vllm [us]':>10} {'fi [us]':>10} {'nvfp4r [us]':>12} | "
          f"{'B vs vllm':>10} {'B vs fi':>9} {'B vs bf16':>10}")
    print("-" * 145)
    # Markdown-friendly rows are also emitted (prefix `|md|`) so we can
    # grep them out into the result file without re-formatting.
    print("|md| | layer | N | K | M | path | bf16 [us] | vllm-cutlass [us] | flashinfer [us] | nvfp4r [us] | B vs vllm | B vs fi |")
    print("|md| |---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for layer, N, K in layers:
        for M in Ms:
            r = bench_qwen3_8b_global_one(M, N, K, dtype, dev)
            fi_us = r['fi_us']
            fi_str = f"{fi_us:>10.2f}" if fi_us == fi_us else f"{'n/a':>10}"
            spd_vll = r['vllm_us'] / r['nvfp4r_us'] if r['nvfp4r_us'] > 0 else float('nan')
            spd_fi  = (fi_us / r['nvfp4r_us']) if (fi_us == fi_us and r['nvfp4r_us'] > 0) else float('nan')
            spd_bf  = r['bf16_us'] / r['nvfp4r_us'] if r['nvfp4r_us'] > 0 else float('nan')
            spd_vll_str = f"{spd_vll:>9.2f}x"
            spd_fi_str  = f"{spd_fi:>8.2f}x" if spd_fi == spd_fi else f"{'n/a':>9}"
            spd_bf_str  = f"{spd_bf:>9.2f}x"

            print(f"{layer:>13} {N:>6} {K:>6} {M:>4} {r['path']:>13} | "
                  f"{r['bf16_us']:>10.2f} {r['vllm_us']:>10.2f} {fi_str} {r['nvfp4r_us']:>12.2f} | "
                  f"{spd_vll_str} {spd_fi_str} {spd_bf_str}")

            # Markdown row.
            fi_md = f"{fi_us:.2f}" if fi_us == fi_us else "n/a"
            spd_fi_md = f"{spd_fi:.2f}x" if spd_fi == spd_fi else "n/a"
            print(f"|md| | {layer} | {N} | {K} | {M} | {r['path']} | "
                  f"{r['bf16_us']:.2f} | {r['vllm_us']:.2f} | {fi_md} | "
                  f"{r['nvfp4r_us']:.2f} | {spd_vll:.2f}x | {spd_fi_md} |")
        print("-" * 145)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--gemm-m",      type=int, default=128, help="prefill chunk for GEMM")
    ap.add_argument("--skip-gemv",   action="store_true")
    ap.add_argument("--skip-gemm",   action="store_true")
    ap.add_argument("--batched-gemv", action="store_true",
                    help="Run the batched-GEMV scaling table (M_decode ∈ {1,4,8,16,32}).")
    ap.add_argument("--batched-gemv-ms", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32],
                    help="M_decode values to sweep when --batched-gemv is set.")
    ap.add_argument("--linear-m-sweep", action="store_true",
                    help="Sweep linear-level latency across multiple M values "
                         "(BF16 nn.Linear vs vLLM-CUTLASS vs FlashInfer-CUTLASS "
                         "vs vllm::nvfp4r_linear single-op).")
    ap.add_argument("--linear-m-sweep-ms", type=int, nargs="+",
                    default=[1, 16, 32, 64, 128],
                    help="M values to sweep when --linear-m-sweep is set.")
    ap.add_argument("--qwen3-8b-global", action="store_true",
                    help="Qwen3-8B 4-layer raw __global__ kernel comparison "
                         "(bf16 / vllm-cutlass / flashinfer-cutlass / nvfp4r). "
                         "Standalone mode -- skips the legacy three tables.")
    ap.add_argument("--qwen3-8b-global-ms", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32, 64, 128, 256],
                    help="M values for --qwen3-8b-global. nvfp4r path is "
                         "auto-selected: M=1 gemv / 1<M<=64 batched-gemv / "
                         "M>=128 (and M%128==0) gemm.")
    ap.add_argument("--graph",       action="store_true",
                    help="Time legs via CUDA-graph replay (pure GPU time, no Python/dispatcher overhead)")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dev = torch.device("cuda")

    global _USE_GRAPH
    _USE_GRAPH = args.graph

    print(f"GPU: {torch.cuda.get_device_name(dev)}")
    print(f"dtype: {dtype}")
    print(f"flashinfer available: {_HAS_FLASHINFER}")
    print(f"timing mode: {'CUDA-graph replay (pure GPU)' if _USE_GRAPH else 'eager event timing (host+GPU)'}")
    print()
    print("Model lineup:")
    for m in MODELS:
        print(f"  {m.name:>12}  hidden={m.hidden:>5} inter={m.inter:>5} "
              f"qkv_N={m.qkv_n:>5} layers={m.layers}")

    if args.linear_m_sweep:
        print_linear_m_sweep_table(dtype, dev, args.linear_m_sweep_ms)
        return

    if args.qwen3_8b_global:
        print_qwen3_8b_global_table(dtype, dev, args.qwen3_8b_global_ms)
        return

    if not args.skip_gemv:
        print_gemv_table(dtype, dev)
    if args.batched_gemv:
        print_batched_gemv_table(dtype, dev, args.batched_gemv_ms)
    if not args.skip_gemm:
        print_gemm_table(dtype, dev, M=args.gemm_m)


if __name__ == "__main__":
    main()
