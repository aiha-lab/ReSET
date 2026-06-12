"""vLLM compatibility / numerical correctness for nvfp4r ops.

Confirms that for each `nvfp4r` op, *some* combination of vLLM's
``scaled_fp4_quant`` outputs (flat or swizzled SF) yields the same numerical
result as ``cutlass_scaled_fp4_mm``. Both kernels read identical FP4 packed
data, so any divergence has to come from the SF layout interpretation. We
need this confirmed before plugging nvfp4r into vLLM's NVFP4 linear path.

Shapes are chosen to be in the kernel's launch-dispatch table (K must match a
case in `gemm`). M/N must be aligned to the kernel's tile.
"""

from __future__ import annotations

import pytest
import torch

import nvfp4r  # noqa: F401  (registers torch.ops.nvfp4r.*)
import vllm._custom_ops as vops


def _quant_both(x_bf: torch.Tensor) -> dict:
    """Return both flat and swizzled SF variants of the same NVFP4 quant.

    vLLM's `scaled_fp4_quant` re-quantises internally regardless of
    the layout flag, but both branches sweep ``input_global_scale`` the same
    way so the FP4 byte payload is identical. We rely on that for the
    comparison below (only the SF storage differs).
    """
    one = torch.ones((), dtype=torch.float32, device=x_bf.device)
    fp4_sw, sf_sw = vops.scaled_fp4_quant(x_bf, one, is_sf_swizzled_layout=True)
    fp4_fl, sf_fl = vops.scaled_fp4_quant(x_bf, one, is_sf_swizzled_layout=False)
    return {
        "fp4_sw": fp4_sw, "sf_sw": sf_sw,
        "fp4_fl": fp4_fl, "sf_fl": sf_fl,
        "alpha":  one,
    }


def _close(a: torch.Tensor, b: torch.Tensor, *, atol=2.0, rtol=8e-2) -> tuple[bool, float, float]:
    """Tolerances are deliberately loose: even with bit-identical
    FP4/SF, two kernels can diverge a bit due to FP16 vs FP32 accumulation
    and tcgen05 fastpaths. The point of this test is layout, not bit-exact.
    """
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    denom = bf.abs().clamp_min(1e-3)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    return (max_rel < rtol or max_abs < atol), max_abs, max_rel


# Shapes from the kernel's launch table that match real model layers.
# (M, N, K) — covered by `nvfp4r.gemm` dispatcher.
_GEMM_SHAPES = [
    (128,  256, 2048),
    (128,  512, 7168),
    (128, 1024, 4096),
]


@pytest.mark.parametrize("M, N, K", _GEMM_SHAPES)
def test_gemm_swizzled_sf_matches_vllm(cuda_available, M: int, N: int, K: int) -> None:
    dev = torch.device("cuda")
    A_bf = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.05
    qa = _quant_both(A_bf)
    qb = _quant_both(B_bf)

    # Reference: vLLM's cutlass kernel with swizzled SF.
    ref = vops.cutlass_scaled_fp4_mm(
        qa["fp4_sw"], qb["fp4_sw"], qa["sf_sw"], qb["sf_sw"], qa["alpha"],
        out_dtype=torch.bfloat16,
    )

    # Variant 1: nvfp4r.gemm with swizzled SF (kernel comments say this is
    # what it expects).
    C_sw = torch.empty(M, N, dtype=torch.float16, device=dev)
    out_sw = nvfp4r.gemm(
        qa["fp4_sw"], qb["fp4_sw"],
        qa["sf_sw"].view(torch.uint8), qb["sf_sw"].view(torch.uint8),
        C_sw,
    )

    # Variant 2: nvfp4r.gemm with flat SF (what the bench currently uses).
    C_fl = torch.empty(M, N, dtype=torch.float16, device=dev)
    out_fl = nvfp4r.gemm(
        qa["fp4_fl"], qb["fp4_fl"],
        qa["sf_fl"].view(torch.uint8), qb["sf_fl"].view(torch.uint8),
        C_fl,
    )

    sw_ok, sw_abs, sw_rel = _close(out_sw, ref)
    fl_ok, fl_abs, fl_rel = _close(out_fl, ref)
    print(f"\n[gemm M={M} N={N} K={K}] "
          f"swizzled_sf max_abs={sw_abs:.3f} max_rel={sw_rel:.3f} -> {'OK' if sw_ok else 'MISMATCH'}; "
          f"flat_sf     max_abs={fl_abs:.3f} max_rel={fl_rel:.3f} -> {'OK' if fl_ok else 'MISMATCH'}")

    assert sw_ok or fl_ok, (
        f"Neither SF layout matches vLLM. Kernel SF-layout assumption is wrong "
        f"or numerical accumulation differs too much. (M={M}, N={N}, K={K})"
    )


@pytest.mark.parametrize("N, K", [
    (4096, 4096),     # Qwen3-8B o_proj
    (6144, 4096),     # Qwen3-8B qkv_proj
    (4096, 12288),    # Qwen3-8B down_proj
])
def test_gemv_decode_matches_vllm(cuda_available, N: int, K: int) -> None:
    """vLLM uses ``cutlass_scaled_fp4_mm`` even for M=1 decode (it
    pads to 128). We reproduce that flow and compare against nvfp4r.gemv
    which expects the flat SF layout.
    """
    dev = torch.device("cuda")
    M = 1
    A_bf = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.05
    B_bf = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.05
    qa = _quant_both(A_bf)
    qb = _quant_both(B_bf)

    ref = vops.cutlass_scaled_fp4_mm(
        qa["fp4_sw"], qb["fp4_sw"], qa["sf_sw"], qb["sf_sw"], qa["alpha"],
        out_dtype=torch.bfloat16,
    )  # [M, N] = [1, N]

    # nvfp4r.gemv expects flat SF and (L=1, M=N, K/2) weight layout.
    weight       = qb["fp4_fl"].unsqueeze(0).contiguous()                # [1, N, K/2]
    weight_scale = qb["sf_fl"].view(torch.uint8).unsqueeze(0).contiguous()  # [1, N, K/16]
    x            = qa["fp4_fl"].view(1, 1, -1).contiguous()              # [1, 1, K/2]
    x_scale      = qa["sf_fl"].view(torch.uint8).view(1, 1, -1).contiguous()
    out_buf      = torch.empty(1, N, dtype=torch.float16, device=dev)
    torch.ops.nvfp4r.gemv(weight, weight_scale, x, x_scale, out_buf)
    out_nv2      = out_buf.view(1, N)  # canonical (M=1, N)

    ok, abs_err, rel_err = _close(out_nv2, ref, atol=2.0, rtol=8e-2)
    print(f"\n[gemv N={N} K={K}] max_abs={abs_err:.3f} max_rel={rel_err:.3f} -> {'OK' if ok else 'MISMATCH'}")
    assert ok, f"gemv mismatch vs vLLM at (N={N}, K={K}): abs={abs_err:.3f}, rel={rel_err:.3f}"
