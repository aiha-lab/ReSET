"""End-to-end test of the native NVFP4R backend.

Builds a synthetic ModelOpt-style layer module (same attribute shape
as ``ModelOptNvFp4LinearMethod`` produces after ``process_weights_after_loading``)
and runs :func:`apply_nvfp4r_linear` against vLLM's stock VLLM_CUTLASS path.
Both must agree up to FP16 accumulation tolerance.

We exercise:
  * decode batch (M ∈ {1, 2, 8, 32}) -> batched-GEMV (over the L axis)
  * prefill (M=128, 256)             -> nvfp4r.gemm
  * unsupported K / misaligned M     -> transparent CUTLASS fallback
"""

from __future__ import annotations

import pytest
import torch

import nvfp4r  # noqa: F401
import vllm._custom_ops as vops
from nvfp4r.vllm_integration import (
    apply_nvfp4r_linear,
    prime_nvfp4r_caches,
    _GEMV_MAX_M,
)
from vllm.model_executor.layers.quantization.utils import nvfp4_utils
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
)


def _make_layer(N: int, K: int, dtype: torch.dtype, device) -> torch.nn.Module:
    """Build a layer module that mimics what
    ``ModelOptNvFp4LinearMethod.process_weights_after_loading`` produces
    *and* what ``convert_to_nvfp4_linear_kernel_format(NVFP4R, layer)``
    leaves behind (i.e. with our caches primed).
    """
    layer = torch.nn.Module()
    layer.input_size_per_partition  = K
    layer.output_size_per_partition = N
    layer.logical_widths            = [N]
    layer.weights_padding_cols      = 0

    # BF16 weights -> NVFP4 weight + (swizzled) weight scale.
    W_bf = torch.randn(N, K, dtype=dtype, device=device) * 0.05
    one  = torch.ones((), dtype=torch.float32, device=device)
    W_fp4, W_sf_swizzled = vops.scaled_fp4_quant(W_bf, one, is_sf_swizzled_layout=True)

    layer.weight       = torch.nn.Parameter(W_fp4, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(W_sf_swizzled, requires_grad=False)

    layer.input_global_scale     = torch.nn.Parameter(one.clone(), requires_grad=False)
    layer.input_global_scale_inv = torch.nn.Parameter(one.clone(), requires_grad=False)
    layer.weight_global_scale    = torch.nn.Parameter(one.clone(), requires_grad=False)
    layer.alpha                  = torch.nn.Parameter(one.clone(), requires_grad=False)

    prime_nvfp4r_caches(layer)
    return layer


def _close(a: torch.Tensor, b: torch.Tensor, *, atol=0.05, rtol=8e-2):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = bf.abs().clamp_min(1e-3)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    return (max_rel < rtol or max_abs < atol), max_abs, max_rel


_DECODE_SHAPES = [
    (4096, 4096),    # Qwen3-8B o_proj
    (6144, 4096),    # Qwen3-8B qkv_proj
    (4096, 12288),   # Qwen3-8B down_proj
]

_DECODE_M = [1, 2, 8, 32]


@pytest.mark.parametrize("N, K", _DECODE_SHAPES)
@pytest.mark.parametrize("M", _DECODE_M)
def test_apply_decode_path(cuda_available, M, N, K):
    """Native NVFP4R batched-GEMV vs vLLM CUTLASS reference."""
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    layer = _make_layer(N, K, dtype=dtype, device=dev)
    x = torch.randn(M, K, dtype=dtype, device=dev) * 0.05

    out_nv = apply_nvfp4r_linear(layer, x)
    out_ref = nvfp4_utils.apply_nvfp4_linear(NvFp4LinearBackend.VLLM_CUTLASS, layer, x)

    assert out_nv.shape == out_ref.shape == (M, N)
    assert out_nv.dtype == out_ref.dtype == dtype
    ok, abs_, rel_ = _close(out_nv, out_ref)
    print(f"\n[decode M={M} N={N} K={K}] abs={abs_:.4f} rel={rel_:.4f}")
    assert ok, f"decode mismatch (M={M},N={N},K={K}): abs={abs_:.4f}, rel={rel_:.4f}"


_PREFILL_SHAPES = [
    (128, 4096, 4096),
    (128, 6144, 4096),
    (256, 4096, 12288),
]


@pytest.mark.parametrize("M, N, K", _PREFILL_SHAPES)
def test_apply_prefill_path(cuda_available, M, N, K):
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    layer = _make_layer(N, K, dtype=dtype, device=dev)
    x = torch.randn(M, K, dtype=dtype, device=dev) * 0.05

    out_nv = apply_nvfp4r_linear(layer, x)
    out_ref = nvfp4_utils.apply_nvfp4_linear(NvFp4LinearBackend.VLLM_CUTLASS, layer, x)

    assert out_nv.shape == out_ref.shape == (M, N)
    ok, abs_, rel_ = _close(out_nv, out_ref)
    print(f"\n[prefill M={M} N={N} K={K}] abs={abs_:.4f} rel={rel_:.4f}")
    assert ok, f"prefill mismatch (M={M},N={N},K={K}): abs={abs_:.4f}, rel={rel_:.4f}"


def test_apply_falls_back_for_misaligned(cuda_available):
    """Phase 1 doesn't pad: M values that fall in the gap between the
    batched-GEMV upper bound and the gemm M-tile alignment must transparently
    route to the inlined CUTLASS fallback so the kernel never sees a bad
    shape.

    M=63 is past ``_GEMV_MAX_M`` (32) and not divisible by 64 -- exactly the
    "irregular prefill chunk" case we want to bail on.
    """
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    N, K, M = 4096, 4096, 63
    layer = _make_layer(N, K, dtype=dtype, device=dev)
    x = torch.randn(M, K, dtype=dtype, device=dev) * 0.05

    assert M > _GEMV_MAX_M and M % 64 != 0, "test setup: must miss every fast path"

    out_nv = apply_nvfp4r_linear(layer, x)
    out_ref = nvfp4_utils.apply_nvfp4_linear(NvFp4LinearBackend.VLLM_CUTLASS, layer, x)
    # Both go through cutlass_scaled_fp4_mm with identical inputs, so the
    # outputs must be bit-equal.
    assert torch.equal(out_nv, out_ref), "fallback must be exact (no-op)"


def test_apply_3d_input_preserves_shape(cuda_available):
    """vLLM passes [batch, seqlen, hidden] tensors; the adapter must reshape
    correctly and return the same leading shape."""
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    N, K = 4096, 4096
    layer = _make_layer(N, K, dtype=dtype, device=dev)

    x = torch.randn(2, 64, K, dtype=dtype, device=dev) * 0.05  # M=128 total
    out = apply_nvfp4r_linear(layer, x)
    assert out.shape == (2, 64, N)
