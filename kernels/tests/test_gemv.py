"""Numerical correctness tests for `torch.ops.nvfp4r.gemv`."""

from __future__ import annotations

import pytest
import torch

import nvfp4r  # noqa: F401  (registers torch.ops.nvfp4r.*)
from conftest import random_nvfp4_tensor, reference_gemv


@pytest.mark.parametrize(
    "L, M, K",
    [
        # Generic / sweep coverage.
        (1, 256, 4096),
        (1, 1024, 8192),
        (1, 2048, 16384),
        (4, 512, 2048),
        (8, 512, 7168),
        (2, 4096, 4096),
        (1, 4608, 3584),    # DR-7B / Qwen2.5-7B qkv_proj
        (1, 3584, 3584),    # DR-7B o_proj
        (1, 3584, 18944),   # DR-7B down_proj
        (1, 6144, 4096),    # Qwen3-8B qkv_proj
        (1, 4096, 4096),    # Qwen3-8B o_proj
        (1, 4096, 12288),   # Qwen3-8B down_proj
    ],
)
def test_gemv_matches_reference(cuda_available, L: int, M: int, K: int) -> None:
    N_pad = 1
    # The L axis batches L decode tokens that share one weight (broadcast,
    # stride_l=0): the small-M decode path loads each weight tile once and
    # reuses it across tokens. Build a single weight and broadcast it.
    weight, weight_scale = random_nvfp4_tensor((1, M), K, seed=0)
    weight = weight.expand(L, M, K // 2)
    weight_scale = weight_scale.expand(L, M, K // 16)
    x, x_scale = random_nvfp4_tensor((L, N_pad), K, seed=1)

    y_ref = reference_gemv(weight, weight_scale, x, x_scale)
    y_ours = nvfp4r.gemv(weight, weight_scale, x, x_scale)

    assert y_ours.shape == y_ref.shape
    assert y_ours.dtype == torch.float16

    diff = (y_ours.float() - y_ref.float()).abs()
    rel = diff / (y_ref.float().abs() + 1e-3)
    max_abs = diff.max().item()
    max_rel = rel.max().item()
    assert max_rel < 5e-2 or max_abs < 1.0, (
        f"gemv mismatch: max_abs={max_abs:.4f}, max_rel={max_rel:.4f} "
        f"(L={L}, M={M}, K={K})"
    )


def test_gemv_writes_into_provided_out(cuda_available) -> None:
    L, M, K = 1, 256, 4096
    weight, weight_scale = random_nvfp4_tensor((L, M), K, seed=2)
    x, x_scale = random_nvfp4_tensor((L, 1), K, seed=3)
    out = torch.empty((L, M), dtype=torch.float16, device="cuda")
    out_ptr_before = out.data_ptr()
    ret = torch.ops.nvfp4r.gemv(weight, weight_scale, x, x_scale, out)
    assert ret is None
    assert out.data_ptr() == out_ptr_before
    # And the buffer must contain finite values (smoke check).
    assert torch.isfinite(out).all().item()
