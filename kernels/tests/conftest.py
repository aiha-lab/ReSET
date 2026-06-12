"""Shared NVFP4 utilities for tests and benchmarks."""

from __future__ import annotations

import math

import pytest
import torch


_FP4_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fp8_e4m3_decode(byte: torch.Tensor) -> torch.Tensor:
    """Decode FP8 E4M3 (no NaN/Inf saturation) using PyTorch's native cast."""
    return byte.view(torch.float8_e4m3fn).to(torch.float32)


def dequant_nvfp4(
    packed: torch.Tensor,    # [..., K/2] uint8
    scale: torch.Tensor,     # [..., K/16] uint8 (FP8 E4M3)
    block: int = 16,
) -> torch.Tensor:
    """Decode an NVFP4 tensor to float32. Last dim becomes K (= 2 * K/2)."""
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8
    assert packed.shape[-1] * 2 == scale.shape[-1] * block, (
        "shape mismatch: packed last-dim*2 must equal scale last-dim * block_size"
    )

    table = _FP4_E2M1.to(packed.device)
    lo = table[(packed & 0xF).long()]
    hi = table[(packed >> 4).long()]

    out = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    s = _fp8_e4m3_decode(scale).to(out.device)
    out = out.view(*out.shape[:-1], -1, block) * s.unsqueeze(-1)
    return out.reshape(*out.shape[:-2], -1)


def reference_gemv(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference NVFP4 GEMV: dequant in FP32 and matmul, then cast to FP16."""
    w = dequant_nvfp4(weight, weight_scale)            # [L, M, K]
    a = dequant_nvfp4(x, x_scale)                      # [L, N_pad, K]
    a0 = a[:, 0:1, :]                                   # [L, 1, K]
    y = torch.einsum("lmk,lnk->lmn", w, a0).squeeze(-1)
    return y.to(torch.float16)


def random_nvfp4_tensor(
    shape_outer: tuple[int, ...],
    K: int,
    *,
    device: str = "cuda",
    seed: int = 0,
    scale_max_byte: int = 0x38,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random uint8 NVFP4 (data, scale) pair with the given shape.

    ``shape_outer`` is the shape *before* the K dimension (e.g. ``(L, M)``).

    We bound the FP8 E4M3 scale below ~1.0 so that the ground-truth
    sum across K stays inside FP16 range; otherwise random tests overflow
    inside the kernel's FP16 accumulator and produce NaNs that are not real
    correctness violations.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    K_half = K // 2
    K_sf = K // 16
    data = torch.randint(0, 256, (*shape_outer, K_half), dtype=torch.uint8,
                         device=device, generator=g)
    scale = torch.randint(1, scale_max_byte, (*shape_outer, K_sf), dtype=torch.uint8,
                          device=device, generator=g)
    return data, scale


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for nvfp4r tests")
    return True
