"""Smoke tests for nvfp4r.gemm.

Numerical correctness against an FP32 reference would require us to
build SFA/SFB in the kernel's swizzled layout (`[M/128, K/64, 32, 4, 4]`).
That work belongs in a dedicated `test_gemm_correctness.py`; here we only
verify the op runs end-to-end on a shape covered by the launch dispatcher.
"""

from __future__ import annotations

import torch
import pytest

torch.ops.load_library  # ensure torch.ops loader is initialized
import nvfp4r  # noqa: F401  (registers torch.ops.nvfp4r.*)


# Each entry must match the in-kernel launch table (`if (K == K_) ...`).
# Shapes here are chosen so the dispatcher actually launches a kernel.
# Format: (M, N, K). Using SWAP_AB=true paths so M can stay small.
_GEMM_SHAPES = [
    (128, 256, 256),
    (128, 512, 2048),
    (128, 256, 7168),
]


@pytest.mark.parametrize("M, N, K", _GEMM_SHAPES)
def test_gemm_runs(M: int, N: int, K: int) -> None:
    dev = torch.device("cuda")
    A = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    B = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    SFA = torch.zeros(M * K // 16, dtype=torch.uint8, device=dev)
    SFB = torch.zeros(N * K // 16, dtype=torch.uint8, device=dev)
    C = torch.zeros(M, N, dtype=torch.float16, device=dev)

    out = nvfp4r.gemm(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    assert out.shape == (M, N), f"unexpected output shape {out.shape}"
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), "non-finite values in gemm output"
