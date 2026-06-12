"""Python-facing wrappers around `torch.ops.nvfp4r.*`.

These wrappers are convenience-only. vLLM / TRT-LLM integrations should call
`torch.ops.nvfp4r.<op>(...)` directly so the calls remain traceable through
``torch.compile`` and FX without going through Python.
"""

from __future__ import annotations

from typing import Optional

import torch


def _ensure_3d(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dim() == 3:
        return t
    if t.dim() == 2:
        return t.unsqueeze(0)
    raise ValueError(f"{name}: expected 2D or 3D tensor, got {t.dim()}D")


def gemv(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """NVFP4 W4A4 batched GEMV: ``y = alpha * W @ x``.

    All tensors are pre-quantized:

    * ``weight``       : ``[L, M, K/2]`` ``uint8`` (FP4 E2M1, two values per byte)
    * ``weight_scale`` : ``[L, M, K/16]`` ``uint8`` (FP8 E4M3 per-block scale)
    * ``x``            : ``[L, N_pad, K/2]`` ``uint8`` (only the leading row is read;
      ``N_pad`` reflects the engine's padded N dimension)
    * ``x_scale``      : ``[L, N_pad, K/16]`` ``uint8``
    * ``out``          : optional ``[L, M]`` ``float16`` or ``bfloat16`` buffer
    * ``alpha``        : optional FP32 epilogue scalar; for vLLM-style NVFP4
      callers, set this to ``input_global_scale * weight_global_scale`` so the
      result is directly the dequantised value -- saves a separate post-scale
      kernel in the adapter. Defaults to 1.0 (legacy bit-exact behaviour).

    The leading ``L`` axis can be used to batch multiple decode tokens against
    a *broadcast* weight (``stride_l == 0``). The kernel issues ``L * M / RPCTA``
    CTAs in parallel, so this is a true batched-GEMV path for small decode M.
    """
    w = _ensure_3d(weight, "weight")
    ws = _ensure_3d(weight_scale, "weight_scale")
    a = _ensure_3d(x, "x")
    a_s = _ensure_3d(x_scale, "x_scale")
    if out is None:
        L, M, _ = w.shape
        out = torch.empty((L, M), dtype=torch.float16, device=w.device)
    torch.ops.nvfp4r.gemv(w, ws, a, a_s, out, alpha)
    return out



def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    C: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """NVFP4 W4A4 GEMM (prefill path): ``C = alpha * (A @ B^T)``.

    All shapes are NVFP4-packed (``uint8`` for both data and scales). The
    launch-config table inside the kernel currently only handles ``K`` values
    in {256, 512, 1536, 2048, 2304, 7168, 16384} and aligned ``M``/``N``.
    Caller must pre-allocate ``C`` (``float16`` *or* ``bfloat16`` of shape
    ``[M, N]``); the kernel writes into it in-place and returns the
    (possibly transposed) view.

    ``alpha`` is folded into the epilogue store (single FMA per output
    element), so callers do not need a trailing ``torch.mul``.

    The underlying kernel returns ``C.view([N, M, 1]).transpose(0, 1)``
    when ``C_N_MAJOR=false``; squeeze the trailing unit dim so callers always
    get a plain 2D ``[M, N]`` tensor.
    """
    M, N = C.shape
    torch.ops.nvfp4r.gemm(A, B, SFA, SFB, C, alpha)
    return C.view(N, M).transpose(0, 1)
