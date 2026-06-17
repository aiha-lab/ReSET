"""nvfp4r — NVFP4 W4A4 inference kernels for Blackwell (B200).

End goal: drop into vLLM / TRT-LLM and replace the existing W4A4 NVFP4 path,
including (eventually) fusing the surrounding activation-quantization and
block-scale kernels into our matmul ops.

``vllm_integration`` is intentionally NOT imported eagerly so this
package can still be imported when vLLM is unavailable. Pull it in
explicitly with ``from nvfp4r import vllm_integration``.
"""

from __future__ import annotations

import os

import torch  # noqa: F401  (load libtorch first so _C's shared-lib deps resolve)

from . import _C  # noqa: F401  (registers torch.ops.nvfp4r.*)
from . import _fake  # noqa: F401  (registers FakeTensor impls for torch.compile)
from .ops import gemm, gemv

__all__ = ["gemv", "gemm", "enable", "disable", "configure", "status"]
__version__ = "0.0.1"


# ---------------------------------------------------------------------------
# Public runtime API — a thin, documented layer over the NVFP4R_* env vars so
# callers configure the backend from Python instead of the environment.
# ---------------------------------------------------------------------------
def enable(backend: str = "nvfp4r") -> None:
    """Route vLLM's NVFP4 linear layers through the ``nvfp4r`` kernels.

    Equivalent to ``VLLM_NVFP4_GEMM_BACKEND=nvfp4r``. Call **before**
    constructing the vLLM engine (the backend is read at weight load).
    """
    os.environ["VLLM_NVFP4_GEMM_BACKEND"] = str(backend)


def disable() -> None:
    """Revert to vLLM's stock NVFP4 (CUTLASS) path."""
    os.environ.pop("VLLM_NVFP4_GEMM_BACKEND", None)


def configure(
    *,
    gemv_max_m: int | None = None,
    enable_gemm: bool | None = None,
    gemm_pad_max_m: int | None = None,
    fallback_backend: str | None = None,
) -> None:
    """Tune the nvfp4r runtime knobs from Python (overrides the ``NVFP4R_*``
    env defaults). Call before the first forward pass.

    Args:
        gemv_max_m: max decode batch routed to the small-``M`` GEMV.
        enable_gemm: use the tensor-core GEMM path for prefill / mid-``M``.
        gemm_pad_max_m: upper ``M`` for the padded-GEMM decode path.
        fallback_backend: CUTLASS fallback provider (``"flashinfer"``/``"cutlass"``).
    """
    from . import vllm_integration as _vi

    if gemv_max_m is not None:
        _vi._GEMV_MAX_M = int(gemv_max_m)
    if enable_gemm is not None:
        _vi._ENABLE_GEMM_PATH = bool(enable_gemm)
    if gemm_pad_max_m is not None:
        _vi._GEMM_PAD_MAX_M = int(gemm_pad_max_m)
    if fallback_backend is not None:
        _vi._FALLBACK_BACKEND = str(fallback_backend).lower()


def status() -> dict:
    """Return the active nvfp4r configuration as a dict (for logging / repr)."""
    from . import vllm_integration as _vi

    return {
        "active": os.environ.get("VLLM_NVFP4_GEMM_BACKEND") == "nvfp4r",
        "gemv_max_m": _vi._GEMV_MAX_M,
        "enable_gemm": _vi._ENABLE_GEMM_PATH,
        "gemm_pad_max_m": _vi._GEMM_PAD_MAX_M,
        "fallback_backend": _vi._FALLBACK_BACKEND,
    }
