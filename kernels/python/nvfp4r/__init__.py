"""nvfp4r — NVFP4 W4A4 inference kernels for Blackwell (B200).

End goal: drop into vLLM / TRT-LLM and replace the existing W4A4 NVFP4 path,
including (eventually) fusing the surrounding activation-quantization and
block-scale kernels into our matmul ops.

``vllm_integration`` is intentionally NOT imported eagerly so this
package can still be imported when vLLM is unavailable. Pull it in
explicitly with ``from nvfp4r import vllm_integration``.
"""

from __future__ import annotations

import torch  # noqa: F401  (load libtorch first so _C's shared-lib deps resolve)

from . import _C  # noqa: F401  (registers torch.ops.nvfp4r.*)
from . import _fake  # noqa: F401  (registers FakeTensor impls for torch.compile)
from .ops import gemm, gemv

__all__ = ["gemv", "gemm"]
__version__ = "0.0.1"
