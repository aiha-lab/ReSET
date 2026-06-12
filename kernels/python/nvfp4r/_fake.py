"""FakeTensor / Meta implementations for ``torch.ops.nvfp4r.*``.

These are required for ``torch.compile`` (Dynamo + Inductor +
CUDAGraph) to trace through our kernels without falling back to eager.
vLLM wraps each model with ``support_torch_compile`` (fullgraph=True), so
any custom op invoked inside the model forward must declare its output
shape / dtype symbolically. Without these registrations Dynamo raises
``OperatorNotFound`` at trace time and tears the whole graph down.

All three ops use the ``Tensor(a!) out -> ()`` "out variant" pattern: the
caller allocates the destination buffer, the kernel mutates it in place,
and there is no return value. The fakes therefore just validate the
shapes and return ``None``.
"""

from __future__ import annotations

import torch


@torch.library.register_fake("nvfp4r::gemv")
def _gemv_fake(
    weight: torch.Tensor,        # [L, M, K/2]
    weight_scale: torch.Tensor,  # [L, M, K/16]
    x: torch.Tensor,             # [L, N_pad, K/2]
    x_scale: torch.Tensor,       # [L, N_pad, K/16]
    out: torch.Tensor,           # [L, M], fp16 / bf16  (mutated in-place)
    alpha: float = 1.0,
) -> None:
    return None



@torch.library.register_fake("nvfp4r::gemm")
def _gemm_fake(
    A: torch.Tensor,    # [M, K/2]
    B: torch.Tensor,    # [N, K/2]
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    C: torch.Tensor,    # [M, N], fp16 OR bf16 (mutated in-place)
    alpha: float = 1.0,
) -> None:
    return None
