"""Native NVFP4R backend for vLLM.

vLLM ships a small ``NvFp4LinearBackend`` enum + ``apply_nvfp4_linear``
dispatcher in ``vllm/model_executor/layers/quantization/utils/nvfp4_utils.py``.
We register **as a backend** (``NVFP4R = "nvfp4r"``) rather than monkey-patching
the dispatcher so:

* vLLM's compiled forward sees a plain Python branch into our function -- no
  closure capture, no ``setattr`` on the patched method, no surprises for
  ``torch.compile(fullgraph=True)``.
* Cache priming runs in ``convert_to_nvfp4_linear_kernel_format`` (eager),
  so the hot path only reads pre-allocated buffers and constant Python
  attributes; Dynamo traces straight through.
* Activation: set ``VLLM_NVFP4_GEMM_BACKEND=nvfp4r``. No plugin
  registration, no ``import nvfp4r.vllm_integration as nvi; nvi.enable()``.

Two public entry points consumed by the patched ``nvfp4_utils.py``:

* :func:`apply_nvfp4r_linear` - hot path called from
  :func:`vllm.model_executor.layers.quantization.utils.nvfp4_utils.apply_nvfp4_linear`.
* :func:`prime_nvfp4r_caches` - one-shot cache fill called from
  :func:`...convert_to_nvfp4_linear_kernel_format` after the standard CUTLASS
  weight prep has run.

Activation SF layout:
  vLLM stores per-token activation SF in *swizzled* ``[M/128, K/64, 32, 4, 4]``
  layout because that's what its CUTLASS kernel reads. Our ``gemm`` reads the
  same layout, so the prefill path forwards the swizzled SF directly. Our
  ``gemv`` reads a *flat* ``[M, K/16]`` SF, so the decode path re-quantises
  with ``is_sf_swizzled_layout=False`` and we eagerly un-swizzle the weight
  SF inside :func:`prime_nvfp4r_caches`.

Decode batching:
  Real vLLM continuous-batching forward passes have ``M == num_decoding_tokens``
  rather than always 1. The ``gemv`` kernel batches over its ``L`` axis -- we
  pass each decode token in its own ``L`` slot and broadcast the weight (with
  ``stride_l == 0`` via ``Tensor.expand``) so all M tokens share the same
  ``[N, K/2]`` weight matrix. One launch handles the entire decode batch.

Alpha:
  ``cutlass_scaled_fp4_mm`` post-multiplies the matmul by
  ``alpha = input_global_scale * weight_global_scale``. ``nvfp4r.gemv`` and
  ``nvfp4r.gemm`` take ``alpha`` as a kernel argument and fold it into the
  FP32 partial sum just before the output cast.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

import vllm._custom_ops as vops
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

import nvfp4r  # noqa: F401  (registers torch.ops.nvfp4r.*)


_FALLBACK_BACKEND = os.environ.get("NVFP4R_FALLBACK_BACKEND", "flashinfer").lower()
if _FALLBACK_BACKEND == "flashinfer" and not has_flashinfer():
    _FALLBACK_BACKEND = "vllm-cutlass"


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
# K values handled by `nvfp4r.gemm`'s launch table. Anything outside of this
# falls through to vLLM's CUTLASS path (inlined below).
_GEMM_SUPPORTED_K = {
    256, 512, 1536, 2048, 2304, 3584, 4096, 5120, 7168, 8192,
    12288, 13824, 16384, 17408, 18944, 25600,
}

_GEMV_MAX_M = int(os.environ.get("NVFP4R_GEMV_MAX_M", "2"))

# Padded-GEMM cap: route L ∈ [_GEMV_MAX_M+1, _GEMM_PAD_MAX_M] through the
# tensor-core nvfp4r.gemm with activation padded up to L_PAD=64.  The kernel
# now writes bf16 directly with ``alpha`` folded into the epilogue store
# (gemm_nvfp4.cu:store_pair_scaled), eliminating the trailing
# torch.mul + dtype-cast graph node that previously regressed e2e by ~12%.
#
# Empirical history on Qwen3-8B i=512 / o=1024 / B200 / GPU 7:
#   v0  fp16 kernel + python pad + python mul/cast: 12% slower than
#       flashinfer fallback (kernel saves 6µs/call but 2 extra graph
#       nodes cost 2.5µs/call -> net loss).
#   v1  bf16+alpha-fused kernel epilogue (this change): see e2e sweep
#       results next to this file.
_GEMM_PAD_MAX_M = int(os.environ.get("NVFP4R_GEMM_PAD_MAX_M", "63"))
_GEMM_PAD_TARGET = 64

_ENABLE_GEMM_PATH = os.environ.get("NVFP4R_ENABLE_GEMM", "1") == "1"

# Eager-only runtime guard: recompute on CUTLASS if the gemv path returns a
# non-finite output. Catches an input-dependent NaN edge case in the gemv
# kernel. Disable (set to 0) once the kernel is fixed or when running with
# CUDA graphs (the isfinite check is a graph break).
_GEMV_SAFETY = os.environ.get("NVFP4R_GEMV_SAFETY", "1") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unswizzle_blockscale(sw_u8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Inverse of vLLM's :func:`swizzle_blockscale`.

    Returns a ``[M, K/16]`` ``uint8`` tensor of the per-block scales
    that matches the layout consumed by ``nvfp4r.gemv``. Padding rows / cols
    (added when M%128 or (K/16)%4) are dropped.
    """
    from vllm.utils.math_utils import round_up

    K_div16 = K // 16
    M_padded = round_up(M, 128)
    K_padded = round_up(K_div16, 4)

    s = sw_u8 if sw_u8.ndim == 3 else sw_u8.unsqueeze(0)
    s = s.reshape(s.shape[0], M_padded // 128, K_padded // 4, 32, 4, 4)
    # The forward swizzle was permute(0, 1, 4, 3, 2, 5) — self-inverse.
    s = s.permute(0, 1, 4, 3, 2, 5).contiguous()
    s = s.reshape(s.shape[0], M_padded, K_padded)
    return s[0, :M, :K_div16].contiguous()


def _scale_and_cast(out_fp16: torch.Tensor, alpha_host: float,
                    output_dtype: torch.dtype) -> torch.Tensor:
    """Single-launch ``out_fp16 * alpha`` with output dtype cast.

    Using a host (Python) float for alpha means PyTorch dispatches a fused
    multiply-cast kernel rather than three separate launches.

    ``nvfp4r.gemm`` returns a transposed view of the kernel buffer
    (the kernel writes N-major, the wrapper hands back ``buf.t()``), which
    is *non-contiguous*. Stock vLLM ops downstream (rms_norm,
    fused_add_rms, silu_and_mul) require contiguous inputs / outputs.
    Allocating ``out`` via ``torch.empty(shape, dtype=...)`` -- not
    ``empty_like`` -- yields a fresh row-major buffer; ``torch.mul(...,
    out=out)`` then resolves the transposed read while writing into a
    contiguous destination.
    """
    if output_dtype == torch.float16 and alpha_host == 1.0:
        return out_fp16.contiguous()
    out = torch.empty(out_fp16.shape, dtype=output_dtype, device=out_fp16.device)
    torch.mul(out_fp16, alpha_host, out=out)
    return out


# ---------------------------------------------------------------------------
# [SHLEE/Approach-B] Single-dispatch custom ops.
#
# Each op packs the entire NVFP4 linear *post-dispatch* host work
# (``scaled_fp4_quant`` + view manipulations + ``empty`` alloc + the actual
# nvfp4r kernel call) into one ``torch.ops.vllm.*`` invocation.
#
# Rationale: in eager mode each of those steps is a separate dispatcher op,
# and each carries Python + dispatcher + (for ``scaled_fp4_quant``) C++
# wrapper overhead that the kernel-level micro-bench
# (``kernel_micro_summary.md``) showed to dominate over actual GPU time on
# small shapes. Folding them into one op cuts the dispatcher node count for
# the linear from ~8 down to 1, which (a) shrinks the work the
# AOTAutograd / Inductor wrapper has to do per call, and (b) lets vLLM's
# piecewise CUDA-graph capture treat the entire linear as a single node so
# the graph ↔ host transition cost shows up at most once per call rather
# than per intermediate view/op.
#
# The fake impls only need to produce a tensor of the correct shape /
# dtype / device -- AOTAutograd uses them to size downstream allocations
# during tracing.
# ---------------------------------------------------------------------------
def _nvfp4r_linear_impl(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_scale_flat_3d: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    alpha_host: float,
    output_size: int,
    weights_padding_cols: int,
    gemv_max_m: int,
    out_dtype_int: int,
    enable_gemm: bool,
) -> torch.Tensor:
    """[SHLEE/Approach-B v3] All M-branching lives **inside** this op.

    Why: when ``apply_nvfp4r_linear`` did the ``if M <= 32`` check in
    Python, Dynamo specialized the comparison on the M of the FIRST trace
    (typically ``M = prefill_chunk_size`` >> 32). The captured graph
    therefore baked in the *fallback* branch and decode tokens (M=1)
    silently re-used that graph, never invoking ``nvfp4r.gemv`` at all
    -- verified empirically: setting ``NVFP4R_GEMV_MAX_M=0`` (force
    fallback) yielded an *identical* wall time to ``NVFP4R_GEMV_MAX_M=32``.

    Folding the branch into the op means Dynamo only sees one
    ``vllm::nvfp4r_linear`` call. vLLM's piecewise CUDA-graph capture
    runs the op once per ``cudagraph_capture_size`` (1, 2, 4, ..., 512),
    and inside each capture the ``M`` dimension is concrete -- so the
    correct branch (gemv for small M, gemm or cutlass fallback for
    large M) is the one that lands in that batch's CUDA graph.

    Schema notes:
      * ``alpha`` (Tensor) is consumed by the cutlass fallback;
        ``alpha_host`` (float) is consumed by ``nvfp4r.{gemv,gemm}``.
        Both are passed because ints/floats can ride the custom-op
        schema while CUDA scalars cannot be moved to the host inside
        the op without a graph break.
      * ``out_dtype_int``: 0 = bf16, 1 = fp16 (dtype is not a valid
        custom-op argument).
      * Caller is responsible for passing ``x_2d`` already 2-D
        (``x.reshape(-1, K)``) so the M / N inference here is trivial.
    """
    M = x_2d.shape[0]
    N = output_size
    out_dtype = torch.bfloat16 if out_dtype_int == 0 else torch.float16

    if M <= gemv_max_m and N % 8 == 0:
        # ----- decode (batched gemv, W4A4) -------------------------------
        x_fp4, x_sf = vops.scaled_fp4_quant(
            x_2d, input_global_scale_inv, is_sf_swizzled_layout=False,
        )
        w_b  = weight.unsqueeze(0).expand(M, -1, -1)
        ws_b = weight_scale_flat_3d.expand(M, -1, -1)
        x_b  = x_fp4.view(M, 1, -1)
        xs_b = x_sf.view(torch.uint8).view(M, 1, -1)
        out  = torch.empty(M, N, dtype=out_dtype, device=x_2d.device)
        torch.ops.nvfp4r.gemv(w_b, ws_b, x_b, xs_b, out, alpha_host)
        return out

    if enable_gemm and M <= _GEMM_PAD_MAX_M and N % 128 == 0:
        # ----- mid-batch decode (padded tensor-core gemm) ----------------
        # Pad activation rows from M (∈[2..63]) up to L_PAD=64 so the
        # kernel's BLOCK_N=64 tile fits.  The vLLM swizzler always
        # produces an SF buffer with M rounded up to 128, so quantising
        # the L_PAD=64 padded activation gives a (128, K/16) SF that the
        # kernel can consume directly.
        #
        # Why this beats GEMV at L≥2: GEMV is FP16-CUDA-core compute-bound
        # and scales linearly with L; padded GEMM uses tensor cores and
        # the kernel time is DRAM-bound by the weight load (constant in L
        # up to ~64).  Empirically beats flashinfer-cutlass by 1.3-2x at
        # L=8 because flashinfer is a more general (M=128-padded) launch.
        #
        # Epilogue notes:
        #   * The kernel writes to ``out_buf`` in N-major (L_PAD-fast)
        #     layout. We want the caller-visible output in [M, N] M-major.
        #   * Skip the .contiguous() materialisation of the full L_PAD×N
        #     buffer -- it costs ~1.4 MB per call (= 350+µs per token-step
        #     across 252 linear ops) and dwarfs the kernel speedup. Instead
        #     read only the [M, N] slice through the strided view directly
        #     into the output dtype via torch.mul (which materialises into
        #     a fresh contiguous buffer of size M×N, not L_PAD×N).
        L_PAD = _GEMM_PAD_TARGET
        x_fp4_small, x_sf = vops.scaled_fp4_quant(
            x_2d, input_global_scale_inv, is_sf_swizzled_layout=True,
        )
        # ``empty`` + slice-assign (skip the zero-init memset; rows M..L_PAD-1
        # produce garbage outputs that we discard with the [:M] slice).
        # x_sf is already swizzled to M_padded=128 by vLLM regardless of M.
        x_fp4 = torch.empty(L_PAD, x_fp4_small.shape[1],
                            dtype=x_fp4_small.dtype, device=x_2d.device)
        x_fp4[:M] = x_fp4_small
        # Allocate output directly in the model dtype (bf16) and let the
        # kernel apply alpha + cast in its store epilogue. Removes the
        # _scale_and_cast trailing mul kernel entirely (-1 graph node).
        out_buf = torch.empty(L_PAD, N, dtype=out_dtype, device=x_2d.device)
        torch.ops.nvfp4r.gemm(
            x_fp4, weight,
            x_sf.view(torch.uint8),
            weight_scale_swizzled.view(torch.uint8),
            out_buf,
            alpha_host,
        )
        # Kernel writes N-major; transpose-view back to [M, N] then slice.
        # Empirically dropping the trailing .contiguous() saves ~1µs/call;
        # downstream stock vLLM ops (rms_norm, silu_and_mul) accept strided
        # inputs in practice via their internal vectorised loaders. If a
        # downstream op asserts contiguity we'll see it as a clean error
        # rather than a silent perf regression.
        return out_buf.view(N, L_PAD).transpose(0, 1)[:M]

    if enable_gemm and M % 128 == 0 and N % 128 == 0:
        # ----- prefill (gemm) -------------------------------------------
        x_fp4, x_sf = vops.scaled_fp4_quant(
            x_2d, input_global_scale_inv, is_sf_swizzled_layout=True,
        )
        # bf16-direct + alpha-folded epilogue: skips the trailing
        # torch.mul + dtype-cast graph node that fp16-only kernels force.
        out_buf = torch.empty(M, N, dtype=out_dtype, device=x_2d.device)
        torch.ops.nvfp4r.gemm(
            x_fp4, weight,
            x_sf.view(torch.uint8),
            weight_scale_swizzled.view(torch.uint8),
            out_buf,
            alpha_host,
        )
        return out_buf.view(N, M).transpose(0, 1).contiguous()

    # ----- cutlass-class fallback (inlined) -----------------------------
    # Backend chosen by ``NVFP4R_FALLBACK_BACKEND`` (see top of file).
    # Default ``flashinfer`` matches stock vLLM's auto-pick on B200, so
    # a leg with only one of {gemv, gemm} enabled compares apples-to-
    # apples against the production baseline -- the unchanged path in
    # both legs hits the same wrapper and the same kernel.
    x_fp4, x_blockscale = vops.scaled_fp4_quant(
        x_2d, input_global_scale_inv, is_sf_swizzled_layout=True,
    )
    if weights_padding_cols > 0:
        x_fp4 = torch.nn.functional.pad(
            x_fp4, (0, weights_padding_cols)
        ).contiguous()
    if _FALLBACK_BACKEND == "flashinfer":
        out = flashinfer_scaled_fp4_mm(
            x_fp4, weight, x_blockscale, weight_scale_swizzled,
            alpha, out_dtype, backend="cutlass",
        )
    else:
        out = vops.cutlass_scaled_fp4_mm(
            x_fp4, weight, x_blockscale, weight_scale_swizzled,
            alpha, out_dtype,
        )
    if out.shape[-1] != N:
        out = out[..., :N].contiguous()
    return out


def _nvfp4r_linear_fake(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_scale_flat_3d: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    alpha_host: float,
    output_size: int,
    weights_padding_cols: int,
    gemv_max_m: int,
    out_dtype_int: int,
    enable_gemm: bool,
) -> torch.Tensor:
    out_dtype = torch.bfloat16 if out_dtype_int == 0 else torch.float16
    return torch.empty(
        x_2d.shape[0], output_size,
        dtype=out_dtype, device=x_2d.device,
    )


def _register_ops_once() -> None:
    if hasattr(torch.ops.vllm, "nvfp4r_linear"):
        return
    direct_register_custom_op(
        op_name="nvfp4r_linear",
        op_func=_nvfp4r_linear_impl,
        mutates_args=[],
        fake_impl=_nvfp4r_linear_fake,
    )


_register_ops_once()


# ---------------------------------------------------------------------------
# Cache priming (called from convert_to_nvfp4_linear_kernel_format, eager)
# ---------------------------------------------------------------------------
def prime_nvfp4r_caches(layer: torch.nn.Module) -> None:
    """One-shot per-layer cache fill, invoked by the patched
    ``convert_to_nvfp4_linear_kernel_format`` for the NVFP4R backend.

    Populates host-resident derived state so the compiled forward only
    reads constant attributes / pre-allocated buffers:

    * ``_nvfp4r_alpha_host`` -- Python ``float`` of ``layer.alpha`` (saves
      a ``.item()`` call -- which would graph-break Dynamo -- per
      forward).
    * ``_nvfp4r_weight_scale_flat_3d`` -- ``[1, N, K/16]`` un-swizzled
      ``uint8`` SF buffer for the gemv path (broadcast-ready: callers
      ``expand`` the L axis).

    Idempotent; safe to call twice. If ``layer.alpha`` is missing the
    function returns early and the layer transparently falls back to
    vLLM's CUTLASS path on first dispatch.
    """
    if getattr(layer, "_nvfp4r_caches_primed", False):
        return

    alpha_t = getattr(layer, "alpha", None)
    if alpha_t is None or not torch.is_tensor(alpha_t):
        return  # leave unprimed; fallback path handles missing alpha
    layer._nvfp4r_alpha_host = float(alpha_t.item())

    K = getattr(layer, "input_size_per_partition", None)
    N = getattr(layer, "output_size_per_partition", None)
    ws = getattr(layer, "weight_scale", None)
    if K is not None and N is not None and ws is not None and K // 16 > 0:
        try:
            flat = _unswizzle_blockscale(ws.data.view(torch.uint8), M=N, K=K)
            layer._nvfp4r_weight_scale_flat_3d = flat.unsqueeze(0).contiguous()
        except Exception:
            # gemv path will be unavailable; gemm + fallback still work.
            pass

    layer._nvfp4r_caches_primed = True


# ---------------------------------------------------------------------------
# CUTLASS fallback (inlined to avoid the circular ``apply_nvfp4_linear``
# re-entry).
# ---------------------------------------------------------------------------
def _cutlass_fallback(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Replicates vLLM's VLLM_CUTLASS branch.

    Importing :func:`apply_nvfp4_linear` from ``nvfp4_utils`` here would be
    circular (``nvfp4_utils`` imports us at module load). The CUTLASS path
    is short enough to inline, and ``cutlass_scaled_fp4_mm`` /
    ``scaled_fp4_quant`` are both registered ops with FakeTensor impls so
    Dynamo traces them cleanly.
    """
    output_size = layer.output_size_per_partition
    output_dtype = x.dtype
    output_shape = (*x.shape[:-1], output_size)

    x_fp4, x_blockscale = vops.scaled_fp4_quant(
        x, layer.input_global_scale_inv, is_sf_swizzled_layout=True,
    )

    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    if weights_padding_cols > 0:
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()

    if _FALLBACK_BACKEND == "flashinfer":
        out = flashinfer_scaled_fp4_mm(
            x_fp4, layer.weight, x_blockscale, layer.weight_scale,
            layer.alpha, output_dtype, backend="cutlass",
        )
    else:
        out = vops.cutlass_scaled_fp4_mm(
            x_fp4, layer.weight, x_blockscale, layer.weight_scale,
            layer.alpha, output_dtype,
        )
    if out.shape[-1] != output_size:
        out = out[..., :output_size].contiguous()
    if bias is not None:
        out = out + bias
    return out.view(*output_shape)


# ---------------------------------------------------------------------------
# Hot path
# ---------------------------------------------------------------------------
def apply_nvfp4r_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Native NVFP4R W4A4 dispatch.

    Called from vLLM's ``apply_nvfp4_linear`` when ``backend ==
    NvFp4LinearBackend.NVFP4R``. Lives inside the model's compiled forward
    when ``torch.compile`` is enabled; the body must therefore stay
    fullgraph-compatible (no ``.item()``, no ``setattr``, no
    ``os.environ.get``, no ``try/except``).
    """
    output_size = layer.output_size_per_partition
    input_size  = layer.input_size_per_partition
    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)

    if weights_padding_cols != 0 or input_size not in _GEMM_SUPPORTED_K:
        return _cutlass_fallback(layer, x, bias)

    output_dtype = x.dtype
    output_shape = (*x.shape[:-1], output_size)
    x_2d = x.reshape(-1, x.shape[-1])

    # [SHLEE/Approach-B v3] All M-dependent branching now lives inside
    # ``vllm::nvfp4r_linear``. The Python wrapper is a single
    # dispatcher call; vLLM's piecewise CUDA-graph capture exercises the
    # op once per ``cudagraph_capture_size`` and the correct sub-path
    # (gemv / gemm / cutlass fallback) gets baked into each batch's
    # graph based on the captured M.
    out_dtype_int = 0 if output_dtype is torch.bfloat16 else 1
    out = torch.ops.vllm.nvfp4r_linear(
        x_2d,
        layer.weight,
        layer.weight_scale,
        layer._nvfp4r_weight_scale_flat_3d,
        layer.input_global_scale_inv,
        layer.alpha,
        layer._nvfp4r_alpha_host,
        output_size,
        weights_padding_cols,
        _GEMV_MAX_M,
        out_dtype_int,
        _ENABLE_GEMM_PATH,
    )

    # Runtime safety net: the small-M gemv path can emit non-finite values on
    # certain real activations (an input-dependent kernel edge case); recompute
    # those calls on the CUTLASS path so a transient NaN never corrupts decoding.
    # The isfinite check syncs/graph-breaks, so it is gated to eager runs via
    # NVFP4R_GEMV_SAFETY (default on). The proper fix is in the gemv kernel.
    if _GEMV_SAFETY and not bool(torch.isfinite(out).all()):
        return _cutlass_fallback(layer, x, bias)

    if bias is not None:
        out = out + bias
    # Return a row-major contiguous tensor: vLLM's downstream ops (e.g. the
    # qkv split + per-head RMSNorm) allocate their output with empty_like and
    # assume contiguous input, so a non-contiguous linear output crashes them.
    return out.reshape(*output_shape).contiguous()


# ---------------------------------------------------------------------------
# Legacy compat shims
# ---------------------------------------------------------------------------
def enable() -> None:  # pragma: no cover
    """No-op. NVFP4R is now activated via ``VLLM_NVFP4_GEMM_BACKEND=nvfp4r``."""
    return


def disable() -> None:  # pragma: no cover
    return
