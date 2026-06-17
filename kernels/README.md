# nvfp4r — details

Overview and build are in the [top-level README](../README.md). This document is
the detailed reference.

## Layout

```
kernels/
├── csrc/                  C++ glue (TORCH_LIBRARY bindings, public API)
├── cuda/
│   ├── gemv/              W4A4 NVFP4 GEMV (small-M decode)
│   └── gemm/              W4A4 NVFP4 GEMM (prefill / large-M)
├── python/nvfp4r/         torch.ops.nvfp4r.* wrappers + vllm_integration.py
├── tests/                 numerical correctness vs reference
└── benchmarks/            latency / throughput micro-benches
```

## NVFP4 tensor layout (W4A4, block size 16)

For an `[M, K]` matrix:

- data:   `[M, K/2]` `uint8` — two FP4 (E2M1) values packed per byte
- scales: `[M, K/16]` `uint8` — one FP8 (E4M3) scale per 16-element block

Each length-16 block `b` dequantizes as `s_W_b · s_x_b · (W_b_fp4 · x_b_fp4)`,
with the scale product folded into a single FP16 broadcast.

## Ops

All ops are registered under `torch.ops.nvfp4r.*`; the Python wrappers in
`python/nvfp4r/ops.py` are convenience only. Out-variant schema
(`Tensor(a!) out -> ()`): the caller allocates the destination and the kernel
mutates it in place.

- **`gemv(weight, weight_scale, x, x_scale, out, alpha)`** — `[L, M, K/2]`
  weight, `[L, N_pad, K/2]` activation (leading row consumed). The `L` axis
  batches `L` decode tokens that **share one broadcast weight** (`stride_l = 0`):
  each weight tile is streamed once and reused across the tokens.
- **`gemm(A, B, SFA, SFB, C, alpha)`** — prefill/large-M. `K` must be in the
  in-kernel dispatch table (see the `switch (K_full)` in `cuda/gemm/gemm_nvfp4.cu`);
  `M`/`N` must be tile-aligned. `C` is caller-allocated `float16` **or** `bfloat16`.

`alpha` is folded into the epilogue store; for vLLM NVFP4 callers set it to
`input_global_scale * weight_global_scale` (default `1.0` is bit-exact legacy).

## Use in vLLM

Activate the backend from Python — call before building the vLLM engine:

```python
import nvfp4r
nvfp4r.enable()                      # == VLLM_NVFP4_GEMM_BACKEND=nvfp4r
nvfp4r.configure(gemv_max_m=2)        # tune runtime knobs (optional)
nvfp4r.status()                      # -> {'active': True, 'gemv_max_m': 2, ...}
```

No plugin registration or model rewrite: vLLM's NVFP4 dispatcher branches into
`nvfp4r` once activated. The equivalent env var (`VLLM_NVFP4_GEMM_BACKEND=nvfp4r`)
still works. `configure()` overrides the `NVFP4R_*` env defaults:
`fallback_backend`, `gemv_max_m`, `gemm_pad_max_m`, `enable_gemm`,
supported in both eager and CUDA-graph (piecewise) execution.

## Build notes

- Targets `sm_100a` (B200); requires CUDA 12.8+ and a matching PyTorch CUDA build.
- `NVFP4R_GEMV_ONLY=1` builds only the GEMV kernel and links a stub for `gemm`
  (for hosts whose `ptxas` cannot assemble the `tcgen05.mma.block16` path).
