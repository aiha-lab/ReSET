<h1 align="center">ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware Temperature Scaling</h1>

Official implementation. NVFP4 (W4A4) execution on NVIDIA Blackwell cuts the cost
of large reasoning models, but applied naively it degrades accuracy and leaves
small-batch decode latency on the table. ReSET fixes both — two components that
work independently:

|  |  |
|--|--|
| **[`reset/`](reset/)** | Step-aware entropy-based decoding-temperature scaling. A drop-in vLLM logits processor — **+~2 pts** over the NVFP4 baseline, no extra forward passes. |
| **[`kernels/`](kernels/)** | `nvfp4r` — CUDA-core small-`M` NVFP4 W4A4 decode kernels for Blackwell — **2.5×** kernel-level and **~2×** end-to-end decode speedup. |

## ReSET decoding

Per token with entropy `H_t`, using the running mean `H̄` of token entropies and
the within-step estimate `Ĥ_step` (steps split on `\n\n`):

```
T_t = T_low   if  H_t <  τ_t           τ_t = τ_0      (confident step:  Ĥ_step ≤ H̄)
T_t = T_high  if  H_t ≥  τ_t           τ_t = Ĥ_step   (uncertain step:  Ĥ_step > H̄)
```

```bash
cd reset && pip install -r requirements.txt && pip install -e .
python quantize.py --model Qwen/Qwen3-8B --output Qwen3-8B-nvfp4   # HF -> NVFP4
python run_reset.py --model Qwen3-8B-nvfp4 --task aime120 --t-low 0.1 --tau0 0.5505
```

Tasks: `aime120`, `gpqa_diamond`, `livecodebench`; per-model hyperparameters in
`reset/configs/hparams.json`. → **[`reset/README.md`](reset/README.md)**

## nvfp4r kernels

A CUDA-core NVFP4 GEMV for latency-critical small-`M` decode, where the
Tensor-Core `tcgen05.mma` tile sits ≤6.25% occupied at M≤8. Registered under
`torch.ops.nvfp4r.*` — `gemv` (decode) and `gemm` (prefill / large-`M`).

```bash
cd kernels && pip install -e . --no-build-isolation   # Blackwell (sm_100a), CUDA 12.8+
pytest tests/
```

A drop-in vLLM linear adapter ships in `python/nvfp4r/vllm_integration.py`.
→ **[`kernels/README.md`](kernels/README.md)**

## License

See [LICENSE](LICENSE).

<sub>_The code is being tidied up for public release, so some rough edges may remain; we are actively reviewing it and will keep it updated._</sub>
