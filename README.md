<p align="center">
  <img src="docs/static/logo/nvfp4r-logo.png" alt="nvfp4r" width="104"/>
</p>

<h2 align="center">
ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware Temperature Scaling
</h2>

<p align="center">
| <a href="https://aiha-lab.github.io/ReSET/"><b>Project Page</b></a> |
<a href="https://arxiv.org/abs/2606.13233"><b>Paper</b></a> |
<a href="https://github.com/aiha-lab/ReSET"><b>Code</b></a> |
</p>

**ReSET** makes NVFP4 (W4A4) reasoning on NVIDIA Blackwell both **accurate** and
**fast for latency-critical decoding**. NVFP4 cuts the cost of large reasoning
models, but applied naively it degrades accuracy and leaves small-batch decode
latency on the table. ReSET fixes both, with two components that work
independently.

## Highlights

- **+~2 pts** reasoning accuracy over the NVFP4 baseline — no extra forward passes.
- **2.5×** kernel-level decode speedup over NVFP4 in vLLM.
- **~2×** end-to-end decoding speedup over BF16.
- The first **CUDA-core NVFP4 inference path** for small-batch long decoding
  (`nvfp4r`), drop-in for vLLM.

## Core components

1. **[`reset/`](reset/) — Step-aware temperature scaling.** A drop-in vLLM logits
   processor that estimates step-level uncertainty online and adapts the decoding
   temperature from both token- and step-level entropy.
2. **[`kernels/`](kernels/) — `nvfp4r`.** CUDA-core small-`M` NVFP4 W4A4 decode
   kernels for Blackwell. At `M ≤ 8` the Tensor-Core `tcgen05.mma` tile is ≤6.25%
   occupied, so `nvfp4r` streams W4A4 weights through the CUDA cores with a
   broadcast-weight GEMV instead. Registered under `torch.ops.nvfp4r.*`.

Built on top of [vLLM](https://github.com/vllm-project/vllm).

## Quick start

### Install

```bash
cd reset      && pip install -r requirements.txt && pip install -e .
cd ../kernels && pip install -e . --no-build-isolation   # nvfp4r — Blackwell (sm_100a), CUDA 12.8+
pytest tests/                                            # optional: kernel correctness
```

### Quantize a model to NVFP4

```bash
cd ../reset
python quantize.py --model Qwen/Qwen3-8B --output Qwen3-8B-nvfp4    # HF -> NVFP4 (modelopt)
```

### Run end-to-end (NVFP4 kernels + ReSET decoding)

Set `VLLM_NVFP4_GEMM_BACKEND=nvfp4r` to run the linear projections on the
CUDA-core `nvfp4r` kernels; ReSET decoding is applied automatically.

```bash
VLLM_NVFP4_GEMM_BACKEND=nvfp4r \
python run_reset.py --model Qwen3-8B-nvfp4 --task aime120 \
    --t-low 0.1 --tau0 0.5505 --enforce-eager
```

Drop `VLLM_NVFP4_GEMM_BACKEND` to use vLLM's stock NVFP4 path. The `nvfp4r`
decode path currently runs in eager mode (`--enforce-eager`). Tasks: `aime120`,
`gpqa_diamond`, `livecodebench`; per-model `t_low` / `tau_0` in
[`reset/configs/hparams.json`](reset/configs/hparams.json). →
**[`reset/README.md`](reset/README.md)** · **[`kernels/README.md`](kernels/README.md)**

## Supported models

Qwen3 (8B / 14B / 32B) and DeepSeek-R1-Distill-Qwen (7B / 14B), in NVFP4 (W4A4).

## Citation

```bibtex
@article{lee2026reset,
  title   = {ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware Temperature Scaling},
  author  = {Lee, Sihwa and Lee, Janghwan and Yoo, Donghoon and Kim, Jae Gon and
             Ryu, Hanyul and Ryu, Soojung and Choi, Jungwook},
  journal = {arXiv preprint arXiv:2606.13233},
  year    = {2026}
}
```

## License

See [LICENSE](LICENSE).

<sub>_The code is being tidied up for public release, so some rough edges may remain; we are actively reviewing it and will keep it updated._</sub>
