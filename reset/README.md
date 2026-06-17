# ReSET — details

Overview, install, and quick start are in the [top-level README](../README.md).
This document is the detailed reference.

## Reproduce from scratch

```bash
# 1. Quantize an HF model to NVFP4 (modelopt). Produces an HF checkpoint.
reset-quantize --model Qwen/Qwen3-8B --output Qwen3-8B-nvfp4

# 2. (optional) Re-derive tau_0 for a new model. Paper values are in
#    configs/hparams.json, so this is only needed off-table. Per the paper,
#    tau_0 = 80th-pct NVFP4 token entropy over 5 random NuminaMath-1.5 problems
#    (the dump_entropy.py default; held out from the eval benchmarks).
python dump_entropy.py --model Qwen3-8B-nvfp4 --out nvfp4.jsonl
python calibrate.py --nvfp4 nvfp4.jsonl                      # tau_0 = NVFP4 80th percentile

# 3. Evaluate (per-model t_low / tau_0 in configs/hparams.json).
#    Add --backend nvfp4r to use the CUDA-core kernels.
reset --model Qwen3-8B-nvfp4 --task aime120 \
    --t-high 1.0 --t-low 0.1 --tau0 0.5505 --window 32

# overhead microbenchmark (the ~1.5% per-step cost)
python bench_overhead.py
# observation figure (Sec. 3): BF16 vs NVFP4 token-entropy distributions
python analysis/plot_entropy_distribution.py --bf16-path bf16.jsonl --nvfp4-path nvfp4.jsonl
```

`configs/hparams.json` holds the paper's per-model `t_low` / `tau_0` (T_high=1.0,
w=32 throughout).

**Baselines.** The PTQ baselines compared in the paper (RTN, BRQ, 4/6, MR-GPTQ)
are *not* bundled here — 4/6 and MR-GPTQ (QuTLASS) are separate projects with
their own licenses; use their upstream repositories.

## Algorithm

ReSET (internally the policy in `reset/logits_processor.py::ReSETRequest`) has
two parts:

- **Step-Aware Threshold (SAT).** A reasoning step is uncertain iff its online
  step-entropy estimate `Ĥ_step` exceeds the running global mean `H̄` of all
  token entropies seen so far. In confident steps the token-entropy threshold is
  the global `τ_0`; in uncertain steps it is the step-relative `Ĥ_step`. The gate
  adds no sweepable hyperparameter — `H̄` is tracked online.
- **Hybrid Step-entropy Estimator (HSE).** `Ĥ_step(t)` is the within-step running
  average once the step has at least `w` tokens, and a size-`w` sliding-window
  average (spanning the step boundary) during the first `w` tokens of a step —
  trading a small boundary bias for lower variance early in the step.

Steps are delimited by double-newline (`"\n\n"`) tokens: a boundary is hit when
the last token decodes to a string containing `"\n\n"`, or the last two tokens
both decode to `"\n"`. `get_newline_token_ids()` scans the vocabulary for these.

## Hyperparameters

| flag       | symbol  | paper value | notes |
|------------|---------|-------------|-------|
| `--t-high` | `T_high`| `1.0`       | Restores diversity NVFP4 removes in uncertain steps; improves accuracy as it grows 0.6→1.0. |
| `--t-low`  | `T_low` | calibrated  | Selected per (model, task) on a held-out split. |
| `--tau0`   | `τ_0`   | calibrated  | 80th-percentile token entropy on the calibration split. |
| `--window` | `w`     | `32`        | HSE window / step-transition init length. |

`τ_0` is calibrated per model from the entropy percentile on a small held-out
set (the paper uses five NuminaMath problems); `T_low` is chosen on a held-out
split disjoint from evaluation.

## Use in vLLM

```python
from vllm import LLM, SamplingParams
from reset import ReSETAdapter, get_newline_token_ids

llm = LLM(model="path/to/Qwen3-8B-nvfp4", quantization="modelopt_fp4",
          logits_processors=[ReSETAdapter])
nl_ids, dnl_ids = get_newline_token_ids(llm.get_tokenizer())

params = SamplingParams(temperature=1.0, max_tokens=32768, extra_args={
    "reset": True, "t_high": 1.0, "t_low": 0.3, "tau_0": 0.63, "window": 32,
    "reset_nl_ids": nl_ids, "reset_dnl_ids": dnl_ids,
})
```

## `extra_args` reference

Keep `SamplingParams.temperature = 1.0` — the processor applies the effective
temperature itself (otherwise it double-scales).

| key             | type        | meaning |
|-----------------|-------------|---------|
| `reset`         | `bool`      | Must be truthy to activate. |
| `t_high`        | `float`     | `T_high`. |
| `t_low`         | `float`     | `T_low`. |
| `tau_0`         | `float`     | `τ_0` (also accepts `h_threshold`). |
| `window`        | `int`       | `w` (also accepts `saat_window`). |
| `reset_nl_ids`  | `list[int]` | Single-newline token ids (from `get_newline_token_ids`). |
| `reset_dnl_ids` | `list[int]` | Double-newline token ids. |

## Benchmarks

`run_reset.py` reports average accuracy over `--n-samples` samples per problem
(the paper averages 8 seeds, top-p 0.95, `max_tokens` 32k).

| `--task`        | benchmark                       | type | source |
|-----------------|---------------------------------|------|--------|
| `aime120`       | AIME-120 (AIME 2022–2025 union) | math | `xiaoyuanliu/AIME90` + `yentinglin/aime_2025` |
| `aime90`        | AIME 2022–2024                  | math | `xiaoyuanliu/AIME90` |
| `aime25`        | AIME 2025                       | math | `yentinglin/aime_2025` |
| `gpqa_diamond`  | GPQA-Diamond                    | mcq  | `Idavidrein/gpqa` (gated; HF login required) |
| `livecodebench` | LiveCodeBench                   | code | `livecodebench/code_generation_lite` |

`--baseline --base-temp 0.6` decodes at a single fixed temperature (the NVFP4 /
BF16 baseline) instead of ReSET.

> ⚠️ `--task livecodebench` executes model-generated Python in a subprocess to
> check test cases. Run it only in a sandboxed / disposable environment.

## Files

```
reset/
├── reset/
│   ├── __init__.py
│   └── logits_processor.py   # ReSETRequest (SAT + HSE), ReSETAdapter, helpers
├── run_reset.py              # evaluation harness (math / mcq / code scoring)
├── quantize.py               # HF model -> NVFP4 (modelopt) checkpoint
├── dump_entropy.py           # per-token entropy dump (-> calibrate / plot)
├── calibrate.py              # tau_0 = 80th-pct token entropy from the dump
├── bench_overhead.py         # per-token overhead microbenchmark
├── configs/hparams.json      # per-model t_low / tau_0 (paper values)
├── analysis/
│   └── plot_entropy_distribution.py   # Sec. 3 BF16-vs-NVFP4 entropy figure
├── requirements.txt
└── pyproject.toml
```
