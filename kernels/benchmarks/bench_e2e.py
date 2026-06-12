"""E2E latency/throughput sweep: BF16 vs vllm-cutlass vs flashinfer vs nvfp4r.

Supports batch sizes > 1.  One backend per invocation; the companion shell
script runs all four across all batch sizes.

Usage:
  CUDA_VISIBLE_DEVICES=7 python bench_e2e.py \
      --backend nvfp4r \
      --batch-sizes 1 2 4 8 \
      --output-lens 1024 2048 4096 8192 16384 \
      --iters 3 \
      --out results/e2e_nvfp4r.jsonl

Reported metrics (per row):
  ms_per_tok   — wall_s * 1000 / output_len  (per-step decode latency)
  tok_per_s    — batch_size * output_len / wall_s  (aggregate throughput)
"""

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import torch

MODELZOO = Path(
    os.environ.get(
        "MODELZOO",
        "models",
    )
)

MODEL_PAIRS = {
    "8b":  ("Qwen3-8B",  "Qwen3-8B-nvfp4-real"),
    "14b": ("Qwen3-14B", "Qwen3-14B-nvfp4-real"),
    "32b": ("Qwen3-32B", "Qwen3-32B-nvfp4-real"),
}


def build_llm(backend: str, max_model_len: int, max_num_seqs: int, model_size: str,
              gpu_mem_util: float = 0.90):
    from vllm import LLM

    bf16_name, nvfp4_name = MODEL_PAIRS[model_size]
    bf16_model  = str(MODELZOO / bf16_name)
    nvfp4_model = str(MODELZOO / nvfp4_name)

    kwargs = dict(
        dtype="bfloat16",
        enforce_eager=False,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    if backend == "bf16":
        return LLM(model=bf16_model, **kwargs)

    quant_env = {
        "cutlass":       "cutlass",
        "flashinfer":    "flashinfer-cutlass",
        "nvfp4r":        "nvfp4r",
    }[backend]
    os.environ["VLLM_NVFP4_GEMM_BACKEND"] = quant_env
    if backend == "nvfp4r":
        os.environ.setdefault("NVFP4R_ENABLE_GEMM", "1")
    return LLM(model=nvfp4_model, quantization="modelopt_fp4", **kwargs)


def run_once(llm, prompts: list[str], output_len: int, iters: int) -> dict:
    from vllm import SamplingParams

    bs = len(prompts)
    params = SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True)

    # warmup
    for _ in range(1):
        llm.generate(prompts, params, use_tqdm=False)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = llm.generate(prompts, params, use_tqdm=False)
        times.append(time.perf_counter() - t0)

    times.sort()
    wall = times[len(times) // 2]
    total_toks = sum(len(r.outputs[0].token_ids) for r in out)

    return {
        "wall_s":     round(wall, 4),
        "ms_per_tok": round(wall / output_len * 1000, 3),   # per-step latency
        "tok_per_s":  round(total_toks / wall, 2),           # aggregate throughput
        "total_toks": total_toks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",      required=True,
                    choices=["bf16", "cutlass", "flashinfer", "nvfp4r"])
    ap.add_argument("--model",        default="8b",
                    choices=list(MODEL_PAIRS.keys()),
                    help="Model size: 8b / 14b / 32b")
    ap.add_argument("--input-len",    type=int, default=512)
    ap.add_argument("--batch-sizes",  type=int, nargs="+", default=[1])
    ap.add_argument("--output-lens",  type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--iters",        type=int, default=3)
    ap.add_argument("--out",          default=None,
                    help="JSONL output file (appended)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90,
                    help="vLLM gpu_memory_utilization (default 0.90)")
    args = ap.parse_args()

    max_num_seqs  = max(args.batch_sizes) + 4
    max_model_len = args.input_len + max(args.output_lens) + 64

    # Resume: skip (bs, olen) pairs already recorded as "ok" in the output file.
    done: set[tuple[int, int]] = set()
    if args.out and Path(args.out).exists():
        with open(args.out) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _r = json.loads(_line)
                    if _r.get("status") == "ok":
                        done.add((_r["batch_size"], _r["output_len"]))
                except Exception:
                    pass
        if done:
            print(f"[resume] skipping {len(done)} already-completed (bs, olen) pairs", flush=True)

    print(f"backend={args.backend}  model={args.model}  input_len={args.input_len}  "
          f"batch_sizes={args.batch_sizes}  output_lens={args.output_lens}  "
          f"iters={args.iters}  gpu_mem_util={args.gpu_mem_util}", flush=True)
    print(f"max_model_len={max_model_len}  max_num_seqs={max_num_seqs}", flush=True)

    print("\nLoading model...", flush=True)
    t0 = time.time()
    llm = build_llm(args.backend, max_model_len, max_num_seqs, args.model,
                    gpu_mem_util=args.gpu_mem_util)
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)

    base_prompt = "A " * args.input_len
    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for bs in args.batch_sizes:
        prompts = [base_prompt] * bs
        print(f"── batch_size={bs} ──")
        print(f"{'output_len':>12}  {'wall_s':>8}  {'ms/tok':>8}  {'tok/s':>10}  {'status':>6}")
        print("-" * 62)
        for olen in args.output_lens:
            if (bs, olen) in done:
                print(f"{olen:>12}  {'(skip)':>8}  {'':>8}  {'':>10}  {'skip':>6}", flush=True)
                continue
            try:
                r = run_once(llm, prompts, olen, args.iters)
                row = {
                    "backend":    args.backend,
                    "input_len":  args.input_len,
                    "batch_size": bs,
                    "output_len": olen,
                    "status":     "ok",
                    **r,
                }
                print(f"{olen:>12}  {r['wall_s']:>8.3f}  {r['ms_per_tok']:>8.3f}  "
                      f"{r['tok_per_s']:>10.2f}  {'ok':>6}", flush=True)
            except Exception as e:
                is_oom = (
                    isinstance(e, torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(e).lower()
                    or "cuda oom" in str(e).lower()
                )
                status = "OOM" if is_oom else f"ERR:{type(e).__name__}"
                torch.cuda.empty_cache()
                row = {
                    "backend":    args.backend,
                    "input_len":  args.input_len,
                    "batch_size": bs,
                    "output_len": olen,
                    "status":     status,
                    "wall_s":     None,
                    "ms_per_tok": None,
                    "tok_per_s":  None,
                    "total_toks": None,
                }
                print(f"{olen:>12}  {'':>8}  {'':>8}  {'':>10}  {status:>6}", flush=True)
            if out_path:
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                n_written += 1
        print()

    if out_path:
        print(f"Wrote {n_written} rows → {args.out}")


if __name__ == "__main__":
    main()
