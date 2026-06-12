#!/usr/bin/env python3
"""Dump per-token next-token entropy for a model on a small calibration set.

Writes one JSON line per prompt::

    {"idx": int, "correct": null, "n_tokens": int, "entropies": [float, ...]}

The output feeds ``calibrate.py`` (tau_0 = 80th-percentile token entropy) and
``analysis/plot_entropy_distribution.py``.

Following the paper, tau_0 is calibrated on 5 randomly sampled NuminaMath-1.5
problems (the default ``--task numinamath``), held out from the evaluation
benchmarks. The ``aime*`` tasks are provided only for the Sec. 3 observation
figure, not for calibration.

For NVFP4 checkpoints, install ``nvidia-modelopt`` so ``transformers`` can load
the quantized weights (generation runs through modelopt's emulation).

Example::

    python dump_entropy.py --model path/to/Qwen3-8B        --out bf16.jsonl
    python dump_entropy.py --model path/to/Qwen3-8B-nvfp4  --out nvfp4.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# tau_0 is calibrated on a small NuminaMath-1.5 subset (held out from the
# evaluation benchmarks). aime90/aime25 are provided for the Sec. 3 observation
# figure, NOT for calibration (they are the evaluation sets).
TASKS = {
    "numinamath": ("AI-MO/NuminaMath-1.5", "train", "problem"),
    "aime90": ("xiaoyuanliu/AIME90", "train", "problem"),
    "aime25": ("yentinglin/aime_2025", "train", "problem"),
}


def load_prompts(task: str, n: int, seed: int) -> list[str]:
    path, split, field = TASKS[task]
    ds = load_dataset(path, split=split)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)            # paper: randomly sample n problems
    return [ds[i][field] for i in idxs[:n]]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=sorted(TASKS), default="numinamath",
                    help="calibration source (paper: numinamath); aime* are for "
                         "the Sec. 3 figure only")
    ap.add_argument("--num-problems", type=int, default=5,
                    help="problems to sample (paper: 5 random NuminaMath problems)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    dev = next(model.parameters()).device

    prompts = load_prompts(args.task, args.num_problems, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for idx, q in enumerate(prompts):
            text = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tok(text, return_tensors="pt").to(dev)
            gen = model.generate(
                **inputs,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # gen.scores: tuple of [1, vocab] logits, one per generated token.
            ents = []
            for step in gen.scores:
                p = torch.softmax(step[0].float(), dim=-1)
                ents.append(float(-(p * p.clamp_min(1e-12).log()).sum()))
            f.write(json.dumps(
                {"idx": idx, "correct": None, "n_tokens": len(ents), "entropies": ents}
            ) + "\n")
            print(f"[dump_entropy] prob {idx}: {len(ents)} tokens")

    print(f"[dump_entropy] wrote {out_path}")


if __name__ == "__main__":
    main()
