#!/usr/bin/env python3
"""Minimal example: one prompt in, generated text out.

Runs an NVFP4 checkpoint through the `nvfp4r` CUDA-core kernels with ReSET
decoding. Quantize a model first (see the top-level README):

    reset-quantize --model Qwen/Qwen3-8B --output Qwen3-8B-nvfp4

Then:

    python examples/generate.py --model Qwen3-8B-nvfp4 \
        --prompt "What is 17 times 24? Think step by step."
"""
import argparse

import nvfp4r
from vllm import LLM, SamplingParams

from reset import ReSETAdapter, get_newline_token_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3-8B-nvfp4",
                    help="NVFP4 checkpoint (produced by reset-quantize)")
    ap.add_argument("--prompt",
                    default="What is 17 times 24? Think step by step, then give the answer.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--t-low", type=float, default=0.1)
    ap.add_argument("--tau0", type=float, default=0.5505)
    ap.add_argument("--no-reset", action="store_true",
                    help="plain decoding (skip the ReSET temperature scaling)")
    args = ap.parse_args()

    nvfp4r.enable()   # route the NVFP4 linear layers through the nvfp4r kernels
    llm = LLM(model=args.model, quantization="modelopt_fp4",
              hf_overrides={"quantization_config": None},
              logits_processors=[ReSETAdapter])
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True)

    if args.no_reset:
        sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens)
    else:
        nl, dnl = get_newline_token_ids(tok)
        # ReSET sets the per-token temperature itself, so SamplingParams stays at 1.0.
        sp = SamplingParams(
            temperature=1.0, top_p=0.95, max_tokens=args.max_tokens,
            extra_args={"reset": True, "t_high": 1.0, "t_low": args.t_low,
                        "tau_0": args.tau0, "window": 32,
                        "reset_nl_ids": nl, "reset_dnl_ids": dnl})

    out = llm.generate([prompt], sp)
    print("\n=== prompt ===\n" + args.prompt)
    print("\n=== output ===\n" + out[0].outputs[0].text)


if __name__ == "__main__":
    main()
