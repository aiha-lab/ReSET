#!/usr/bin/env python3
import argparse
import os
import time

import torch
import transformers

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

QFORMAT_CHOICES = {
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",
}


def make_calib_dataloader(tokenizer, num_samples=32, seq_len=2048, batch_size=1):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for sample in ds:
        texts.append(sample["text"])
        if len(texts) >= num_samples:
            break

    batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        batches.append(encoded)
    return batches


def main():
    parser = argparse.ArgumentParser(description="NVFP4 PTQ with modelopt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--qformat", type=str, default="nvfp4",
                        choices=list(QFORMAT_CHOICES.keys()))
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--calib-seq-len", type=int, default=2048)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    print(f"[quantize] Model:           {args.model}")
    print(f"[quantize] Output:          {args.output}")
    print(f"[quantize] Format:          {args.qformat}")
    print(f"[quantize] Calib samples:   {args.calib_samples}")

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    print(f"\n[quantize] Loading model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    print(f"[quantize] Model loaded: {type(model).__name__}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[quantize] Preparing calibration data ({args.calib_samples} samples)...")
    calib_dataloader = make_calib_dataloader(
        tokenizer,
        num_samples=args.calib_samples,
        seq_len=args.calib_seq_len,
        batch_size=args.calib_batch_size,
    )

    quant_cfg = getattr(mtq, QFORMAT_CHOICES[args.qformat])

    device = next(model.parameters()).device

    def forward_loop(model):
        for i, batch in enumerate(calib_dataloader):
            input_ids = batch["input_ids"].to(device)
            model(input_ids)
            print(f"  Calib batch {i + 1}/{len(calib_dataloader)}")

    print(f"[quantize] Running PTQ (quantize + calibrate input_scale)...")
    t0 = time.time()
    model = mtq.quantize(model, quant_cfg, forward_loop)
    t_quant = time.time() - t0
    print(f"[quantize] PTQ complete in {t_quant:.1f}s")

    mtq.print_quant_summary(model)

    print(f"\n[quantize] Exporting to {args.output}...")
    t0 = time.time()
    os.makedirs(args.output, exist_ok=True)
    # export_hf_checkpoint signature: (model, dtype=None, export_dir=...)
    # Use keyword arg to avoid signature drift between modelopt versions.
    export_hf_checkpoint(model, export_dir=args.output)
    t_export = time.time() - t0
    print(f"[quantize] Export complete in {t_export:.1f}s")

    tokenizer.save_pretrained(args.output)

    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith((".safetensors", ".bin"))
    )

    print(f"\n[quantize] Done!")
    print(f"  Checkpoint size: {total_size / 1e9:.2f} GB")
    print(f"  Quantization:    {t_quant:.1f}s")
    print(f"  Export:          {t_export:.1f}s")
    print(f"  Output:          {args.output}")
    print(f"\n  To use with vLLM:")
    print(f"    LLM(model='{args.output}', quantization='modelopt_fp4')")


if __name__ == "__main__":
    main()
