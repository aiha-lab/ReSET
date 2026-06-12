"""Per-token overhead of ReSET's entropy-conditional temperature scaling.

ReSET applies a fixed-threshold logits-processor on every decode step:
    H(p_t) = -sum p_t * log p_t
    T_eff  = T_low  if H <= H_thr else T_high
    logits' = logits / T_eff

We microbenchmark this processor on synthetic logits tensors at production
vocab sizes and compare to a typical decode-step latency.

Three configurations measured per vocab size:
  (1) entropy_only    : softmax + xlogx + sum + .item()  (the host sync cost)
  (2) temp_apply_only : logits / temperature               (the scaling cost)
  (3) full_processor  : (1) + threshold compare + (2)      (what runs every token)

We also compare against vLLM's own default temperature path (a single
SamplingParams.temperature, applied without a custom processor) to confirm
the *additional* cost of ReSET is just the entropy step.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

DEV = "cuda"
_LOGIT_INF_CLAMP = 80.0


def entropy_from_logits(logits: torch.Tensor) -> float:
    x = logits.to(dtype=torch.float32, copy=False).view(-1)
    x = x.clamp(-_LOGIT_INF_CLAMP, _LOGIT_INF_CLAMP)
    p = torch.softmax(x, dim=-1)
    log_p = torch.log(p + 1e-12)
    return float(-(p * log_p).sum().item())


def scale_by_temperature(logits: torch.Tensor, temp: float) -> torch.Tensor:
    if temp <= 1e-6:
        idx = int(torch.argmax(logits).item())
        out = torch.full_like(logits, float("-inf"))
        out[idx] = 0.0
        return out
    return logits / temp


class FixedThr:
    """Entropy-thresholded temperature processor (the per-token work ReSET adds)."""
    __slots__ = ("t_hi", "t_lo", "h_thr", "_last_T")

    def __init__(self, t_hi: float, t_lo: float, h_thr: float):
        self.t_hi, self.t_lo, self.h_thr = t_hi, t_lo, h_thr
        self._last_T = self.t_hi

    def __call__(self, _output_ids, logits: torch.Tensor):
        H = entropy_from_logits(logits)
        temp = self.t_lo if H <= self.h_thr else self.t_hi
        self._last_T = temp
        return scale_by_temperature(logits, temp)


# ── Vanilla baseline (vLLM default sampling) ────────────────────────────────
def vllm_default_temp(logits: torch.Tensor, T: float) -> torch.Tensor:
    """What vLLM's default sampler does internally for a fixed temperature."""
    return logits / T


def time_fn(fn, n_warm: int = 50, n_iter: int = 500) -> float:
    """Wall-clock eager timing in μs.

    NOTE: we use time.perf_counter, NOT cuda.Event, because the
    `.item()` host-sync inside the entropy step interacts badly
    with CUDA Event timing (the events get pushed past the syncs
    and overestimate per-call latency by ~10×). The wall-clock view
    is also what an inference server actually sees per token.
    """
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6  # μs/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocabs", type=int, nargs="+",
                    default=[151936, 152064, 200064],  # Qwen3, Qwen3-VL, Llama3
                    help="vocab sizes to test")
    ap.add_argument("--decode-step-ms", type=float, default=3.5,
                    help="reference decode-step latency for ratio computation")
    ap.add_argument("--output", default=None, help="optional JSON path to write results")
    args = ap.parse_args()

    print("ReSET per-token overhead microbenchmark\n")
    print(f"{'Vocab':>8}  "
          f"{'(a) entropy':>14}  "
          f"{'(b) temp_apply':>16}  "
          f"{'(c) FixedThr':>15}  "
          f"{'(d) vLLM /T':>13}  "
          f"{'extra (c−d)':>14}  "
          f"{'% of decode':>14}")
    print("-" * 110)

    rows = []
    for V in args.vocabs:
        # Build a realistic logits tensor (varied magnitudes so entropy ≠ 0)
        torch.manual_seed(42)
        logits = (torch.randn(1, V, device=DEV, dtype=torch.float32) * 4.0).contiguous()
        logits.sub_(logits.max(dim=-1, keepdim=True).values)  # safe softmax

        proc = FixedThr(t_hi=0.6, t_lo=0.1, h_thr=0.55)

        # (a) entropy only
        t_ent = time_fn(lambda: entropy_from_logits(logits))

        # (b) scale only (T > 0 path; no .item())
        t_scale = time_fn(lambda: scale_by_temperature(logits, 0.6))

        # (c) full FixedThr processor
        t_full = time_fn(lambda: proc(None, logits))

        # (d) vLLM default-temperature path (logits / T inside sampler)
        t_default = time_fn(lambda: vllm_default_temp(logits, 0.6))

        # how much extra ReSET adds vs vLLM default sampling
        extra_us = t_full - t_default
        pct = extra_us / (args.decode_step_ms * 1000) * 100

        print(f"{V:>8,}  "
              f"{t_ent:>13.2f}μs  "
              f"{t_scale:>15.2f}μs  "
              f"{t_full:>14.2f}μs  "
              f"{t_default:>12.2f}μs  "
              f"{extra_us:>+13.2f}μs  "
              f"{pct:>13.3f}%")
        rows.append({
            "vocab": V,
            "entropy_us": t_ent,
            "temp_apply_us": t_scale,
            "fixed_thr_us": t_full,
            "vllm_default_us": t_default,
            "extra_overhead_us": extra_us,
            "pct_of_decode_step": pct,
        })

    print()
    print(f"Reference decode-step latency = {args.decode_step_ms:.2f} ms (typical NVFP4 Qwen3-8B).")
    print()
    print("Components:")
    print("  (a) entropy_from_logits : softmax + xlogx + sum + .item()  (forces host sync)")
    print("  (b) scale_by_temperature: logits / temperature              (no sync)")
    print("  (c) FixedThr (full)     : (a) + threshold compare + (b)     (per-token)")
    print("  (d) vLLM default sampler: just logits / T (no entropy)      (baseline)")
    print()
    print("'extra' is the additional cost of ReSET vs running with a fixed temperature.")
    print("'% of decode' is that extra cost as a fraction of one decode step.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {len(rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
