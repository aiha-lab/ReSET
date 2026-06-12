"""ReSET: Reasoning Step Entropy-based Temperature Scaling.

A vLLM (v1) logits processor. Per token with entropy ``H_t``:

    T_t = T_low   if  H_t <  tau_t          tau_t = tau_0   if  H_step <= H_bar
    T_t = T_high  if  H_t >= tau_t          tau_t = H_step  if  H_step >  H_bar

``H_bar`` is the running mean of all token entropies; ``H_step`` is the within-
step entropy (size-``w`` sliding window for the first ``w`` tokens of a step,
within-step running average after). Steps are split on ``"\\n\\n"``. Defaults:
``T_high=1.0``, ``T_low`` and ``tau_0`` (80th-pct token entropy) calibrated per
(model, task), ``w=32``.
"""

from __future__ import annotations

import torch

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


def scale_by_temperature(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """Divide logits by ``temp`` (greedy argmax when ``temp`` ~ 0)."""
    if temp <= 1e-6:
        idx = int(torch.argmax(logits).item())
        out = torch.full_like(logits, float("-inf"))
        out[idx] = 0.0
        return out
    return logits / temp


def entropy_of(logits: torch.Tensor) -> float:
    """Shannon entropy (nats) of softmax(logits)."""
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)
    return float(ent.item())


def get_newline_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Scan the vocabulary for step-boundary token ids.

    Returns ``(nl_ids, dnl_ids)`` where:
        nl_ids  — token ids that decode to exactly ``"\\n"``
        dnl_ids — token ids whose decoded text contains ``"\\n\\n"``
                  (covers composite tokens like ``".\\n\\n"``, ``"\\n\\n\\n"``, …)
    A step boundary is hit when the last token is a ``dnl`` token, or the last
    two tokens are both ``nl`` tokens.
    """
    nl_ids, dnl_ids = [], []
    for tid in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        if decoded == "\n":
            nl_ids.append(tid)
        if "\n\n" in decoded:
            dnl_ids.append(tid)
    return nl_ids, dnl_ids


class ReSETRequest:
    """Per-request ReSET temperature policy (paper Eqs. 1–2).

    Args:
        t_high:   T_high — temperature for above-threshold (high-entropy) tokens.
        t_low:    T_low  — temperature for below-threshold (low-entropy) tokens.
        tau_raw:  tau_0  — global token-entropy threshold for confident steps
                  (80th-percentile token entropy on the calibration split).
        window:   w       — HSE window / step-transition init length.
        nl_ids:   token ids decoding to a single newline.
        dnl_ids:  token ids whose decoded text contains a double newline.
    """

    __slots__ = ("t_high", "t_low", "tau_raw", "window",
                 "nl_ids", "dnl_ids", "sw_buffer", "step_buffer",
                 "_global_sum", "_global_n")

    def __init__(self, t_high, t_low, tau_raw, window, nl_ids, dnl_ids):
        self.t_high   = float(t_high)
        self.t_low    = float(t_low)
        self.tau_raw  = float(tau_raw)
        self.window   = int(window)
        self.nl_ids   = frozenset(int(x) for x in nl_ids)
        self.dnl_ids  = frozenset(int(x) for x in dnl_ids)
        self.sw_buffer:   list[float] = []   # size-w sliding window (spans steps)
        self.step_buffer: list[float] = []   # entropies since current step start
        self._global_sum: float = 0.0        # running sum of all token entropies
        self._global_n:   int   = 0          # running count -> H_bar = sum / n

    def _is_boundary(self, output_ids) -> bool:
        n = len(output_ids)
        if n == 0:
            return False
        if output_ids[-1] in self.dnl_ids:
            return True
        if n >= 2 and output_ids[-1] in self.nl_ids and output_ids[-2] in self.nl_ids:
            return True
        return False

    def __call__(self, output_ids, logits):
        # New step: reset the within-step buffer (HSE start t_0).
        if self._is_boundary(output_ids):
            self.step_buffer = []

        ent = entropy_of(logits)

        # HSE (Eq. 2): window average during step init, within-step average after.
        if len(self.step_buffer) < self.window:
            H_step_est = (sum(self.sw_buffer) / len(self.sw_buffer)
                          if self.sw_buffer else ent)
        else:
            H_step_est = sum(self.step_buffer) / len(self.step_buffer)

        # SAT (Eq. 1): step is "uncertain" iff H_step exceeds the running global
        # mean H_bar; in that regime the threshold is step-relative (H_step),
        # otherwise it falls back to the global tau_0 (tau_raw).
        global_mean = self._global_sum / self._global_n if self._global_n > 0 else H_step_est
        high_step   = H_step_est > global_mean

        if not high_step:
            temp = self.t_high if ent >= self.tau_raw else self.t_low
        else:
            temp = self.t_high if ent >= H_step_est else self.t_low

        # Bookkeeping: append to step + sliding-window buffers and update H_bar.
        self.step_buffer.append(ent)
        self.sw_buffer.append(ent)
        if len(self.sw_buffer) > self.window:
            del self.sw_buffer[0]
        self._global_sum += ent
        self._global_n   += 1

        return scale_by_temperature(logits, temp)


class ReSETAdapter(AdapterLogitsProcessor):
    """vLLM v1 adapter: register with ``LLM(logits_processors=[ReSETAdapter])``
    and enable per request via ``SamplingParams.extra_args`` (see README)."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        ea = getattr(params, "extra_args", None) or {}
        if not ea.get("reset"):
            return None
        return ReSETRequest(
            t_high  = float(ea["t_high"]),
            t_low   = float(ea["t_low"]),
            tau_raw = float(ea.get("tau_0", 0.6349)),
            window  = int(ea.get("window", 32)),
            nl_ids  = ea.get("reset_nl_ids", []),
            dnl_ids = ea.get("reset_dnl_ids", []),
        )
