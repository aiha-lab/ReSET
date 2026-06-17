#!/usr/bin/env python3
"""Evaluation harness for ReSET on vLLM (v1).

Tasks: aime120 (AIME 2022-2025), gpqa_diamond, livecodebench (also aime90 /
aime25). 8 samples/problem, top-p=0.95, max_tokens=32k. See reset/README.md.

WARNING: livecodebench runs model-generated Python in a subprocess; use a
sandboxed environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import zlib
from fractions import Fraction

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from reset import ReSETAdapter, get_newline_token_ids


# ==========================================================================
# Datasets (the three benchmarks reported in the paper).
# ==========================================================================
DATASETS = {
    "aime90": dict(path="xiaoyuanliu/AIME90", question_field="problem",
                   answer_field="answer", split="train", task_type="math"),
    "aime25": dict(path="yentinglin/aime_2025", question_field="problem",
                   answer_field="answer", split="train", task_type="math"),
    "gpqa_diamond": dict(path="Idavidrein/gpqa", subset="gpqa_diamond",
                         split="train", task_type="mcq", builder="gpqa"),
    "livecodebench": dict(path="livecodebench/code_generation_lite",
                          lcb_release_file="test6.jsonl",
                          task_type="code", builder="livecodebench"),
}

TASK_MAX_NEW_TOKENS = {
    "aime90": 32768, "aime25": 32768, "aime120": 32768,
    "gpqa_diamond": 32768, "livecodebench": 32768,
}


# --------------------------------------------------------------------------
# Builders (ported verbatim from the paper's evaluation code).
# --------------------------------------------------------------------------
_GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def _build_gpqa_examples(ds):
    import random as _r
    questions: list[str] = []
    answers: list[str] = []
    for idx, row in enumerate(ds):
        rng = _r.Random(idx)
        choices = [row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"],
                   row["Incorrect Answer 3"]]
        gold_idx = rng.randint(0, 3)
        choices.insert(gold_idx, row["Correct Answer"])
        prompt = _GPQA_TEMPLATE.format(
            question=row["Question"].strip(),
            A=choices[0].strip(), B=choices[1].strip(),
            C=choices[2].strip(), D=choices[3].strip(),
        )
        questions.append(prompt)
        answers.append("ABCD"[gold_idx])
    return questions, answers


_LCB_TEMPLATE_STDIN = (
    "You will be given a competitive programming problem. Read the input from "
    "standard input and write the output to standard output. Wrap your final "
    "Python solution in a single ```python ... ``` code block.\n\n"
    "Problem:\n{question}"
)
_LCB_TEMPLATE_FUNCTIONAL = (
    "You will be given a competitive programming problem. Implement the "
    "Solution class shown in the starter code. Wrap your final Python solution "
    "in a single ```python ... ``` code block.\n\n"
    "Problem:\n{question}\n\nStarter code:\n```python\n{starter}\n```"
)


def _build_lcb_examples(cfg, max_samples=None):
    from huggingface_hub import hf_hub_download

    fp = hf_hub_download(cfg["path"], cfg["lcb_release_file"], repo_type="dataset")
    rows = []
    with open(fp) as f:
        for line in f:
            rows.append(json.loads(line))

    questions: list[str] = []
    answers: list[str] = []
    for row in rows:
        starter = row.get("starter_code", "") or ""
        q = row["question_content"]
        if starter.strip():
            prompt = _LCB_TEMPLATE_FUNCTIONAL.format(question=q, starter=starter)
        else:
            prompt = _LCB_TEMPLATE_STDIN.format(question=q)

        public_tc = json.loads(row.get("public_test_cases", "[]") or "[]")
        priv_raw = row.get("private_test_cases", "")
        private_tc = []
        if priv_raw:
            try:
                private_tc = json.loads(zlib.decompress(base64.b64decode(priv_raw)).decode())
            except Exception:
                try:
                    private_tc = json.loads(priv_raw)
                except Exception:
                    private_tc = []

        meta_raw = row.get("metadata") or {}
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except Exception:
                meta = {}
        else:
            meta = meta_raw
        fn_name = meta.get("func_name", "") if isinstance(meta, dict) else ""
        gold = json.dumps({
            "tests": (public_tc + private_tc),
            "starter": starter,
            "fn_name": fn_name,
        })
        questions.append(prompt)
        answers.append(gold)

    return questions, answers


def _load_one(task_name: str, max_problems: int | None):
    cfg = DATASETS[task_name]
    builder = cfg.get("builder")
    if builder == "livecodebench":
        questions, answers = _build_lcb_examples(cfg, max_problems)
    else:
        load_kwargs = {"path": cfg["path"], "split": cfg["split"]}
        if "subset" in cfg:
            load_kwargs["name"] = cfg["subset"]
        ds = load_dataset(**load_kwargs)
        if builder == "gpqa":
            questions, answers = _build_gpqa_examples(ds)
        else:
            questions = [row[cfg["question_field"]] for row in ds]
            answers = [str(row[cfg["answer_field"]]) for row in ds]
    if max_problems is not None:
        questions, answers = questions[:max_problems], answers[:max_problems]
    return questions, answers, cfg["task_type"]


def load_task(task: str, max_problems: int | None):
    """Return ``(questions, answers, task_type)``. ``aime120`` = aime90 ∪ aime25."""
    if task == "aime120":
        q90, a90, _ = _load_one("aime90", None)
        q25, a25, _ = _load_one("aime25", None)
        questions, answers = q90 + q25, a90 + a25
        if max_problems is not None:
            questions, answers = questions[:max_problems], answers[:max_problems]
        return questions, answers, "math"
    return _load_one(task, max_problems)


def format_prompts(tokenizer, questions: list[str]) -> list[str]:
    prompts = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(prompt)
    return prompts


# ==========================================================================
# Answer extraction / scoring (ported verbatim from the paper's eval code).
# ==========================================================================
def extract_boxed(text: str) -> str | None:
    matches: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len("\\boxed{")
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    matches.append(text[start:j])
                    break
                depth -= 1
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return matches[-1] if matches else None


def normalize_math_answer(ans: str) -> str:
    ans = ans.strip().strip("$").strip().replace(" ", "")
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = ans.replace("\\%", "%").rstrip(".")
    if ans.startswith("+"):
        ans = ans[1:]
    return ans


def try_parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if not s:
        return None
    if "/" in s:
        try:
            return float(Fraction(s))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(s)
    except ValueError:
        return None


def math_equal(pred: str, gold: str) -> bool:
    pred_n, gold_n = normalize_math_answer(pred), normalize_math_answer(gold)
    if pred_n == gold_n:
        return True
    pv, gv = try_parse_number(pred_n), try_parse_number(gold_n)
    if pv is not None and gv is not None:
        return abs(pv - gv) < 1e-6
    return False


def extract_mcq_answer(text: str) -> str | None:
    matches = re.findall(
        r"[Aa]nswer\s*[:\-=]?\s*\$?\(?\*?\*?([A-D])\*?\*?\)?\$?", text)
    if matches:
        return matches[-1].upper()
    boxed = extract_boxed(text)
    if boxed and boxed.strip().upper() in ("A", "B", "C", "D"):
        return boxed.strip().upper()
    m = re.search(r"(?:answer|choice|option)\s+is\s*[:\s]*\(?([A-D])\)?",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"(?:^|\W)([A-D])(?:\W|$)", text)
    if letters:
        return letters[-1].upper()
    return None


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str:
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    fb = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if fb:
        return fb[-1].strip()
    return text.strip()


def _run_code_against_test(code: str, test: dict, fn_name: str = "",
                           timeout: float = 8.0) -> bool:
    import subprocess
    import textwrap

    ttype = test.get("testtype", "stdin")
    inp = test.get("input", "")
    expected = (test.get("output") or "").rstrip()

    if ttype == "functional" or (fn_name and ttype != "stdin"):
        driver = textwrap.dedent(f"""
            import sys, json
            from io import StringIO
            {code}
            try:
                _sol = Solution()
            except NameError:
                _sol = None
            _fn = getattr(_sol, {fn_name!r}, None) if _sol else None
            if _fn is None:
                _fn = globals().get({fn_name!r})
            _raw = {inp!r}
            _args = []
            for _ln in _raw.split("\\n"):
                _ln = _ln.strip()
                if not _ln:
                    continue
                try:
                    _args.append(eval(_ln, {{}}, {{}}))
                except Exception:
                    _args.append(_ln)
            _out = _fn(*_args)
            print(json.dumps(_out, sort_keys=True))
        """).strip()
        try:
            r = subprocess.run(["python3", "-c", driver],
                               capture_output=True, text=True, timeout=timeout)
            try:
                got = json.dumps(json.loads(r.stdout.strip()), sort_keys=True)
                want = json.dumps(json.loads(expected), sort_keys=True)
                return got == want
            except Exception:
                return r.stdout.strip() == expected.strip()
        except Exception:
            return False

    try:
        r = subprocess.run(["python3", "-c", code],
                           input=inp, capture_output=True, text=True, timeout=timeout)
        return r.stdout.rstrip() == expected
    except Exception:
        return False


def evaluate_code_sample(pred_text: str, gold_json: str) -> bool:
    code = extract_python_code(pred_text)
    try:
        gold = json.loads(gold_json)
    except Exception:
        return False
    tests = gold.get("tests", [])
    fn_name = gold.get("fn_name", "")
    if not tests:
        return False
    for t in tests:
        if not _run_code_against_test(code, t, fn_name=fn_name, timeout=8.0):
            return False
    return True


def is_correct(pred_text: str, gold: str, task_type: str) -> bool:
    if task_type == "mcq":
        pred = extract_mcq_answer(pred_text) or ""
        return pred.upper() == gold.strip().upper()
    if task_type == "code":
        return evaluate_code_sample(pred_text, gold)
    pred = extract_boxed(pred_text)
    if pred is None:
        nums = re.findall(r"[-+]?\d*\.?\d+", pred_text)
        pred = nums[-1] if nums else ""
    return math_equal(pred, gold)


# ==========================================================================
# Engine construction.
# ==========================================================================
def build_llm(args) -> LLM:
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        logits_processors=[ReSETAdapter],
    )
    quant = args.quantization
    if quant is None and ("nvfp4" in args.model.lower() or "fp4" in args.model.lower()):
        quant = "modelopt_fp4"
    if quant == "modelopt_fp4":
        llm_kwargs["quantization"] = "modelopt_fp4"
        llm_kwargs["hf_overrides"] = {"quantization_config": None}
    elif quant:
        llm_kwargs["quantization"] = quant
    return LLM(**llm_kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="aime120",
                    choices=["aime120", "aime90", "aime25", "gpqa_diamond", "livecodebench"])
    ap.add_argument("--max-problems", type=int, default=None)

    # ReSET hyperparameters (paper Sec. 4).
    ap.add_argument("--t-high", type=float, default=1.0, help="T_high (paper: 1.0)")
    ap.add_argument("--t-low", type=float, default=0.3,
                    help="T_low, calibrated per (model, task)")
    ap.add_argument("--tau0", type=float, default=0.6349,
                    help="tau_0 = 80th-pct token entropy on the calibration split")
    ap.add_argument("--window", type=int, default=32, help="HSE window w (paper: 32)")

    # Baseline mode: single fixed temperature, no ReSET.
    ap.add_argument("--baseline", action="store_true",
                    help="decode at a single fixed temperature (no ReSET)")
    ap.add_argument("--base-temp", type=float, default=0.6,
                    help="fixed temperature used in --baseline mode")

    # Sampling / engine.
    ap.add_argument("--n-samples", type=int, default=8,
                    help="samples per problem (paper averages over 8 seeds)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default: per-task value (32k for the reported benchmarks)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=34816)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--quantization", default=None,
                    help="override vLLM quant backend (default: auto modelopt_fp4 "
                         "for *nvfp4* checkpoints)")
    ap.add_argument("--backend", choices=["auto", "nvfp4r"], default="auto",
                    help="NVFP4 linear backend; 'nvfp4r' uses the CUDA-core kernels")
    args = ap.parse_args()

    if args.backend == "nvfp4r":
        import nvfp4r
        nvfp4r.enable()
        print(f"[reset] nvfp4r backend: {nvfp4r.status()}")

    max_tokens = args.max_tokens or TASK_MAX_NEW_TOKENS.get(args.task, 32768)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)
    questions, answers, task_type = load_task(args.task, args.max_problems)
    prompts = format_prompts(tokenizer, questions)
    print(f"[reset] {args.task} ({task_type}): {len(prompts)} problems, "
          f"n_samples={args.n_samples}, max_tokens={max_tokens}, "
          f"mode={'baseline' if args.baseline else 'ReSET'}")

    llm = build_llm(args)

    if args.baseline:
        sp = SamplingParams(n=args.n_samples, temperature=args.base_temp,
                            top_p=args.top_p, max_tokens=max_tokens, seed=args.seed)
    else:
        nl_ids, dnl_ids = get_newline_token_ids(tokenizer)
        print(f"[reset] step-boundary tokens: nl={len(nl_ids)}, dnl={len(dnl_ids)}")
        # vLLM applies the per-token temperature inside the logits processor, so
        # SamplingParams temperature stays at 1.0 (no double scaling).
        sp = SamplingParams(
            n=args.n_samples, temperature=1.0, top_p=args.top_p,
            max_tokens=max_tokens, seed=args.seed,
            extra_args={
                "reset": True,
                "t_high": args.t_high, "t_low": args.t_low,
                "tau_0": args.tau0, "window": args.window,
                "reset_nl_ids": nl_ids, "reset_dnl_ids": dnl_ids,
            },
        )

    outputs = llm.generate(prompts, [sp] * len(prompts))

    total = correct = 0
    for out, gold in zip(outputs, answers):
        for comp in out.outputs:
            total += 1
            correct += int(is_correct(comp.text, gold, task_type))
    acc = correct / total if total else 0.0
    print(f"\n[reset] avg@{args.n_samples} accuracy on {args.task}: "
          f"{acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
