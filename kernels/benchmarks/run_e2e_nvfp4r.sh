#!/usr/bin/env bash
# Run only the nvfp4r backend on a single GPU.
#
# Usage:
#   bash run_e2e_nvfp4r.sh [--smoke]
#
# To drop results into an existing sweep run so they can be merged later:
#   WORK_DIR=../results/e2e_bs_sweep/run_20260503_154823 bash run_e2e_nvfp4r.sh
#
# Key env vars (all optional):
#   WORK_DIR    — existing run directory to write part_nvfp4r.jsonl into
#   GPU         — GPU index to use (default: 3)
#   MODEL_SIZE  — 8b | 14b | 32b (default: 32b)

set -euo pipefail
trap 'echo "[INTERRUPTED] killing background jobs..."; kill 0; wait; exit 1' INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"

NVFP4R_PYTHON_PATH="${NVFP4R_PYTHON_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../python" && pwd)}"
TORCH_LIB="${TORCH_LIB:-$("$PY" -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')}"
MODELZOO="${MODELZOO:?set MODELZOO to the dir holding your NVFP4 model checkpoints}"
MODEL_SIZE="${MODEL_SIZE:-32b}"
GPU="${GPU:-3}"
BACKEND="${BACKEND:-nvfp4r}"

RESULTS="${SCRIPT_DIR}/../results/e2e_bs_sweep"

# Use provided WORK_DIR or create a new one
if [[ -n "${WORK_DIR:-}" ]]; then
    mkdir -p "$WORK_DIR/logs"
    echo "Using existing WORK_DIR: $WORK_DIR"
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    WORK_DIR="${RESULTS}/run_${TIMESTAMP}"
    mkdir -p "$WORK_DIR/logs"
    echo "Created new WORK_DIR: $WORK_DIR"
fi

LOG_DIR="${WORK_DIR}/logs"
part="${WORK_DIR}/part_${BACKEND}.jsonl"
log="${LOG_DIR}/${BACKEND}.log"

# ── Sweep parameters ──────────────────────────────────────────────────────────
SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

if (( SMOKE )); then
    OUTPUT_LENS="1024"
    BATCH_SIZES="1 2 4 8"
    ITERS=2
    echo "=== SMOKE TEST (output_len=1024, BS=1 2 4 8, iters=2) ==="
else
    OUTPUT_LENS="1024 2048 4096 8192 16384 32768"
    BATCH_SIZES="1 2 4 8"
    ITERS=1
    echo "=== FULL SWEEP ==="
fi

echo "GPU:      $GPU"
echo "Model:    $MODEL_SIZE"
echo "Backend:  $BACKEND"
echo "Output →  $part"
echo "Log    →  $log"
echo

env CUDA_VISIBLE_DEVICES="$GPU" \
    MODELZOO="$MODELZOO" \
    PYTHONPATH="$NVFP4R_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    NVFP4R_ENABLE_GEMM=1 \
    "$PY" "$SCRIPT_DIR/bench_e2e.py" \
        --backend      "$BACKEND" \
        --model        "$MODEL_SIZE" \
        --input-len    512 \
        --batch-sizes  $BATCH_SIZES \
        --output-lens  $OUTPUT_LENS \
        --iters        "$ITERS" \
        --out          "$part" \
    2>&1 | tee "$log"

echo
echo "[$(date +%H:%M:%S)] $BACKEND — done"
echo "Results → $part"

# ── Optional: print table if other part files exist in WORK_DIR ───────────────
shopt -s nullglob
parts=("$WORK_DIR"/part_*.jsonl)
if (( ${#parts[@]} > 1 )); then
    OUT_MERGED="${WORK_DIR}/merged.jsonl"
    echo
    echo "Merging ${#parts[@]} part files → $OUT_MERGED"
    cat "${parts[@]}" > "$OUT_MERGED"
    "$PY" - "$OUT_MERGED" <<'PYEOF'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
backends    = list(dict.fromkeys(r["backend"]    for r in rows))
batch_sizes = sorted(set(r["batch_size"] for r in rows))
out_lens    = sorted(set(r["output_len"]  for r in rows))

def get(rows, backend, bs, olen, key):
    for r in rows:
        if r["backend"] == backend and r["batch_size"] == bs and r["output_len"] == olen:
            return r.get(key), r.get("status", "ok")
    return None, None

def fmt_cell(val, status, width, fmt):
    if status == "OOM": return f"{'OOM':>{width}}"
    if val is None or status not in (None, "ok"): return f"{'n/a':>{width}}"
    return f"{val:{fmt}>{width}}"

COL = 12
for bs in batch_sizes:
    print(f"\n{'─'*8} batch_size={bs} {'─'*8}")
    hdr = f"{'output_len':>{COL}}"
    for b in backends:
        hdr += f"  {(b+' ms/tok'):{COL}}  {(b+' tok/s'):{COL}}"
    print(hdr)
    print("─" * (COL + len(backends) * (2 + COL + 2 + COL)))
    for olen in out_lens:
        line = f"{olen:>{COL}}"
        for b in backends:
            ms_val, st = get(rows, b, bs, olen, "ms_per_tok")
            ts_val, _  = get(rows, b, bs, olen, "tok_per_s")
            line += f"  {fmt_cell(ms_val, st, COL, '.3f')}  {fmt_cell(ts_val, st, COL, '.1f')}"
        print(line)

if "nvfp4r" in backends and "flashinfer" in backends:
    print(f"\n{'─'*8} nvfp4r speedup vs flashinfer (tok/s ratio) {'─'*8}")
    hdr = f"{'output_len':>{COL}}"
    for bs in batch_sizes: hdr += f"  {'BS='+str(bs):>{COL}}"
    print(hdr)
    for olen in out_lens:
        line = f"{olen:>{COL}}"
        for bs in batch_sizes:
            fi_val, fi_st = get(rows, "flashinfer", bs, olen, "tok_per_s")
            nr_val, nr_st = get(rows, "nvfp4r",     bs, olen, "tok_per_s")
            if fi_st == "OOM" or nr_st == "OOM": cell = f"{'OOM':>{COL}}"
            elif fi_val and nr_val: cell = f"{nr_val/fi_val:>{COL}.3f}x"
            else: cell = f"{'n/a':>{COL}}"
            line += f"  {cell}"
        print(line)
PYEOF
fi
