#!/bin/bash
#
# Full experiment pipeline: hyperparameter search + main experiments.
#
# Runs everything from scratch with automatic hyperparameter selection.
#
# Usage:
#   bash scripts/run_essay_experiments.sh [--fresh|--resume-run]  # interactive
#   nohup bash scripts/run_essay_experiments.sh --fresh &         # background (overnight)
#
# Modes:
#   --fresh (default)   Archive existing results and start from scratch
#   --resume-run        Keep existing results and continue from cache
#
# Prerequisites: OPENROUTER_API_KEY set in environment or .env file.
#
# Phases:
#   0   Archive old results + pre-flight check
#   1A  Prompt variant selection (val, pilot model)
#   1B  N-shot ablation (val, pilot model)
#   1C  Example sensitivity (val, pilot model)
#   1D  Confirmation on second model (val)
#   2   Run-to-run variance (test, 2 models)
#   3   Main experiments (test, all 4 models)
#

set -euo pipefail
cd "$(dirname "$0")/.."

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Export all config vars upfront so inline Python heredocs can read them
export PILOT_MODEL="gemini-3-flash-preview"
export CONFIRM_MODEL="claude-haiku-4.5"
export VARIANCE_MODELS="gemini-3-flash-preview claude-haiku-4.5"
export ALL_MODELS="gemini-3-flash-preview gpt-5-mini claude-haiku-4.5 kimi-k2.5"

CORE_METHODS="zs_e2e fs_e2e fs_cot_e2e zs_pipe fs_pipe"
ZS_METHODS="zs_e2e zs_pipe gold_zs"
E2E_FS_METHODS="fs_e2e fs_cot_e2e"
PIPE_FS_METHODS="fs_pipe gold_fs"

VARIANTS="default enhanced minimal"
N_VALUES="1 3 5 10"
SEEDS="0 1 2 3 4"
VARIANCE_RUNS="1 2 3"

# Run mode: fresh by default, optional resume without archiving.
RUN_MODE="fresh"
for arg in "$@"; do
    case "$arg" in
        --fresh)
            RUN_MODE="fresh"
            ;;
        --resume-run)
            RUN_MODE="resume"
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_essay_experiments.sh [--fresh|--resume-run]"
            echo "  --fresh      Archive existing results and start from scratch (default)"
            echo "  --resume-run Keep existing results and continue with --resume"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $arg" >&2
            echo "Use --help for usage." >&2
            exit 2
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/main_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
    # Run experiment, log full output, continue on failure.
    # The "|| rc=$?" pattern prevents set -e from killing the script
    # when the python command returns non-zero.
    log "    CMD: python run_essay_experiment.py $*"
    local rc=0
    python run_essay_experiment.py "$@" >> "$LOG" 2>&1 || rc=$?
    if [ $rc -ne 0 ]; then
        log "    WARNING: exit code $rc — continuing"
    fi
    return 0
}

# Load API key from .env if not already set
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
    _env_line=$(grep -v '^#' .env | grep OPENROUTER_API_KEY | head -1 || true)
    if [ -n "${_env_line:-}" ]; then
        export "$(echo "$_env_line" | xargs)"
    fi
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    log "ERROR: OPENROUTER_API_KEY is not set. Aborting."
    exit 1
fi

log "════════════════════════════════════════════════════════════════"
log "  Full Experiment Pipeline — Starting"
log "  Log: $LOG"
log "  Mode: $RUN_MODE"
log "  Pilot model: $PILOT_MODEL"
log "  All models:  $ALL_MODELS"
log "════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: Archive old results + pre-flight check
# ═══════════════════════════════════════════════════════════════════════

log ""
log "══════ PHASE 0: Preparation ══════"

ARCHIVE_DIR="archive/pre_rerun_${TIMESTAMP}"
if [ "$RUN_MODE" = "fresh" ]; then
    if [ -d "results" ] || [ -d "results_val" ]; then
        mkdir -p "$ARCHIVE_DIR"
        if [ -d "results" ]; then
            log "  Archiving results/ → $ARCHIVE_DIR/results"
            mv results "$ARCHIVE_DIR/results"
        fi
        if [ -d "results_val" ]; then
            log "  Archiving results_val/ → $ARCHIVE_DIR/results_val"
            mv results_val "$ARCHIVE_DIR/results_val"
        fi
        log "  Old results archived."
    else
        log "  No existing results to archive."
    fi
else
    log "  Resume mode: keeping existing results/ and results_val/."
fi

log "  Pre-flight model check (required models)..."
export PRECHECK_MODELS="$ALL_MODELS"
python3 << 'PYEOF' >> "$LOG" 2>&1
import os
import sys

from run_essay_experiment import MODELS
from src.llm_client import LLMClient

required = os.environ.get("PRECHECK_MODELS", "").split()
client = LLMClient()
failed = []

print("Pre-flight check (strict):")
for mk in required:
    if mk not in MODELS:
        print(f"  {mk:25s} UNKNOWN MODEL")
        failed.append(mk)
        continue
    ok = client.check_model(mk)
    print(f"  {mk:25s} {'OK' if ok else 'FAILED'}")
    if not ok:
        failed.append(mk)

if failed:
    print("Pre-flight FAILED for required models:", ", ".join(failed))
    sys.exit(1)

print("Pre-flight OK for all required models.")
PYEOF
log "  Pre-flight check complete: all required models OK."

log "Phase 0 complete."

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1A: Prompt Variant Selection (val, pilot model)
# ═══════════════════════════════════════════════════════════════════════

log ""
log "══════ PHASE 1A: Prompt Variant Selection ══════"
log "  Model: $PILOT_MODEL | Split: val | Methods: $CORE_METHODS"
log "  Variants: $VARIANTS"

for VARIANT in $VARIANTS; do
    for METHOD in $CORE_METHODS; do
        log "  Running: $METHOD  variant=$VARIANT"
        run --model "$PILOT_MODEL" --method "$METHOD" \
            --split val --prompt-variant "$VARIANT" --resume
    done
done
log "Phase 1A complete. Analyzing..."

# --- Auto-select best variant ---
BEST_PV=$(python3 << 'PYEOF'
import json, os, sys

variants = ['default', 'enhanced', 'minimal']
model = os.environ.get('PILOT_MODEL', 'gemini-3-flash-preview')
methods = ['zs_e2e', 'fs_e2e', 'fs_cot_e2e', 'zs_pipe', 'fs_pipe']

scores = {v: [] for v in variants}
for v in variants:
    for method in methods:
        tag = f'.pv_{v}' if v != 'default' else ''
        path = f'results_val/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            s = data.get('macro', {}).get('relation_macro_f1', 0)
            scores[v].append(s)
            print(f'  {v:10s} {method:15s}: {s:.4f}', file=sys.stderr)

print('', file=sys.stderr)
expected = len(methods)
for v in variants:
    avg = sum(scores[v]) / len(scores[v]) if scores[v] else 0
    status = '' if len(scores[v]) == expected else f'  WARNING: only {len(scores[v])}/{expected} methods'
    print(f'  {v:10s} mean={avg:.4f} (n={len(scores[v])}){status}', file=sys.stderr)

# Fail fast if any variant is incomplete
incomplete = [v for v in variants if len(scores[v]) != expected]
if incomplete:
    print('  ERROR: incomplete variant sweep; refusing to auto-select', file=sys.stderr)
    for v in incomplete:
        print(f'    - {v}: {len(scores[v])}/{expected} methods present', file=sys.stderr)
    sys.exit(2)

# Pick best; prefer "default" if within 1% of the best
avgs = {v: sum(scores[v])/len(scores[v]) if scores[v] else 0 for v in variants}
best = max(variants, key=lambda v: avgs[v])
if best != 'default' and avgs['default'] >= avgs[best] - 0.01:
    best = 'default'
    print(f'  >>> "default" within 1% of best; selecting default', file=sys.stderr)
else:
    print(f'  >>> Selected: {best}', file=sys.stderr)
print(best)
PYEOF
) 2>> "$LOG"

export BEST_PV
BEST_PV=$(echo "$BEST_PV" | tr -d '[:space:]')
log "  Selected variant: $BEST_PV"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1B: N-Shot Ablation (val, pilot model)
# ═══════════════════════════════════════════════════════════════════════

log ""
log "══════ PHASE 1B: N-Shot Ablation ══════"
log "  Model: $PILOT_MODEL | Variant: $BEST_PV | N values: $N_VALUES"

# Search N for fs_e2e (determines N_E2E)
log "  --- Searching N for fs_e2e ---"
for N in $N_VALUES; do
    log "  Running: fs_e2e  n=$N"
    run --model "$PILOT_MODEL" --method fs_e2e \
        --split val --prompt-variant "$BEST_PV" --n-examples "$N" --resume
done

# Search N for fs_pipe (determines N_Pipe)
log "  --- Searching N for fs_pipe ---"
for N in $N_VALUES; do
    log "  Running: fs_pipe  n=$N"
    run --model "$PILOT_MODEL" --method fs_pipe \
        --split val --prompt-variant "$BEST_PV" --n-examples "$N" --resume
done
log "Phase 1B complete. Analyzing..."

# --- Auto-select best N for each method group ---
_N_RESULT=$(python3 << 'PYEOF' 2>> "$LOG"
import json, os, sys

variant = os.environ['BEST_PV']
model = os.environ.get('PILOT_MODEL', 'gemini-3-flash-preview')

def compute_tag(variant, n):
    parts = []
    if variant != 'default':
        parts.append(f'pv_{variant}')
    if n != 3:
        parts.append(f'n{n}')
    return '.' + '.'.join(parts) if parts else ''

def find_best_n(method, n_values):
    scores = {}
    missing = []
    for n in n_values:
        tag = compute_tag(variant, n)
        path = f'results_val/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            s = data.get('macro', {}).get('relation_macro_f1', 0)
            scores[n] = s
            print(f'  {method:10s} n={n:2d}: {s:.4f}', file=sys.stderr)
        else:
            print(f'  {method:10s} n={n:2d}: MISSING ({path})', file=sys.stderr)
            missing.append(path)

    if missing:
        print(f'  ERROR: incomplete N sweep for {method}; missing {len(missing)}/{len(n_values)} files', file=sys.stderr)
        for path in missing:
            print(f'    - {path}', file=sys.stderr)
        sys.exit(2)

    # Best N; prefer smaller if within 2%
    sorted_ns = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
    best = sorted_ns[0]
    for n in sorted(scores.keys()):
        if scores[n] >= scores[best] - 0.02:
            best = n
            break

    runner_up = sorted_ns[1] if len(sorted_ns) > 1 else best
    return best, runner_up

n_values = [1, 3, 5, 10]

print('', file=sys.stderr)
print('--- fs_e2e ---', file=sys.stderr)
best_e2e, runner_e2e = find_best_n('fs_e2e', n_values)
print(f'  >>> N_E2E = {best_e2e} (runner-up: {runner_e2e})', file=sys.stderr)

print('', file=sys.stderr)
print('--- fs_pipe ---', file=sys.stderr)
best_pipe, _ = find_best_n('fs_pipe', n_values)
print(f'  >>> N_Pipe = {best_pipe}', file=sys.stderr)

# Output: BEST_N_E2E BEST_N_PIPE RUNNER_UP_N_E2E
print(f'{best_e2e} {best_pipe} {runner_e2e}')
PYEOF
)
read -r BEST_N_E2E BEST_N_PIPE RUNNER_UP_N_E2E <<< "$_N_RESULT"

BEST_N_E2E=$(echo "$BEST_N_E2E" | tr -d '[:space:]')
BEST_N_PIPE=$(echo "$BEST_N_PIPE" | tr -d '[:space:]')
RUNNER_UP_N_E2E=$(echo "$RUNNER_UP_N_E2E" | tr -d '[:space:]')
export BEST_N_E2E BEST_N_PIPE RUNNER_UP_N_E2E

log "  Selected: N_E2E=$BEST_N_E2E  N_Pipe=$BEST_N_PIPE  (runner-up E2E: $RUNNER_UP_N_E2E)"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1C: Example Sensitivity (val, pilot model)
# ═══════════════════════════════════════════════════════════════════════

log ""
log "══════ PHASE 1C: Example Sensitivity ══════"
log "  Model: $PILOT_MODEL | fs_e2e | variant=$BEST_PV | n=$BEST_N_E2E"
log "  Seeds: $SEEDS"

for SEED in $SEEDS; do
    log "  Running: fs_e2e  seed=$SEED"
    run --model "$PILOT_MODEL" --method fs_e2e \
        --split val --prompt-variant "$BEST_PV" \
        --n-examples "$BEST_N_E2E" --example-seed "$SEED" --resume
done
log "Phase 1C complete."

# Quick summary of seed results
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, statistics

variant = os.environ.get('BEST_PV', 'default')
best_n = int(os.environ.get('BEST_N_E2E', '3'))
model = os.environ.get('PILOT_MODEL', 'gemini-3-flash-preview')

def compute_tag(variant, n, seed):
    parts = []
    if variant != 'default':
        parts.append(f'pv_{variant}')
    if n != 3:
        parts.append(f'n{n}')
    parts.append(f'seed{seed}')
    return '.' + '.'.join(parts)

scores = []
for seed in range(5):
    tag = compute_tag(variant, best_n, seed)
    path = f'results_val/PersuasiveEssaysV2/metrics/{model}_fs_e2e{tag}.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        s = data.get('macro', {}).get('relation_macro_f1', 0)
        scores.append(s)
        print(f'  seed={seed}: RelM F1 = {s:.4f}')

if len(scores) > 1:
    m, s = statistics.mean(scores), statistics.stdev(scores)
    print(f'  Mean +/- Std: {m:.4f} +/- {s:.4f}')
PYEOF

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1D: Confirmation on Second Model (val)
# ═══════════════════════════════════════════════════════════════════════

log ""
log "══════ PHASE 1D: Confirmation on $CONFIRM_MODEL ══════"
log "  Best config: variant=$BEST_PV  N_E2E=$BEST_N_E2E  N_Pipe=$BEST_N_PIPE"

# Best N on fs_e2e
log "  Running: $CONFIRM_MODEL fs_e2e n=$BEST_N_E2E (best)"
run --model "$CONFIRM_MODEL" --method fs_e2e \
    --split val --prompt-variant "$BEST_PV" --n-examples "$BEST_N_E2E" --resume

# Runner-up N on fs_e2e (if different)
if [ "$RUNNER_UP_N_E2E" != "$BEST_N_E2E" ]; then
    log "  Running: $CONFIRM_MODEL fs_e2e n=$RUNNER_UP_N_E2E (runner-up)"
    run --model "$CONFIRM_MODEL" --method fs_e2e \
        --split val --prompt-variant "$BEST_PV" --n-examples "$RUNNER_UP_N_E2E" --resume
fi

# Best N on fs_pipe
log "  Running: $CONFIRM_MODEL fs_pipe n=$BEST_N_PIPE"
run --model "$CONFIRM_MODEL" --method fs_pipe \
    --split val --prompt-variant "$BEST_PV" --n-examples "$BEST_N_PIPE" --resume

log "Phase 1D complete. Analyzing..."

# Report confirmation results (all vars already exported)
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, os

variant = os.environ.get('BEST_PV', 'default')
best_n = int(os.environ.get('BEST_N_E2E', '3'))
runner_n = int(os.environ.get('RUNNER_UP_N_E2E', '3'))
best_pipe = int(os.environ.get('BEST_N_PIPE', '3'))
model = os.environ.get('CONFIRM_MODEL', 'claude-haiku-4.5')

def compute_tag(variant, n):
    parts = []
    if variant != 'default':
        parts.append(f'pv_{variant}')
    if n != 3:
        parts.append(f'n{n}')
    return '.' + '.'.join(parts) if parts else ''

def read_score(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get('macro', {}).get('relation_macro_f1', 0)
    return None

print(f'\n  Confirmation results ({model}):')
for label, method, n in [
    (f'fs_e2e n={best_n} (best)', 'fs_e2e', best_n),
    (f'fs_e2e n={runner_n} (runner-up)', 'fs_e2e', runner_n),
    (f'fs_pipe n={best_pipe}', 'fs_pipe', best_pipe),
]:
    tag = compute_tag(variant, n)
    s = read_score(f'results_val/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json')
    if s is not None:
        print(f'    {label:35s}: {s:.4f}')
    else:
        print(f'    {label:35s}: MISSING')

print(f'\n  Final hyperparameters (from {os.environ.get("PILOT_MODEL", "pilot")}):')
print(f'    Variant: {variant}')
print(f'    N_E2E:   {best_n}')
print(f'    N_Pipe:  {best_pipe}')
PYEOF

log ""
log "════════════════════════════════════════════════════════════════"
log "  PHASE 1 COMPLETE — Hyperparameters selected"
log "  Variant: $BEST_PV"
log "  N_E2E:   $BEST_N_E2E"
log "  N_Pipe:  $BEST_N_PIPE"
log "════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Run-to-Run Variance (test, 2 models)
# ═══════════════════════════════════════════════════════════════════════

log ""
log "══════ PHASE 2: Run-to-Run Variance ══════"
log "  Models: $VARIANCE_MODELS | 3 runs each | fs_e2e on test set"

for RUN_ID in $VARIANCE_RUNS; do
    for MODEL in $VARIANCE_MODELS; do
        log "  Running: $MODEL  fs_e2e  run=$RUN_ID"
        run --model "$MODEL" --method fs_e2e \
            --prompt-variant "$BEST_PV" --n-examples "$BEST_N_E2E" \
            --run-id "$RUN_ID" --resume
    done
done
log "Phase 2 runs complete. Verifying completeness..."

# Strict completeness check: fail if any expected Phase 2 file is missing or partial.
python3 << 'PYEOF' >> "$LOG" 2>&1
import json
import os
import sys

variant = os.environ.get("BEST_PV", "default")
best_n = int(os.environ.get("BEST_N_E2E", "3"))
models = os.environ.get("VARIANCE_MODELS", "").split()
run_ids = [1, 2, 3]

def compute_tag(variant, n, run_id):
    parts = []
    if variant != "default":
        parts.append(f"pv_{variant}")
    if n != 3:
        parts.append(f"n{n}")
    parts.append(f"run{run_id}")
    return "." + ".".join(parts)

missing = []
aborted = []
incomplete = []

for model in models:
    for run_id in run_ids:
        tag = compute_tag(variant, best_n, run_id)
        path = f"results/PersuasiveEssaysV2/metrics/{model}_fs_e2e{tag}.json"
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            data = json.load(f)
        if data.get("aborted", False):
            aborted.append(path)
        n_pred = data.get("n_predicted")
        n_essays = data.get("n_essays")
        if isinstance(n_pred, int) and isinstance(n_essays, int) and n_pred < n_essays:
            incomplete.append((path, n_pred, n_essays))

if missing or aborted or incomplete:
    print("ERROR: Phase 2 completeness check failed.")
    if missing:
        print(f"  Missing files ({len(missing)}):")
        for p in missing:
            print(f"    - {p}")
    if aborted:
        print(f"  Aborted runs ({len(aborted)}):")
        for p in aborted:
            print(f"    - {p}")
    if incomplete:
        print(f"  Partial runs ({len(incomplete)}):")
        for p, n_pred, n_essays in incomplete:
            print(f"    - {p}: n_predicted={n_pred}, n_essays={n_essays}")
    sys.exit(2)

print("Phase 2 completeness check passed.")
PYEOF

log "Phase 2 completeness check passed."
log "Phase 2 complete. Analyzing..."

# Report variance
export BEST_PV BEST_N_E2E
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, statistics

variant = os.environ.get('BEST_PV', 'default')
best_n = int(os.environ.get('BEST_N_E2E', '3'))
models = os.environ.get('VARIANCE_MODELS', 'gemini-3-flash-preview claude-haiku-4.5').split()

def compute_tag_for_run(variant, n, run_id):
    parts = []
    if variant != 'default':
        parts.append(f'pv_{variant}')
    if n != 3:
        parts.append(f'n{n}')
    parts.append(f'run{run_id}')
    return '.' + '.'.join(parts)

print('\n  Run-to-run variance (test set, fs_e2e):')
for model in models:
    scores = []
    for run_id in [1, 2, 3]:
        tag = compute_tag_for_run(variant, best_n, run_id)
        path = f'results/PersuasiveEssaysV2/metrics/{model}_fs_e2e{tag}.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            s = data.get('macro', {}).get('relation_macro_f1', 0)
            scores.append(s)
    print(f'  {model}:')
    for i, s in enumerate(scores, 1):
        print(f'    Run {i}: {s:.4f}')
    if len(scores) > 1:
        print(f'    Std:   {statistics.stdev(scores):.4f}')
PYEOF

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Main Experiments (test, all 4 models)
# ═══════════════════════════════════════════════════════════════════════

log ""
log "════════════════════════════════════════════════════════════════"
log "  PHASE 3: Main Experiments"
log "  Models: $ALL_MODELS"
log "  Variant: $BEST_PV | N_E2E: $BEST_N_E2E | N_Pipe: $BEST_N_PIPE"
log "════════════════════════════════════════════════════════════════"

# Build common variant args
PV_ARGS=""
if [ "$BEST_PV" != "default" ]; then
    PV_ARGS="--prompt-variant $BEST_PV"
fi

for MODEL in $ALL_MODELS; do
    log ""
    log "  ────── Model: $MODEL ──────"

    # --- Zero-shot methods (N irrelevant) ---
    for METHOD in $ZS_METHODS; do
        log "  Running: $METHOD"
        run --model "$MODEL" --method "$METHOD" $PV_ARGS --resume
    done

    # --- Few-shot E2E methods (use N_E2E) ---
    for METHOD in $E2E_FS_METHODS; do
        log "  Running: $METHOD  n=$BEST_N_E2E"
        run --model "$MODEL" --method "$METHOD" \
            $PV_ARGS --n-examples "$BEST_N_E2E" --resume
    done

    # --- Pipeline + gold_fs (use N_Pipe) ---
    for METHOD in $PIPE_FS_METHODS; do
        log "  Running: $METHOD  n=$BEST_N_PIPE"
        run --model "$MODEL" --method "$METHOD" \
            $PV_ARGS --n-examples "$BEST_N_PIPE" --resume
    done

    log "  Model $MODEL complete."
done
log "Phase 3 runs complete. Verifying completeness..."

# Strict completeness check: fail if any expected Phase 3 file is missing or partial.
python3 << 'PYEOF' >> "$LOG" 2>&1
import json
import os
import sys

variant = os.environ.get("BEST_PV", "default")
n_e2e = int(os.environ.get("BEST_N_E2E", "3"))
n_pipe = int(os.environ.get("BEST_N_PIPE", "3"))
models = os.environ.get("ALL_MODELS", "").split()

def compute_tag(variant, n=None):
    parts = []
    if variant != "default":
        parts.append(f"pv_{variant}")
    if n is not None and n != 3:
        parts.append(f"n{n}")
    return "." + ".".join(parts) if parts else ""

methods = [
    ("zs_e2e", None),
    ("fs_e2e", n_e2e),
    ("fs_cot_e2e", n_e2e),
    ("zs_pipe", None),
    ("fs_pipe", n_pipe),
    ("gold_zs", None),
    ("gold_fs", n_pipe),
]

missing = []
aborted = []
incomplete = []

for model in models:
    for method, n in methods:
        tag = compute_tag(variant, n)
        path = f"results/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json"
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            data = json.load(f)
        if data.get("aborted", False):
            aborted.append(path)
        n_pred = data.get("n_predicted")
        n_essays = data.get("n_essays")
        if isinstance(n_pred, int) and isinstance(n_essays, int) and n_pred < n_essays:
            incomplete.append((path, n_pred, n_essays))

if missing or aborted or incomplete:
    print("ERROR: Phase 3 completeness check failed.")
    if missing:
        print(f"  Missing files ({len(missing)}):")
        for p in missing:
            print(f"    - {p}")
    if aborted:
        print(f"  Aborted runs ({len(aborted)}):")
        for p in aborted:
            print(f"    - {p}")
    if incomplete:
        print(f"  Partial runs ({len(incomplete)}):")
        for p, n_pred, n_essays in incomplete:
            print(f"    - {p}: n_predicted={n_pred}, n_essays={n_essays}")
    sys.exit(2)

print("Phase 3 completeness check passed.")
PYEOF

log "Phase 3 completeness check passed."
log "Phase 3 complete."

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════

log ""
log "════════════════════════════════════════════════════════════════"
log "  ALL PHASES COMPLETE"
log "════════════════════════════════════════════════════════════════"

export BEST_PV BEST_N_E2E BEST_N_PIPE ALL_MODELS VARIANCE_MODELS
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, statistics

variant = os.environ.get('BEST_PV', 'default')
n_e2e = int(os.environ.get('BEST_N_E2E', '3'))
n_pipe = int(os.environ.get('BEST_N_PIPE', '3'))
models = os.environ.get('ALL_MODELS', '').split()

def compute_tag(variant, n):
    parts = []
    if variant != 'default':
        parts.append(f'pv_{variant}')
    if n != 3:
        parts.append(f'n{n}')
    return '.' + '.'.join(parts) if parts else ''

def read_score(path, key='relation_macro_f1'):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get('macro', {}).get(key, None)
    return None

methods_config = [
    ('zs_e2e',     'ZS E2E',      None),
    ('fs_e2e',     'FS E2E',      n_e2e),
    ('fs_cot_e2e', 'FS CoT',      n_e2e),
    ('zs_pipe',    'ZS Pipe',     None),
    ('fs_pipe',    'FS Pipe',     n_pipe),
    ('gold_zs',    'Gold ZS',     None),
    ('gold_fs',    'Gold FS',     n_pipe),
]

print('=' * 80)
print('EXPERIMENT RESULTS SUMMARY')
print(f'Variant: {variant} | N_E2E: {n_e2e} | N_Pipe: {n_pipe}')
print('=' * 80)

# Header
header = f'{"Model":28s}'
for _, display, _ in methods_config:
    header += f'  {display:>8s}'
print(header)
print('-' * 80)

for model in models:
    row = f'{model:28s}'
    for method, _, n in methods_config:
        tag = compute_tag(variant, n) if n is not None else (
            compute_tag(variant, 3) if variant != 'default' else ''
        )
        # Zero-shot methods: only variant tag, no n tag
        if n is None:
            tag_parts = []
            if variant != 'default':
                tag_parts.append(f'pv_{variant}')
            tag = '.' + '.'.join(tag_parts) if tag_parts else ''

        path = f'results/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json'
        s = read_score(path)
        if s is not None:
            row += f'  {s:8.3f}'
        else:
            row += f'  {"---":>8s}'
    print(row)

print('=' * 80)
print('Metric: Macro Relation F1 (macro-averaged across essays)')
print()

# Best result per model
print('Best E2E method per model:')
e2e_methods = [('zs_e2e', None), ('fs_e2e', n_e2e), ('fs_cot_e2e', n_e2e),
               ('zs_pipe', None), ('fs_pipe', n_pipe)]
for model in models:
    best_method, best_score = None, -1
    for method, n in e2e_methods:
        if n is None:
            tag_parts = []
            if variant != 'default':
                tag_parts.append(f'pv_{variant}')
            tag = '.' + '.'.join(tag_parts) if tag_parts else ''
        else:
            tag = compute_tag(variant, n)
        path = f'results/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json'
        s = read_score(path)
        if s is not None and s > best_score:
            best_score = s
            best_method = method
    if best_method:
        print(f'  {model:28s}: {best_method:15s} = {best_score:.3f}')

print()
PYEOF

log ""
log "  Hyperparameters: variant=$BEST_PV  N_E2E=$BEST_N_E2E  N_Pipe=$BEST_N_PIPE"
log "  Full log: $LOG"
log "  Done!"
