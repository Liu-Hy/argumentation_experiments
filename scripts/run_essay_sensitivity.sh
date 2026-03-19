#!/bin/bash
#
# Overnight runner for all 5 sensitivity experiment phases.
#
# Usage:
#   nohup bash scripts/run_essay_sensitivity.sh > /dev/null 2>&1 &
#
# Or interactively (see live progress):
#   bash scripts/run_essay_sensitivity.sh
#
# Prerequisites: OPENROUTER_API_KEY set in environment or .env file.
#
# Estimated: ~3,100 API calls, ~4-8 hours depending on model latency.
# All output is logged. Progress messages print to terminal.
# The script is restartable: --resume skips completed essays.
#
# Phases:
#   0  CoT rerun with fixed prompt (test set, all 6 existing models)
#   1  Prompt variant selection (val set, 2 models × 5 methods × 3 variants)
#   2  n-shot ablation (val set, 2 models, best variant from Phase 1)
#   3  Example set sensitivity (val set, 2 models, best variant + best n)
#   4  Run-to-run variance (test set, 2 models, 3 runs)
#

cd "$(dirname "$0")/.."

# ─── Setup ────────────────────────────────────────────────────────────────

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/sensitivity_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
    # Run experiment, log full output, continue on failure
    python run_essay_experiment.py "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "  WARNING: exit code $rc — continuing"
    fi
    return 0
}

# Load API key from .env if not already set
if [ -z "$OPENROUTER_API_KEY" ] && [ -f .env ]; then
    export "$(grep -v '^#' .env | grep OPENROUTER_API_KEY | head -1 | xargs)"
fi
if [ -z "$OPENROUTER_API_KEY" ]; then
    log "ERROR: OPENROUTER_API_KEY is not set. Aborting."
    exit 1
fi

log "============================================================"
log "  Sensitivity Experiments — Starting"
log "  Log: $LOG"
log "============================================================"

MODELS_SENS="gemini-3-flash-preview claude-haiku-4.5"
MODELS_COT="claude-sonnet-4.5 claude-haiku-4.5 gemini-3-flash-preview gemini-3-pro-preview gpt-5.2 gpt-5-mini"

# ─── PHASE 0: CoT Rerun ──────────────────────────────────────────────────
log ""
log "══════ PHASE 0: CoT Rerun (test set, 6 models) ══════"

# Clear old raw results so --resume doesn't load stale CoT outputs
for MODEL in $MODELS_COT; do
    if [ -d "results/PersuasiveEssaysV2/raw/$MODEL/fs_cot_e2e" ]; then
        log "  Clearing old CoT results for $MODEL"
        rm -rf "results/PersuasiveEssaysV2/raw/$MODEL/fs_cot_e2e"
    fi
done

for MODEL in $MODELS_COT; do
    log "  Running: $MODEL  fs_cot_e2e"
    run --model "$MODEL" --method fs_cot_e2e --resume
done
log "Phase 0 complete."

# ─── PHASE 1: Prompt Variant Selection ────────────────────────────────────
log ""
log "══════ PHASE 1: Prompt Variant Selection (val set, 2 models × 5 methods × 3 variants) ══════"

for VARIANT in default enhanced minimal; do
    for METHOD in zs_e2e fs_e2e fs_cot_e2e zs_pipe fs_pipe; do
        for MODEL in $MODELS_SENS; do
            log "  Running: $MODEL  $METHOD  variant=$VARIANT"
            run --model "$MODEL" --method "$METHOD" \
                --split val --prompt-variant "$VARIANT" --resume
        done
    done
done
log "Phase 1 complete. Analyzing..."

# Analyze Phase 1: pick best variant
BEST_VARIANT=$(python3 << 'PYEOF'
import json, os, sys

variants = ['default', 'enhanced', 'minimal']
models = ['gemini-3-flash-preview', 'claude-haiku-4.5']
methods = ['zs_e2e', 'fs_e2e', 'fs_cot_e2e', 'zs_pipe', 'fs_pipe']

scores = {v: [] for v in variants}
for v in variants:
    for method in methods:
        for model in models:
            tag = f'.pv_{v}' if v != 'default' else ''
            path = f'results_val/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                s = data.get('macro', {}).get('relation_macro_f1', 0)
                scores[v].append(s)
                print(f'  {v:10s} {method:15s} {model}: {s:.4f}', file=sys.stderr)

print('', file=sys.stderr)
for v in variants:
    avg = sum(scores[v]) / len(scores[v]) if scores[v] else 0
    print(f'  {v:10s} mean={avg:.4f} (n={len(scores[v])})', file=sys.stderr)

best = max(variants, key=lambda v: sum(scores[v])/len(scores[v]) if scores[v] else 0)
print(f'  >>> Selected: {best}', file=sys.stderr)
print(best)
PYEOF
) 2>> "$LOG"

BEST_VARIANT=$(echo "$BEST_VARIANT" | tr -d '[:space:]')
log "  Best variant: $BEST_VARIANT"

# ─── PHASE 2: n-Shot Ablation ─────────────────────────────────────────────
log ""
log "══════ PHASE 2: n-Shot Ablation (val set, variant=$BEST_VARIANT) ══════"

for N in 1 3 5 8; do
    for MODEL in $MODELS_SENS; do
        log "  Running: $MODEL  fs_e2e  n=$N  variant=$BEST_VARIANT"
        run --model "$MODEL" --method fs_e2e \
            --split val --prompt-variant "$BEST_VARIANT" --n-examples "$N" --resume
    done
done
log "Phase 2 complete. Analyzing..."

# Analyze Phase 2: pick best n
export BEST_VARIANT
BEST_N=$(python3 << 'PYEOF'
import json, os, sys

variant = os.environ['BEST_VARIANT']
models = ['gemini-3-flash-preview', 'claude-haiku-4.5']

scores = {}
for n in [1, 3, 5, 8]:
    scores[n] = []
    for model in models:
        parts = []
        if variant != 'default':
            parts.append(f'pv_{variant}')
        if n != 3:
            parts.append(f'n{n}')
        tag = '.'.join(parts) if parts else None
        eff = f'fs_e2e.{tag}' if tag else 'fs_e2e'
        path = f'results_val/PersuasiveEssaysV2/metrics/{model}_{eff}.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            s = data.get('macro', {}).get('relation_macro_f1', 0)
            scores[n].append(s)
            print(f'  n={n}  {model}: {s:.4f}', file=sys.stderr)

print('', file=sys.stderr)
for n in [1, 3, 5, 8]:
    avg = sum(scores[n]) / len(scores[n]) if scores[n] else 0
    print(f'  n={n}: mean={avg:.4f} (n_results={len(scores[n])})', file=sys.stderr)

best = max([1, 3, 5, 8], key=lambda n: sum(scores[n])/len(scores[n]) if scores[n] else 0)
print(f'  >>> Selected: {best}', file=sys.stderr)
print(best)
PYEOF
) 2>> "$LOG"

BEST_N=$(echo "$BEST_N" | tr -d '[:space:]')
log "  Best n: $BEST_N"

# ─── PHASE 3: Example Set Sensitivity ─────────────────────────────────────
log ""
log "══════ PHASE 3: Example Sensitivity (val set, variant=$BEST_VARIANT, n=$BEST_N) ══════"

for SEED in 0 1 2 3 4; do
    for MODEL in $MODELS_SENS; do
        log "  Running: $MODEL  fs_e2e  seed=$SEED  variant=$BEST_VARIANT  n=$BEST_N"
        run --model "$MODEL" --method fs_e2e \
            --split val --prompt-variant "$BEST_VARIANT" \
            --n-examples "$BEST_N" --example-seed "$SEED" --resume
    done
done
log "Phase 3 complete."

# ─── PHASE 4: Run-to-Run Variance ────────────────────────────────────────
log ""
log "══════ PHASE 4: Run-to-Run Variance (test set, 3 runs) ══════"

for RUN_ID in 1 2 3; do
    for MODEL in $MODELS_SENS; do
        log "  Running: $MODEL  fs_e2e  run=$RUN_ID"
        run --model "$MODEL" --method fs_e2e --run-id "$RUN_ID" --resume
    done
done
log "Phase 4 complete."

# ─── SUMMARY ──────────────────────────────────────────────────────────────
log ""
log "============================================================"
log "  ALL PHASES COMPLETE"
log "============================================================"
log "  Best variant: $BEST_VARIANT"
log "  Best n:       $BEST_N"
log ""

export BEST_VARIANT BEST_N
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, sys, statistics

variant = os.environ.get('BEST_VARIANT', 'default')
best_n = int(os.environ.get('BEST_N', '3'))
models = ['gemini-3-flash-preview', 'claude-haiku-4.5']

def read_score(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get('macro', {}).get('relation_macro_f1', 0)
    return None

print('=' * 60)
print('SENSITIVITY EXPERIMENTS — RESULTS SUMMARY')
print('=' * 60)

# Phase 1
print('\n--- Phase 1: Prompt Variant Selection (val) ---')
print(f'{"Variant":12s}  {"Avg Rel Macro F1":>18s}  {"n":>3s}')
for v in ['default', 'enhanced', 'minimal']:
    scores = []
    for method in ['zs_e2e', 'fs_e2e', 'fs_cot_e2e', 'zs_pipe', 'fs_pipe']:
        for model in models:
            tag = f'.pv_{v}' if v != 'default' else ''
            s = read_score(f'results_val/PersuasiveEssaysV2/metrics/{model}_{method}{tag}.json')
            if s is not None:
                scores.append(s)
    avg = sum(scores)/len(scores) if scores else 0
    mark = '  <<<' if v == variant else ''
    print(f'{v:12s}  {avg:18.4f}  {len(scores):3d}{mark}')

# Phase 2
print(f'\n--- Phase 2: n-Shot Ablation (val, variant={variant}) ---')
print(f'{"n":>4s}  {"Avg Rel Macro F1":>18s}')
for n in [1, 3, 5, 8]:
    scores = []
    for model in models:
        parts = []
        if variant != 'default':
            parts.append(f'pv_{variant}')
        if n != 3:
            parts.append(f'n{n}')
        tag = '.'.join(parts) if parts else None
        eff = f'fs_e2e.{tag}' if tag else 'fs_e2e'
        s = read_score(f'results_val/PersuasiveEssaysV2/metrics/{model}_{eff}.json')
        if s is not None:
            scores.append(s)
    avg = sum(scores)/len(scores) if scores else 0
    mark = '  <<<' if n == best_n else ''
    print(f'{n:4d}  {avg:18.4f}{mark}')

# Phase 3
print(f'\n--- Phase 3: Example Sensitivity (val, variant={variant}, n={best_n}) ---')
seed_avgs = []
for seed in range(5):
    scores = []
    for model in models:
        parts = []
        if variant != 'default':
            parts.append(f'pv_{variant}')
        if best_n != 3:
            parts.append(f'n{best_n}')
        parts.append(f'seed{seed}')
        tag = '.'.join(parts)
        s = read_score(f'results_val/PersuasiveEssaysV2/metrics/{model}_fs_e2e.{tag}.json')
        if s is not None:
            scores.append(s)
    avg = sum(scores)/len(scores) if scores else 0
    seed_avgs.append(avg)
    print(f'  seed={seed}: {avg:.4f}')

if len(seed_avgs) > 1:
    m, s = statistics.mean(seed_avgs), statistics.stdev(seed_avgs)
    print(f'  Mean +/- Std:  {m:.4f} +/- {s:.4f}')

# Phase 4
print('\n--- Phase 4: Run-to-Run Variance (test) ---')
for model in models:
    all_scores = []
    orig = read_score(f'results/PersuasiveEssaysV2/metrics/{model}_fs_e2e.json')
    if orig is not None:
        all_scores.append(orig)
    for r in [1, 2, 3]:
        s = read_score(f'results/PersuasiveEssaysV2/metrics/{model}_fs_e2e.run{r}.json')
        if s is not None:
            all_scores.append(s)
    print(f'  {model}:')
    if orig is not None:
        print(f'    Original:  {orig:.4f}')
    for i, s in enumerate(all_scores[1 if orig else 0:], 1):
        print(f'    Run {i}:     {s:.4f}')
    if len(all_scores) > 1:
        print(f'    Std:       {statistics.stdev(all_scores):.4f}')

print('\n' + '=' * 60)
print(f'SELECTED:  variant={variant}  n={best_n}')
print('=' * 60)
PYEOF

log ""
log "Full log: $LOG"
log "Done!"
