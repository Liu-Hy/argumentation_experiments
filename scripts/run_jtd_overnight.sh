#!/usr/bin/env bash
#
# Overnight run: all JTD baselines (B0-B6, BH) x 3 models
#
# Models:
#   1. gemini-3-flash-preview  (Gemini 3 Flash)
#   2. kimi-k2.5               (Kimi K2.5)
#   3. minimax-m2.5            (MiniMax M2.5)
#
# Run order per model:
#   BH  (heuristic, instant, no LLM)
#   B2  (Dung AF, needed before B3)
#   B3  (reuses B2's AFs, no extra LLM calls)
#   B4  (BAF)
#   B5  (QuAF)
#   B6  (QBAF)
#   B0  (direct LLM)
#   B1  (AF-guided LLM, 2 calls/case)
#
# With --subsample 100 (default): 100 * 6 = 600 LLM calls per model
#   (B3 reuses B2; BH has none; B1 uses 2 calls/case so effectively 700)
#
# Usage:
#   chmod +x scripts/run_jtd_overnight.sh
#   nohup ./scripts/run_jtd_overnight.sh > logs/jtd_overnight.log 2>&1 &
#
#   # Full dataset run:
#   SUBSAMPLE=0 ./scripts/run_jtd_overnight.sh
#
#   # Custom sample size:
#   SUBSAMPLE=500 ./scripts/run_jtd_overnight.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=("gemini-3-flash-preview" "kimi-k2.5" "minimax-m2.5")
# Subsample size: 0 = use all data, >0 = stratified subsample
SUBSAMPLE="${SUBSAMPLE:-100}"
# Order: BH first (instant), B2 before B3 (reuse), B1 last (most expensive)
METHODS=("BH" "B2" "B3" "B4" "B5" "B6" "B0" "B1")

DATASET="./data/japanese-tort-case/Japanese_tort_training.jsonl"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/jtd_overnight_${TIMESTAMP}.log"

echo "=========================================="  | tee -a "$MAIN_LOG"
echo "JTD Overnight Experiment Run"               | tee -a "$MAIN_LOG"
echo "Started: $(date)"                            | tee -a "$MAIN_LOG"
echo "Models: ${MODELS[*]}"                        | tee -a "$MAIN_LOG"
echo "Methods: ${METHODS[*]}"                      | tee -a "$MAIN_LOG"
if [ "$SUBSAMPLE" -gt 0 ] 2>/dev/null; then
    echo "Subsample: $SUBSAMPLE cases"             | tee -a "$MAIN_LOG"
    SUBSAMPLE_FLAG="--subsample $SUBSAMPLE"
else
    echo "Subsample: all cases (full dataset)"     | tee -a "$MAIN_LOG"
    SUBSAMPLE_FLAG=""
fi
echo "=========================================="  | tee -a "$MAIN_LOG"

for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        RUN_LOG="$LOG_DIR/jtd_${MODEL}_${METHOD}_${TIMESTAMP}.log"

        echo ""                                                    | tee -a "$MAIN_LOG"
        echo ">>> $(date) | Starting: $MODEL / $METHOD"            | tee -a "$MAIN_LOG"
        echo "    Log: $RUN_LOG"                                   | tee -a "$MAIN_LOG"

        # shellcheck disable=SC2086
        python3 run_jtd_experiment.py \
            --model "$MODEL" \
            --method "$METHOD" \
            --dataset "$DATASET" \
            --resume \
            $SUBSAMPLE_FLAG \
            > "$RUN_LOG" 2>&1 \
            || {
                echo "    FAILED (exit $?) -- see $RUN_LOG"        | tee -a "$MAIN_LOG"
                # Continue to next method; don't abort the whole run
                continue
            }

        echo "    Done: $(date)"                                   | tee -a "$MAIN_LOG"

        # Print summary line from metrics file if it exists
        METRICS_FILE="results_jtd/metrics/${MODEL}_${METHOD}.json"
        if [ -f "$METRICS_FILE" ]; then
            python3 -c "
import json, sys
with open('$METRICS_FILE') as f:
    d = json.load(f)
sb = d.get('summary_tp_b', {})
re = sb.get('RE Micro-F1', {})
tp = sb.get('TP Accuracy', {})
print(f'    RE-F1={re.get(\"mean\",0):.4f}+/-{re.get(\"std\",0):.4f}  '
      f'TP-Acc={tp.get(\"mean\",0):.4f}+/-{tp.get(\"std\",0):.4f}')
" | tee -a "$MAIN_LOG"
        fi
    done
done

echo ""                                                | tee -a "$MAIN_LOG"
echo "=========================================="      | tee -a "$MAIN_LOG"
echo "All experiments completed: $(date)"              | tee -a "$MAIN_LOG"
echo "=========================================="      | tee -a "$MAIN_LOG"
