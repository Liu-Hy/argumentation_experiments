# Integrated Experiment Plan: Full Rerun from Scratch

## Context

All previous results (6 models on test set, sensitivity experiments on val set) were
generated under different API configurations. The recent changes --- enabling
reasoning/thinking for all models (`is_reasoning=True`), increasing `max_tokens` to
16384, and adjusting temperature handling --- fundamentally change model behavior.
Results are not comparable across configurations, so we start clean.

## Models

| Key | Name | Category | Notes |
|-----|------|----------|-------|
| `gemini-3-flash-preview` | Gemini 3 Flash Preview | Proprietary | HP search pilot model |
| `gpt-5-mini` | GPT-5 mini | Proprietary | `temperature=None` (OpenAI reasoning) |
| `claude-haiku-4.5` | Claude Haiku 4.5 | Proprietary | HP confirmation model |
| `kimi-k2.5` | Kimi K2.5 | Open-source | |

All accessed through OpenRouter. Reasoning enabled (`effort="medium"`) for all models.

## Methods

| Key | Name | Few-shot? | N group |
|-----|------|-----------|---------|
| `zs_e2e` | Zero-shot E2E | No | --- |
| `fs_e2e` | Few-shot E2E | Yes | N_E2E |
| `fs_cot_e2e` | Few-shot CoT E2E | Yes | N_E2E |
| `zs_pipe` | Zero-shot Pipeline | No | --- |
| `fs_pipe` | Few-shot Pipeline | Yes | N_Pipe |
| `gold_zs` | Gold-arg Zero-shot | No | --- |
| `gold_fs` | Gold-arg Few-shot | Yes | N_Pipe |

---

## Phase 0: Preparation

### 0A. Archive old results

```bash
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p archive/pre_rerun_${TS}
mv results/ archive/pre_rerun_${TS}/results 2>/dev/null
mv results_val/ archive/pre_rerun_${TS}/results_val 2>/dev/null
```

### 0B. Pre-flight model check

The script's built-in pre-flight check verifies all 4 required models are reachable
via OpenRouter. It runs automatically at the start of `scripts/run_essay_experiments.sh` and
aborts if any model is unavailable.

For a quick non-strict check without launching the full pipeline:

```bash
python run_essay_experiment.py --check
```

Note: `--check` reports model status but does not hard-fail with a non-zero exit code.
`scripts/run_essay_experiments.sh` contains the strict pre-flight gate used for overnight runs.

### 0C. Commit code snapshot

All code changes should be committed before the first API call for traceability.

---

## Phase 1: Hyperparameter Search (Validation Set)

**Pilot model**: Gemini 3 Flash Preview (cost-efficient, high parse rate).
**Eval set**: 40 validation essays (stratified split from training, `seed=42`).

All results stored in `results_val/` with auto-generated tags for full traceability.

### Phase 1A: Prompt Variant Selection

**Goal**: Pick one global prompt variant for all methods.

3 variants (default, enhanced, minimal) x 5 methods x 1 model x 40 essays
= 600 runs (~840 effective API calls with pipeline doubles).

**Decision rule**: Pick the variant with the highest average macro Relation F1
across the 5 methods. If `default` is within 1% of the best variant, prefer
`default` (simpler).

### Phase 1B: N-Shot Ablation

**Goal**: Find the best number of few-shot examples per method group.

N in {1, 3, 5, 10}, searched with the best variant from Phase 1A:

| Method | Calls | Determines |
|--------|-------|------------|
| fs_e2e | 4 x 40 = 160 | **N_E2E** (for fs_e2e, fs_cot_e2e) |
| fs_pipe | 4 x 40 x 2 = 320 | **N_Pipe** (for fs_pipe, gold_fs) |

Total: 480 effective API calls.

**Decision rule**: Pick the N with the highest Relation Macro F1. If multiple N
values are within 2% of the best, prefer the smallest (lower cost, shorter prompts).

### Phase 1C: Example Sensitivity

**Goal**: Quantify variance from the specific choice of few-shot examples.

5 random seeds x fs_e2e x best variant x best N_E2E x 40 essays = 200 API calls.

Report mean +/- std of Relation Macro F1.

### Phase 1D: Confirmation on Second Model

**Goal**: Verify that hyperparameters chosen on Gemini Flash transfer to Claude Haiku.

- Best config (best variant + best N_E2E) on fs_e2e: 40 calls
- Runner-up N on fs_e2e: 40 calls
- Best config on fs_pipe: 80 calls

Total: ~120-160 calls.

**If Claude Haiku disagrees on best N**: Keep Gemini Flash's choice (the primary
search model). Report the discrepancy in the paper appendix.

### Phase 1 Cost Summary

| Sub-phase | Calls | Model |
|-----------|-------|-------|
| 1A: Variant | ~840 | Gemini Flash |
| 1B: N-shot | ~480 | Gemini Flash |
| 1C: Seeds | ~200 | Gemini Flash |
| 1D: Confirm | ~160 | Claude Haiku |
| **Total** | **~1,680** | |

---

## Phase 2: Run-to-Run Variance (Test Set)

**Goal**: Verify reproducibility under the new reasoning-enabled API parameters
before committing to the full run.

2 models (Gemini Flash + Claude Haiku) x 3 runs x fs_e2e x 80 essays = **480 calls**.

Expected: Relation Macro F1 varies by <= +/-2%. This also serves as a feasibility
check for parse rates, latencies, and metric ranges.

---

## Phase 3: Main Experiments (Test Set, All 4 Models)

With hyperparameters locked in from Phase 1:

| Method | Prompt variant | N examples |
|--------|---------------|------------|
| zs_e2e | BEST_PV | --- |
| fs_e2e | BEST_PV | N_E2E |
| fs_cot_e2e | BEST_PV | N_E2E |
| zs_pipe | BEST_PV | --- |
| fs_pipe | BEST_PV | N_Pipe |
| gold_zs | BEST_PV | --- |
| gold_fs | BEST_PV | N_Pipe |

Per-model call count:

| Methods | Calls |
|---------|-------|
| 3 single-call (zs_e2e, fs_e2e, fs_cot_e2e) x 80 | 240 |
| 2 pipeline (zs_pipe, fs_pipe) x 80 x 2 steps | 320 |
| 2 gold (gold_zs, gold_fs) x 80 | 160 |
| **Total per model** | **720** |

4 models x 720 = **2,880 API calls**.

---

## Phase 4: Post-Hoc Analysis (No API Calls)

### 4A: Neural Evaluation

Run `run_essay_neural_eval.py --all` for BERTScore + NLI on all saved results.
GPU/CPU-bound, no API cost.

### 4B: Results Generation

Update `generate_essay_results_xlsx.py` for the 4-model matrix and tagged method names.

### 4C: Error Analysis

~20 test essays from the best-performing condition: categorize errors as boundary
errors, missing/hallucinated arguments, missing/hallucinated relations, attack
blindness, structural issues.

---

## Design Notes

### Reasoning Models + Explicit CoT = Double Reasoning

All models now use built-in reasoning (thinking tokens via OpenRouter). The
`fs_cot_e2e` method adds explicit CoT instructions on top. This double-reasoning
is theoretically redundant and may reduce the output token budget for the actual
JSON answer. Previous findings that CoT hurts are likely reinforced.

### N=10 Context Cost

With N=10 few-shot examples, input prompts reach ~20K-30K tokens of examples alone.
All 4 models have 100K+ context windows, so this is technically fine, but it
increases per-call input cost substantially. The N-shot ablation will reveal whether
the gains justify the cost.

### Support/Attack Class Imbalance

Attack relations are ~5.7% of all relations. The primary metric (Macro Relation F1)
amplifies noise in the rare class. Report micro Relation F1 as a complementary
stable metric.

### temperature=None for OpenAI Models

GPT-5 mini rejects the temperature parameter, so it is omitted. Other models use
temperature=0.0. If GPT-5 mini shows higher run-to-run variance, this asymmetry is
the likely cause.

---

## Cost Summary

| Phase | API Calls | Models | Set |
|-------|-----------|--------|-----|
| 1: HP search | ~1,680 | Gemini Flash + Claude Haiku | Val (40) |
| 2: Variance | ~480 | Gemini Flash + Claude Haiku | Test (80) |
| 3: Main | ~2,880 | All 4 | Test (80) |
| **Total** | **~5,040** | | |

---

## Execution

Phases 0-3 are automated in `scripts/run_essay_experiments.sh` (Phase 4 is post-hoc/manual).
The script:

- Supports fresh rerun mode (`--fresh`, default): archives old results before starting
- Supports resume mode (`--resume-run`): keeps existing results and continues from cache
- Runs each phase sequentially (HP search requires analysis between sub-phases)
- Uses `--resume` throughout experiment calls to skip completed essays
- Auto-selects best hyperparameters between phases via inline Python analysis
- Logs all output to `logs/` and prints a final summary

```bash
# Interactive (see live progress):
bash scripts/run_essay_experiments.sh --fresh

# Background (overnight):
mkdir -p logs && nohup bash scripts/run_essay_experiments.sh --fresh > logs/launcher.log 2>&1 &

# Resume after interruption:
bash scripts/run_essay_experiments.sh --resume-run
```
