# Sensitivity Experiments & Enhancements

This document describes the changes made to improve the experimental rigor of the BAF extraction evaluation, and the plan for running additional sensitivity experiments. The goal is to provide enough evidence to convince an informed reviewer at a top ML venue that the baselines are fairly tested and results are stable.

## 1. What Changed

### 1.1 CoT Prompt Fix (Bug Fix)

**File**: `src/essay_prompts.py` — `OUTPUT_SCHEMA_COT`

The original Chain-of-Thought prompt contained reasoning steps that were inconsistent with the task definition:

```
OLD (problematic):
1. Identify the author's main thesis or central claim.
2. Identify arguments that support this thesis and arguments that oppose it.
3. For each argument, determine what evidence or reasoning supports or attacks it.
4. Trace the full network of support and attack relations.
```

This frames the task as thesis-centric hierarchical analysis, contradicting our BAF formulation where **all** components (claims, premises, theses) are treated uniformly as arguments with no special root node. The mismatch likely confused models into misidentifying the task structure, contributing to the unexpected finding that CoT hurt performance across all models (-3% to -15% Relation Macro F1).

```
NEW (fixed):
1. Read through the essay and identify all argumentative text spans — any
   claim, premise, thesis, evidence, or counter-argument. They are all
   treated uniformly as "arguments". Note each as a verbatim quote.
2. For each pair of arguments, consider whether one provides evidence or
   reasoning that supports or undermines the other. Only note relations
   where there is a clear argumentative connection.
3. Pay special attention to opposing viewpoints, counterpoints, and
   concessions — these often form attack relations.
4. Compile your analysis into the JSON format below.
```

**Key improvements**: (a) treats all arguments uniformly — no special "thesis" step, (b) directly maps to the pairwise relation extraction task, (c) explicitly draws attention to the rare attack class.

**Action required**: Re-run all models on `fs_cot_e2e` to override existing results.

### 1.2 Prompt Variant System (New Feature)

**File**: `src/essay_prompts.py`

Three prompt variants are now available for all methods (E2E, pipeline, CoT, gold-argument):

| Variant | Description |
|---------|-------------|
| `default` | Current prompts (with CoT fixed). Used for main results. |
| `enhanced` | Adds: argument granularity guidance (clause/sentence-length), expected count hint (5–20 args), relation direction clarification, attack awareness hints, sparsity note ("not all pairs have relations"). |
| `minimal` | Stripped to bare task definition + JSON schema. No role-playing, no detailed rules. Tests how much prompt engineering matters vs. raw model capability. |

All prompt-building functions now accept an optional `variant` parameter (defaults to `"default"` for full backward compatibility).

### 1.3 Validation Split (New Feature)

**File**: `src/essay_data_loader.py` — `create_val_split()`

Splits the 322 training essays into **282 train-proper + 40 validation**, using stratified sampling to preserve the proportion of essays with attack relations (~30% in training). The split is seeded (`seed=42`) for reproducibility. When evaluating on the validation set, few-shot examples are drawn only from train-proper (no leakage).

### 1.4 Random Example Selection (New Feature)

**File**: `src/essay_data_loader.py` — `select_fewshot_examples_random()`

A seeded random selection function for few-shot examples, ensuring at least one selected essay contains attack relations (stratified sampling). Different seeds yield different example sets, enabling sensitivity analysis.

### 1.5 Extended CLI (New Feature)

**File**: `run_essay_experiment.py`

New command-line arguments:

| Argument | Default | Purpose |
|----------|---------|---------|
| `--split {test,val}` | `test` | Evaluation split |
| `--prompt-variant {default,enhanced,minimal}` | `default` | Prompt variant |
| `--n-examples N` | `3` | Number of few-shot examples |
| `--example-seed S` | `None` | Seed for random example selection |
| `--val-size N` | `40` | Validation set size |
| `--tag TAG` | auto | Label appended to method name in result paths |
| `--run-id R` | `None` | Run identifier for variance experiments |

**Tag auto-generation**: When non-default settings are used, a tag is automatically generated from the changed parameters (e.g., `--prompt-variant enhanced` → `pv_enhanced`, `--n-examples 5` → `n5`, combined as `pv_enhanced.n5`). An explicit `--tag` overrides the auto-generated value.

### 1.6 Other Prompts — No Changes

The prompts for zero-shot E2E, few-shot E2E, pipeline (both steps), and gold-argument settings were reviewed and found to be consistent with the unified BAF formulation. No modifications were needed.

## 2. Result Directory Structure

```
results/                         # Test set (official 80-essay split)
├── raw/{model}/{method}/        # Standard runs (unchanged)
├── raw/{model}/{method}.{tag}/  # Tagged runs (variance, etc.)
├── metrics/                     # Aggregated metrics JSONs
└── meta/                        # Few-shot example IDs

results_val/                     # Validation set (40 held-out training essays)
├── raw/{model}/{method}/        # Default-variant runs
├── raw/{model}/{method}.{tag}/  # Tagged runs (prompt variants, n-shot, seeds)
├── metrics/                     # Aggregated metrics JSONs
└── meta/                        # Few-shot example IDs + experiment config
```

Metrics files include a `config` block recording the exact experimental settings (split, variant, n_examples, example_seed, fewshot_ids) for full traceability.

## 3. Experimental Plan

### Phase 0: CoT Rerun (Test Set)

Re-run all 6 existing models with the fixed CoT prompt. Results overwrite the old `fs_cot_e2e` files in-place.

```bash
for MODEL in claude-sonnet-4.5 claude-haiku-4.5 gemini-3-flash-preview \
             gemini-3-pro-preview gpt-5.2 gpt-5-mini; do
  python run_essay_experiment.py --model $MODEL --method fs_cot_e2e
done
```

**Cost**: 6 models × 80 essays = 480 API calls.

### Phase 1: Prompt Variant Selection (Validation Set)

Test all 3 prompt variants on the validation set using 2 representative models across all 5 core methods. This determines whether the default prompt is adequate or whether the enhanced prompt yields meaningfully better results.

```bash
for VARIANT in default enhanced minimal; do
  for METHOD in zs_e2e fs_e2e fs_cot_e2e zs_pipe fs_pipe; do
    python run_essay_experiment.py --model gemini-3-flash-preview --method $METHOD \
      --split val --prompt-variant $VARIANT
    python run_essay_experiment.py --model claude-haiku-4.5 --method $METHOD \
      --split val --prompt-variant $VARIANT
  done
done
```

**Cost**: 3 variants × 5 methods × 2 models × 40 essays = 1,200 calls (pipeline methods count as 2 calls each, so effective total ~1,440).

**Decision rule**: If the enhanced variant improves Relation Macro F1 by ≥ 3% consistently across methods and models on val, adopt it for all subsequent test-set runs. Otherwise, keep the default and report the sensitivity analysis in the paper's appendix.

### Phase 2: n-Shot Ablation (Validation Set)

With the best prompt variant selected from Phase 1, test the effect of the number of few-shot examples. Replace `BEST_PV` below with the selected variant (`default`, `enhanced`, or `minimal`).

```bash
for N in 1 3 5 8; do
  python run_essay_experiment.py --model gemini-3-flash-preview --method fs_e2e \
    --split val --prompt-variant BEST_PV --n-examples $N
  python run_essay_experiment.py --model claude-haiku-4.5 --method fs_e2e \
    --split val --prompt-variant BEST_PV --n-examples $N
done
```

**Cost**: 4 values × 2 models × 40 essays = 320 calls. The n=3 run reuses Phase 1 results (same settings, same path).

**Note**: For n=1, the `ensure_attacks` flag in random selection is disabled (only one example). If the default deterministic selector is used (no `--example-seed`), it picks the heuristic best single example. For a fairer test, consider adding `--example-seed 0` to use random selection for all n values consistently.

### Phase 3: Few-Shot Example Sensitivity (Validation Set)

With the best prompt variant and best n selected, test sensitivity to the specific choice of few-shot examples using 5 different random seeds. Replace `BEST_PV` and `BEST_N` below.

```bash
for SEED in 0 1 2 3 4; do
  python run_essay_experiment.py --model gemini-3-flash-preview --method fs_e2e \
    --split val --prompt-variant BEST_PV --n-examples BEST_N --example-seed $SEED
  python run_essay_experiment.py --model claude-haiku-4.5 --method fs_e2e \
    --split val --prompt-variant BEST_PV --n-examples BEST_N --example-seed $SEED
done
```

**Cost**: 5 seeds × 2 models × 40 essays = 400 calls.

**What to report**: Mean ± std of Relation Macro F1 across the 5 seeds. If std < 0.02, the single-set results are trustworthy and no further action is needed. If std is large, report averaged results and consider using the best seed (or a fixed diverse-selection heuristic) for test-set runs.

### Phase 4: Run-to-Run Variance (Test Set)

Verify that temperature=0 yields stable results across repeated API calls. Uses 2 models on the full test set (or a 20-essay subset for lower cost).

```bash
for RUN in 1 2 3; do
  python run_essay_experiment.py --model gemini-3-flash-preview --method fs_e2e \
    --run-id $RUN
  python run_essay_experiment.py --model claude-haiku-4.5 --method fs_e2e \
    --run-id $RUN
done
```

**Cost**: 3 runs × 2 models × 80 essays = 480 calls (or 120 calls if using a 20-essay subset with early stopping).

**What to report**: Compare the 3 tagged runs against the original untagged test result. Report per-essay exact-match rate and metric variance. A one-sentence note in the paper ("Results were stable across 3 independent runs, with Relation Macro F1 varying by at most ±X") suffices if variance is low.

### Phase 5: Final Test Set Runs (If Prompt Changed)

If Phase 1 determined that a non-default variant is substantially better:

```bash
# Re-run all existing models with the better prompt on affected methods
for MODEL in claude-sonnet-4.5 claude-haiku-4.5 gemini-3-flash-preview \
             gemini-3-pro-preview gpt-5.2 gpt-5-mini; do
  for METHOD in zs_e2e fs_e2e fs_cot_e2e zs_pipe fs_pipe gold_zs gold_fs; do
    python run_essay_experiment.py --model $MODEL --method $METHOD \
      --prompt-variant BEST_PV
  done
done
```

If the default prompt was confirmed as best (or within noise), no re-running is needed.

### Phase 6: Remaining Models

After all hyperparameters are finalized, run the remaining 7 models from the experiment plan:

```bash
for MODEL in gpt-5-nano gemini-2.5-flash-lite kimi-k2.5 deepseek-v3.2 \
             minimax-m2.1 grok-4.1-fast qwen3-235b; do
  python run_essay_experiment.py --model $MODEL --method all \
    --prompt-variant BEST_PV --n-examples BEST_N
done
```

## 4. Cost Summary

| Phase | Description | API Calls |
|-------|-------------|-----------|
| 0 | CoT rerun (6 models, test) | ~480 |
| 1 | Prompt selection (2 models × 5 methods × 3 variants, val) | ~1,440 |
| 2 | n-shot ablation (2 models × 4 values, val) | ~320 |
| 3 | Example sensitivity (2 models × 5 seeds, val) | ~400 |
| 4 | Run-to-run variance (2 models × 3 runs, test) | ~480 |
| 5 | Test re-run if prompt changed (6 models × 7 methods) | 0–3,360 |
| **Total (if prompt doesn't change)** | | **~3,120** |
| **Total (if prompt changes)** | | **~6,480** |

This is a 30–60% overhead on top of the existing ~5,200 calls — a reasonable investment for addressing the primary reviewer concerns.

## 5. What This Buys (Reviewer Concerns Addressed)

| Concern | Evidence |
|---------|----------|
| "How sensitive are results to the prompt?" | Phase 1: 3 variants × 5 methods × 2 models on val. Reported as a table in the appendix. |
| "Why 3 examples? Did you try other values?" | Phase 2: n ∈ {1, 3, 5, 8} ablation on val. Reported as a figure. |
| "Are results stable across different example sets?" | Phase 3: 5 random seeds, mean ± std reported. |
| "What's the variance from a single run?" | Phase 4: 3 repeated runs, variance reported. |
| "Did the CoT prompt hurt because of a design flaw?" | Phase 0: fixed CoT re-run. Before/after comparison. |
| "Were prompts tuned on the test set?" | No — prompt selection on held-out val set, test results reported separately. |
