# Experiment Plan: LLM-Based Bipolar Argumentation Framework Extraction

## 1. Research Goal

Evaluate how well current LLMs can construct **Bipolar Argumentation Frameworks (BAFs)** from natural language text, using prompting-based methods (no fine-tuning).

## 2. Task Definition

**Input:** A persuasive essay (plain text).

**Output:** A Bipolar Argumentation Framework consisting of:

- **Arguments**: a set of text spans (exact verbatim substrings from the essay). All argumentation components (claims, premises, theses, counter-arguments) are treated uniformly as "arguments" — no component-type distinction.
- **Support relations**: directed edges where the source argument provides evidence or reasoning that supports the target.
- **Attack relations**: directed edges where the source argument provides evidence or reasoning that undermines the target.

This formulation is **dataset-agnostic**: it abstracts away corpus-specific component taxonomies (e.g., MajorClaim/Claim/Premise in Persuasive Essays, or different schemes in other corpora) and focuses on the minimal structure of a bipolar AF: nodes + directed support/attack edges.

## 3. Datasets

### 3.1 Primary: Argument Annotated Essays v2 (AAEC)

- **Source**: Stab & Gurevych, "Parsing Argumentation Structures in Persuasive Essays" (CL, 2017). https://aclanthology.org/J17-3005/
- **Size**: 402 persuasive essays (322 train / 80 test), using the official split from `train-test-split.csv`.
- **Annotations**: BRAT format. Original annotation uses MajorClaim, Claim, Premise as component types and supports/attacks as relation types, plus For/Against stance attributes on Claims.
- **Conversion to BAF**: MajorClaim, Claim, and Premise are all merged into generic "argument" nodes. Relation types are normalised (supports → support, attacks → attack). Stance attributes are dropped.
- **Key statistics** (after conversion): 6,089 arguments, 3,613 support relations, 219 attack relations (94.3% / 5.7% split).

### 3.2 Future Datasets

The evaluation framework is designed to be portable. Additional argumentation datasets can be integrated by writing a dataset-specific BRAT/other-format parser that outputs the same BAF representation.

## 4. Methods

Two orthogonal experimental axes — **prompting strategy** and **task decomposition** — plus a diagnostic setting:

### 4.1 Core Methods (5 total)

| Key | Name | Strategy | Decomposition | Description |
|-----|------|----------|---------------|-------------|
| `zs_e2e` | Zero-shot E2E | Zero-shot | End-to-end | Single prompt: task description + output schema. No examples. Baseline for raw LLM capability. |
| `fs_e2e` | Few-shot E2E | 3-shot | End-to-end | Same as above, with 3 annotated essay examples prepended. Tests value of in-context examples. |
| `fs_cot_e2e` | Few-shot CoT E2E | 3-shot + CoT | End-to-end | Few-shot examples + explicit step-by-step reasoning instructions (identify thesis → find supporting/opposing arguments → trace relations → output BAF). Tests whether guided reasoning improves structure extraction in the end-to-end setting. |
| `zs_pipe` | Zero-shot Pipeline | Zero-shot | 2-step pipeline | Step 1: identify argument spans. Step 2: given arguments, predict relations. Tests value of decomposition without examples. |
| `fs_pipe` | Few-shot Pipeline | 3-shot | 2-step pipeline | Same 2-step pipeline, with 3 examples at each step. Tests combined value of decomposition + examples. |

### 4.2 Gold-Argument Diagnostic Setting (2 variants)

| Key | Name | Description |
|-----|------|-------------|
| `gold_zs` | Gold-arg Zero-shot | Model receives gold argument spans; predicts only relations. Isolates relation extraction from span identification. |
| `gold_fs` | Gold-arg Few-shot | Same, with 3 few-shot examples of argument-to-relation mapping. |

The gap between gold-argument performance and end-to-end performance reveals whether the bottleneck is argument identification or relation extraction.

### 4.3 Design Rationale

- **CoT is paired with E2E, not pipeline.** Pipeline already decomposes the reasoning into steps; adding CoT within each pipeline step has diminishing returns. The more informative comparison is whether CoT can help the model handle the full complexity in a single pass. If `fs_cot_e2e` matches or exceeds `fs_pipe`, it suggests that explicit reasoning can substitute for task decomposition.
- **No fine-tuning.** The goal is to evaluate LLM capabilities via prompting, which is more practical for researchers without large compute budgets and more reflective of how LLMs are used in practice. Fine-tuning can be added in future work.
- **No dev set.** Prompts are designed based on task understanding and general prompt engineering principles. All 80 test essays are used for evaluation.

### 4.4 Few-Shot Example Selection

3 training essays are selected for diversity:

1. One essay **with attack relations** (medium length) — ensures the model sees the rarer relation type.
2. One **short/simple** essay (fewest arguments, no attacks) — provides a basic structural template.
3. One **long/complex** essay (most arguments) — demonstrates handling of larger argument graphs.

The same 3 examples are used across all test essays and all few-shot methods (fixed for reproducibility). Example IDs are saved to `results/meta/fewshot_examples.json`.

## 5. Models

| Key | Name | OpenRouter Model ID | Category |
|-----|------|---------------------|----------|
| `gpt-5-mini` | GPT-5 mini | `openai/gpt-5-mini` | Proprietary |
| `gpt-5-nano` | GPT-5 nano | `openai/gpt-5-nano` | Proprietary |
| `claude-haiku-4.5` | Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` | Proprietary |
| `gemini-3-flash-preview` | Gemini 3 Flash Preview | `google/gemini-3-flash-preview` | Proprietary |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | `google/gemini-2.5-flash-lite` | Proprietary |
| `kimi-k2.5` | Kimi K2.5 | `moonshotai/kimi-k2.5` | Open-source |
| `deepseek-v3.2` | DeepSeek V3.2 | `deepseek/deepseek-v3.2` | Open-source |
| `minimax-m2.1` | MiniMax M2.1 | `minimax/minimax-m2.1` | Open-source |
| `grok-4.1-fast` | Grok 4.1 Fast | `x-ai/grok-4.1-fast` | Proprietary |
| `qwen3-235b` | Qwen 3 235B A22B | `qwen/qwen3-235b-a22b-2507` | Open-source |

All accessed through **OpenRouter** (single API key, single billing account).

**Configuration**: temperature = 0 (deterministic), max_tokens = 4096.

### 5.1 Experiment Matrix

10 models × 5 core methods = **50 conditions**, each evaluated on 80 test essays.
Plus 10 models × 2 gold-arg variants = **20 diagnostic conditions**.

Total API calls: ~4,000 for core experiments (pipeline methods require 2 calls per essay), ~1,600 for diagnostic. Estimated cost at typical OpenRouter rates: modest (essays are short, ~300-500 words each).

## 6. Evaluation

### 6.1 Evaluation Settings

| Setting | Input to model | Model predicts | Purpose |
|---------|---------------|----------------|---------|
| **End-to-end** | Raw essay text | Arguments + relations | Full BAF extraction ability |
| **Gold-argument** | Essay + gold argument spans | Relations only | Isolates relation extraction |

### 6.2 Metrics

#### Argument Identification (end-to-end setting only)

- **Relaxed Span Match**: a predicted span matches a gold span if character-level IoU ≥ 0.5.
- **Optimal alignment**: Hungarian algorithm for bipartite matching between predicted and gold spans (maximises total IoU).
- **Precision / Recall / F1**: TP = matched pairs, FP = unmatched predictions, FN = unmatched golds.

#### Relation Extraction (both settings)

A predicted relation (source, target, type) counts as a **true positive** iff:

1. The source argument matches a gold argument (IoU ≥ 0.5).
2. The target argument matches a gold argument (IoU ≥ 0.5).
3. A gold relation exists between those matched gold arguments with the **same type**.

Computed metrics:

- **Per-type P / R / F1**: separately for support and attack.
- **Macro Relation F1**: average of support-F1 and attack-F1. This is the **primary metric** — it prevents the dominant support class (94.3%) from masking poor attack performance.
- **Micro Relation F1**: pooled TP/FP/FN across both types. Reflects overall accuracy.

#### Aggregation Levels

- **Micro-aggregated**: pool TP/FP/FN across all 80 test essays, then compute P/R/F1. Gives more weight to essays with more annotations.
- **Macro-averaged**: compute F1 per essay, then average across essays. Gives equal weight to each essay.
- **Bootstrap 95% CIs**: 10,000 bootstrap resamples over the 80 test essays for macro-averaged F1. Reports [2.5th, 97.5th percentile] intervals.

#### Practical Metrics

- **Parse success rate**: % of LLM outputs that yield valid JSON conforming to the expected schema and containing at least one argument.
- **Latency and token usage**: logged per call for cost analysis.

## 7. Experimental Procedure

### Phase 1: Data Preparation

1. Parse all 402 BRAT `.ann` + `.txt` files into unified BAF format (merge component types, normalise relation types).
2. Load official train/test split (322 / 80).
3. Select 3 few-shot examples from the training set.

### Phase 2: Main Experiments

For each (model, method) combination:

1. Iterate over all 80 test essays.
2. Construct prompt(s) using the appropriate template.
3. Call LLM via OpenRouter (temperature = 0).
4. Parse output: extract JSON, fuzzy-match quoted text to essay character offsets.
5. Save raw LLM output + parsed BAF per essay.
6. Compute all metrics.
7. Save aggregate results with bootstrap CIs.

Pipeline methods execute two sequential API calls per essay (step 1 output feeds into step 2). Errors in step 1 propagate naturally.

The `--resume` flag allows re-running without repeating completed essays.

### Phase 3: Analysis

1. **Main results table**: models (rows) × methods (columns), reporting macro relation F1 with 95% CIs.
2. **Per-type breakdown**: support-F1 vs attack-F1 for each condition.
3. **Argument identification table**: span P/R/F1 for all end-to-end methods.
4. **Gold-argument vs end-to-end gap**: quantifies whether the bottleneck is span identification or relation extraction.
5. **Parse success rates**: identifies model/method combinations with output compliance issues.
6. **Error analysis** on ~20 test essays from the best-performing condition, categorising errors as:
   - Boundary errors (span too long/short)
   - Missing vs hallucinated arguments
   - Missing vs hallucinated relations
   - Attack blindness (failure to detect rare attack relations)
   - Structural issues (cycles, disconnected components)

## 8. Hypotheses

| # | Hypothesis | Comparison |
|---|-----------|-----------|
| H1 | Few-shot examples substantially improve BAF extraction | `fs_e2e` vs `zs_e2e` |
| H2 | CoT reasoning helps relation identification more than argument identification | Sub-metric comparison: `fs_cot_e2e` vs `fs_e2e` |
| H3 | Pipeline decomposition helps, especially for relation extraction | `fs_pipe` vs `fs_e2e` |
| H4 | Reasoning-specialised models (o3-mini) improve structural coherence | `o3-mini` vs `gpt-4o` on same methods |
| H5 | Attack relations are disproportionately harder than support relations | Support-F1 vs Attack-F1 across all conditions |
| H6 | The bottleneck is relation extraction, not argument identification | Gold-arg F1 vs end-to-end F1 gap |

## 9. Suggested Execution Order

Given limited budget, run in this priority order (publishable results from step 3 onward):

1. `zs_e2e` on GPT-4o, Claude 3.5 Sonnet, DeepSeek-V3 — feasibility check.
2. `fs_e2e` on same 3 models — measure few-shot gain.
3. `fs_pipe` on best model from step 2 — measure pipeline gain.
4. `fs_cot_e2e` on best model — measure CoT gain.
5. `gold_fs` on all 4 models — diagnostic for bottleneck analysis.
6. Fill remaining cells of the model × method matrix.
7. Error analysis.
