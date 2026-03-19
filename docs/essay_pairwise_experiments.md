# Pairwise Relation Classification Experiment

## 1. Motivation

Our main experiments show a clear bottleneck: argument span identification performs well (82–86% F1), but relation extraction lags behind (<30% Macro Relation F1 across all settings). The current pipeline Step 2 asks the model to predict **all** relations simultaneously—with ~15 arguments per essay, that means considering ~210 directed pairs in a single generation. This combinatorial complexity may be the root cause of poor relation performance, rather than a fundamental inability to recognize argumentative relations.

**Pairwise classification** isolates the relation recognition ability by decomposing Step 2 into individual pair classifications:

> For each unordered pair {a_i, a_j}, ask: *"Is there a support, attack, or no relation between these two arguments?"*

If pairwise classification substantially outperforms bulk relation prediction, the bottleneck is **combinatorial generation complexity**, not relational understanding. If it doesn't help, the task is genuinely hard for LLMs at the pair level, which would indicate a ceiling for prompting-based approaches on this dataset.

## 2. Method Design

### 2.1 New Methods

| Key | Name | Description |
|-----|------|-------------|
| `gold_fs_pairwise` | Gold-arg Few-shot Pairwise | Gold argument spans + pairwise relation classification with 3 example pairs (1 support, 1 attack, 1 none). |
| `fs_pipe_pairwise` | Pipeline Few-shot Pairwise | Same Step 1 as `fs_pipe` (argument extraction), then pairwise Step 2 with 3 example pairs. |
| `gold_fs_pw_graph` | Gold-arg Pairwise (Graph Context) | Gold argument spans + pairwise classification with **full training graph** as context. Shows all arguments and all relations from a training essay to calibrate density expectations. |
| `fs_pipe_pw_graph` | Pipeline Pairwise (Graph Context) | Pipeline Step 1 + pairwise Step 2 with full training graph context. |

### 2.2 Pairwise Prompt Design

Each pair call uses a focused prompt:

- **System**: Defines the 5 possible outputs (support in either direction, attack in either direction, or none). Emphasizes that most pairs have no relation.
- **User**: Few-shot examples (3 pairs: one support, one attack, one no-relation—all from a single training essay for compactness) + the test essay + the two arguments to classify.

**Output format** — the model returns exactly one JSON:
```json
{"source": "arg1", "target": "arg2", "type": "support"}
```
or `{"relation": "none"}`. The runner remaps `arg1`/`arg2` to actual argument IDs.

### 2.3 Few-Shot Examples for Pairwise

Rather than full-essay BAF examples, the pairwise prompt uses **argument-pair examples** extracted from training data:

1. **Support pair**: two arguments from a training essay with a gold support relation.
2. **Attack pair**: two arguments with a gold attack relation.
3. **No-relation pair**: two arguments from the same essay with no gold relation.

All three pairs come from a single training essay (the one selected for having both support and attack relations), so the example essay text appears only once in the prompt.

### 2.4 Graph-Context Prompt Design (pw_graph methods)

Initial results showed that the 3-pair pairwise prompt has a **prior bias problem**: 2 out of 3 example pairs have relations (67%), but the true base rate is ~8%. This leads to massive over-prediction (4.1x gold).

The graph-context variant addresses this by showing the **complete annotation** of a training essay:

- **All arguments** listed with sequential IDs
- **All relations** listed explicitly
- **Sparsity note**: "X out of Y possible pairs have relations. The remaining Z pairs have NO relation."

This lets the model see the true relation density (~10/105 ≈ 9.5%) and the structural pattern (tree-shaped, not dense graph) before classifying each test pair. The system prompt is identical; only the user message differs.

**Example prompt structure:**
```
Here is a complete annotation of a training essay, showing all arguments and their relations:

Essay: [training essay text]
Arguments identified (15 total):
  a1: "..." a2: "..." ...
Relations (10 out of 105 possible pairs):
  a3 -> a1: support  ...
The remaining 95 argument pairs have NO direct relation.
---
Now classify the relation between the following two arguments from a new essay:
Essay: [test essay]
Argument 1 (arg1): "..." Argument 2 (arg2): "..."
```

### 2.5 Execution Flow

**`gold_fs_pairwise` / `gold_fs_pw_graph`:**
1. Receive gold arguments (renumbered a1, a2, ...)
2. Generate all C(n, 2) unordered pairs
3. For each pair: call LLM → parse → classify as support/attack/none
4. Aggregate all identified relations into a BAF
5. Remap IDs back to original gold argument IDs

**`fs_pipe_pairwise` / `fs_pipe_pw_graph`:**
1. Step 1: extract arguments using `fs_pipe` Step 1 prompt (identical to existing pipeline)
2. Step 2: pairwise classification on extracted arguments (as above, but no ID remapping)

**Safety features:**
- Per-essay circuit breaker: aborts pairwise classification after 5 consecutive empty API responses.
- Progress logging every 25 pairs.

## 3. Experimental Plan

### Models

| Model | Key | Rationale |
|-------|-----|-----------|
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | Fast and cheap; good for high-volume pairwise calls |
| Claude Haiku 4.5 | `claude-haiku-4.5` | Fast and cheap; different model family for comparison |

### Commands

```bash
# Gold-argument pairwise (ceiling diagnostic)
for MODEL in gemini-3-flash-preview claude-haiku-4.5; do
  python run_essay_experiment.py --model $MODEL --method gold_fs_pairwise --resume
done

# Full pipeline pairwise
for MODEL in gemini-3-flash-preview claude-haiku-4.5; do
  python run_essay_experiment.py --model $MODEL --method fs_pipe_pairwise --resume
done

# Or run both at once:
for MODEL in gemini-3-flash-preview claude-haiku-4.5; do
  python run_essay_experiment.py --model $MODEL --method pairwise --resume
done

# Graph-context pairwise (addresses over-prediction via full training graph)
for MODEL in gemini-3-flash-preview claude-haiku-4.5; do
  python run_essay_experiment.py --model $MODEL --method pw_graph --resume
done
```

## 4. Cost Analysis

**Per essay:**
- Average ~15 gold arguments → C(15, 2) = 105 unordered pairs per essay.
- Each pair: ~500–800 input tokens (essay ~400 words + 2 arguments + examples), ~20–50 output tokens.

**Total:**
| Method | Pairs/Essay | Essays | Models | Total Calls |
|--------|------------|--------|--------|-------------|
| `gold_fs_pairwise` | ~105 | 80 | 2 | ~16,800 |
| `fs_pipe_pairwise` | ~105 + 1 (Step 1) | 80 | 2 | ~16,960 |
| **Total** | | | | **~33,760** |

**Estimated cost:** At Gemini Flash Lite / Haiku pricing (~$0.001–$0.002 per call), total cost is **~$35–$70**. Manageable for a targeted diagnostic experiment.

**Wall-clock time:** ~1–3 seconds per pair call → ~2–5 minutes per essay → ~3–7 hours per model per method. Use `--resume` to recover from interruptions.

## 5. Evaluation and Comparisons

The pairwise methods use the **same evaluation metrics** as all other methods (Argument F1, Support/Attack/Macro Relation F1). Key comparisons:

| Comparison | What It Shows |
|-----------|---------------|
| `gold_fs_pairwise` vs `gold_fs` | Does pairwise decomposition improve relation classification when arguments are given? |
| `gold_fs_pw_graph` vs `gold_fs_pairwise` | Does showing the full training graph fix the over-prediction problem? |
| `gold_fs_pw_graph` vs `gold_fs` | Can graph-calibrated pairwise beat the bulk approach? |
| `fs_pipe_pairwise` vs `fs_pipe` | Does pairwise Step 2 improve the full pipeline? |
| `gold_fs_pairwise` ceiling | Upper bound on what prompting-based relation extraction can achieve on this dataset. |

### Expected Outcomes

1. **If pairwise >> bulk:** The model understands individual relations but struggles to produce them all at once. This suggests structured decoding, iterative prompting, or finer decomposition could substantially improve results.

2. **If pairwise ≈ bulk:** The bottleneck is at the individual pair level. This points to inherent difficulty in the annotation scheme (e.g., ambiguous relations, implicit reasoning) and/or limitations of the models.

3. **If pairwise < bulk:** Unlikely, but would suggest that seeing the full graph context helps the model make better individual predictions.

## 6. Relation to Inter-Annotator Agreement

This experiment also speaks to the **dataset's performance ceiling**. The AAEC dataset has reported inter-annotator agreement of ~0.60 kappa for relation types (Stab & Gurevych 2017). If even gold-argument pairwise classification with a strong model doesn't substantially exceed 30–40% Macro Relation F1, it suggests that:

- The annotation granularity or relation definitions may be inherently ambiguous.
- The gap between IAA and model performance is smaller than it appears from bulk predictions.
- The relaxed-match evaluation (IoU ≥ 0.5) may interact poorly with relation evaluation when spans are approximate.

## 7. Implementation Summary

### Files Modified

| File | Changes |
|------|---------|
| `src/essay_prompts.py` | Added `_PAIRWISE_SYSTEM` prompt, `extract_pairwise_examples()`, `_format_pairwise_block()`, `pairwise_classify()`. Added `extract_graph_example()`, `_format_graph_block()`, `pairwise_classify_with_graph()` for graph-context variant. Updated `METHODS` registry. |
| `src/output_parser.py` | Added `parse_pairwise_response()` for single-pair JSON parsing with ID remapping. |
| `run_essay_experiment.py` | Added `PAIRWISE_METHODS`, `run_essay_pairwise()`, pairwise dispatch in main loop, `--method pairwise` and `--method pw_graph` CLI groups. Graph-context dispatch via `graph_example` parameter. |
