# Implementation Plan: BAF Extraction Experiment Framework

## 1. Overview

A Python framework for running and evaluating LLM-based Bipolar Argumentation Framework (BAF) extraction experiments. All LLM calls are routed through **OpenRouter**, requiring only a single API key. The codebase is designed to be dataset-agnostic where possible, with dataset-specific logic isolated in the data loader.

## 2. Dependencies

```
numpy>=1.24
scipy>=1.10
openai>=1.0
bert-score>=0.3.13
transformers>=4.30.0
torch>=2.0.0
```

- `numpy`: numerical operations.
- `scipy`: Hungarian algorithm for optimal span matching (`linear_sum_assignment`). A greedy fallback is provided if scipy is unavailable.
- `openai`: OpenAI-compatible SDK, used to call OpenRouter's API endpoint.
- `bert-score`: BERTScore computation for AF-coverage metrics.
- `transformers`: model loading for NLI-based faithfulness metrics.
- `torch`: deep learning backend for BERTScore and NLI inference.

No `anthropic` SDK needed — OpenRouter translates OpenAI-format messages for all providers (including Anthropic models).

Install: `pip install -r requirements.txt`

**Note**: `bert-score`, `transformers`, and `torch` are only required for the neural evaluation step (`run_essay_neural_eval.py`). The main experiment runner (`run_essay_experiment.py`) does not depend on them.

## 3. Project Structure

```
argumentation_experiments/
├── PersuasiveEssaysV2/              # Dataset (existing, unchanged)
│   ├── brat-project-final/          #   402 .ann + .txt files
│   ├── train-test-split.csv
│   ├── prompts.csv
│   ├── guideline.pdf
│   └── README.txt
├── src/                             # Library modules
│   ├── __init__.py
│   ├── baf.py                       # Core data structures
│   ├── essay_data_loader.py          # Dataset parsing
│   ├── evaluation.py                # Metrics computation (span-based)
│   ├── neural_metrics.py            # Neural metrics (BERTScore, NLI)
│   ├── output_parser.py             # LLM output → BAF
│   ├── essay_prompts.py              # Prompt templates
│   └── llm_client.py                # OpenRouter API client
├── run_essay_experiment.py           # CLI entry point (LLM experiments)
├── run_essay_neural_eval.py         # CLI entry point (neural evaluation)
├── requirements.txt
├── docs/essay_experiment_plan.md     # Research design
└── docs/essay_implementation_plan.md # This file
```

Output directories (created automatically at runtime):

```
results/
├── raw/{model}/{method}/            # Per-essay JSON (raw LLM output + parsed BAF)
│   └── essay001.json ... essay402.json
├── metrics/                         # Aggregated span-based evaluation results
│   └── {model}_{method}.json
├── neural_metrics/                  # Aggregated neural evaluation results
│   └── {model}_{method}.json
└── meta/                            # Experiment metadata
    └── fewshot_examples.json
```

## 4. Module Details

### 4.1 `src/baf.py` — Core Data Structures

Three dataclasses that represent a Bipolar Argumentation Framework:

- **`Argument`**: `id`, `text`, `start` (char offset), `end` (char offset). Provides `iou(other)` for character-level Intersection-over-Union between spans.
- **`Relation`**: `source` (argument id), `target` (argument id), `type` ("support" or "attack").
- **`BAF`**: lists of `Argument` and `Relation`. Provides `to_dict()`, `to_json()`, `from_dict()`, `from_json()` for serialisation.

These structures are dataset-agnostic — no reference to MajorClaim/Claim/Premise or any corpus-specific taxonomy.

### 4.2 `src/essay_data_loader.py` — Dataset Parsing

Dataset-specific module. Currently supports Persuasive Essays v2 (BRAT format).

**Key functions:**

- `parse_brat_ann(ann_path, txt_path) -> (BAF, essay_text)`: Parses one BRAT `.ann` file. Merges MajorClaim/Claim/Premise into generic arguments. Maps `supports` → `support`, `attacks` → `attack`. Ignores stance attributes.
- `load_persuasive_essays(dataset_dir) -> dict`: Loads all 402 essays with their BAFs and train/test split labels.
- `get_split(dataset, split) -> dict`: Filters to "train" or "test".
- `select_fewshot_examples(train_data, n=3) -> list[str]`: Selects diverse training essays for few-shot prompts: one with attack relations, one simple, one complex.

**To add a new dataset:** write a new loader function (e.g., `load_my_dataset()`) that returns the same `dict[str, {"baf": BAF, "text": str, "split": str}]` format.

### 4.3 `src/evaluation.py` — Metrics

Fully dataset-agnostic. Takes predicted and gold `BAF` objects as input.

**Argument matching:**

- `match_arguments(pred_args, gold_args, iou_threshold=0.5)`: Builds a cost matrix of -IoU values, solves optimal bipartite matching via Hungarian algorithm. Returns `dict[pred_id -> gold_id]` for pairs with IoU ≥ threshold.

**Per-essay evaluation:**

- `evaluate_essay(pred_baf, gold_baf, iou_threshold=0.5)`: Returns argument P/R/F1, per-type relation P/R/F1, macro and micro relation F1.

**Dataset-level evaluation:**

- `evaluate_dataset(predictions, golds, ...)`: Computes micro-aggregated metrics (pooled TP/FP/FN) and macro-averaged metrics (per-essay F1 averaged).
- `format_results(results)`: Pretty-prints a summary table to stdout.

**Relation evaluation criterion:** a predicted relation is a true positive iff both its source and target arguments match gold arguments (via the span matching) AND the relation type matches.

### 4.4 `src/output_parser.py` — LLM Output Parsing

Converts raw LLM text output into a `BAF` object. Handles common failure modes.

**JSON extraction** (`_extract_json`):

1. Try content inside `` ```json ... ``` `` fences.
2. Try outermost `{ ... }` block (brace matching).
3. Try the full text as JSON.
4. Tolerate trailing commas (common LLM mistake).

**Span resolution** (`_parse_arguments` + `_fuzzy_find`):

LLMs are instructed to quote text verbatim, but may paraphrase or truncate. The parser:

1. If the LLM provided `start`/`end` offsets, validate them (check that the text at those offsets matches the quoted text with ≥ 0.8 similarity ratio).
2. Otherwise, try exact substring match in the essay.
3. Fallback: fuzzy matching via `difflib.SequenceMatcher` with a sliding window. Accepts matches with ratio ≥ 0.6.

**Relation parsing** (`_parse_relations`):

- Normalises type variations (`"supports"` → `"support"`, `"attacks"` → `"attack"`).
- Filters out relations referencing unknown argument IDs.

**Gold-argument setting** (`parse_relations_only`):

- For the diagnostic setting where gold arguments are provided. Re-uses gold argument objects and only parses the relations from the output.

### 4.5 `src/essay_prompts.py` — Prompt Templates

Implements all 5 methods plus 2 gold-argument variants. Each function returns a `list[dict]` of OpenAI-format messages (system + user).

**Shared components:**

- `TASK_DESCRIPTION`: Defines BAF, argument, support, and attack in natural language.
- `OUTPUT_SCHEMA`: JSON schema with strict rules (verbatim text, valid IDs, only support/attack).
- `OUTPUT_SCHEMA_COT`: Same schema but prefixed with step-by-step reasoning instructions.

**Method implementations:**

| Function | Method | Notes |
|----------|--------|-------|
| `zs_e2e(essay)` | Zero-shot E2E | System: task + schema. User: essay. |
| `fs_e2e(essay, examples)` | Few-shot E2E | User message includes 3 formatted examples before the test essay. |
| `fs_cot_e2e(essay, examples)` | Few-shot CoT E2E | Uses `OUTPUT_SCHEMA_COT` which asks for reasoning before JSON. |
| `zs_pipe_step1(essay)` | Pipeline step 1 (ZS) | Argument identification only. |
| `zs_pipe_step2(essay, args_json)` | Pipeline step 2 (ZS) | Relation prediction given arguments. |
| `fs_pipe_step1(essay, examples)` | Pipeline step 1 (FS) | With examples. |
| `fs_pipe_step2(essay, args_json, examples)` | Pipeline step 2 (FS) | With examples. |
| `gold_arg_relations(essay, gold_baf, examples?)` | Gold-argument | Provides gold spans, asks for relations only. |

**Example formatting:** `_format_example()` re-assigns sequential IDs (a1, a2, ...) for clarity in few-shot examples.

### 4.6 `src/llm_client.py` — OpenRouter API Client

All LLM calls go through **OpenRouter** (`https://openrouter.ai/api/v1`), which provides a single OpenAI-compatible endpoint for all model providers.

**Setup:** set `OPENROUTER_API_KEY` environment variable. Get a key at https://openrouter.ai/keys.

**Model registry** (`MODELS` dict): maps short keys to `ModelConfig(name, model_id, max_tokens, temperature)`. The `temperature` field is `Optional[float]`: set to `None` for reasoning models (e.g., GPT-5.2) that do not support the parameter; the API call omits it in that case. Pre-configured models:

| Key | OpenRouter model ID | Notes |
|-----|---------------------|-------|
| `gpt-5-mini` | `openai/gpt-5-mini` | |
| `gpt-5-nano` | `openai/gpt-5-nano` | |
| `gpt-5.2` | `openai/gpt-5.2` | Reasoning model; `temperature=None` |
| `claude-haiku-4.5` | `anthropic/claude-haiku-4.5` | |
| `claude-sonnet-4.5` | `anthropic/claude-sonnet-4.5` | SOTA frontier |
| `gemini-3-flash-preview` | `google/gemini-3-flash-preview` | |
| `gemini-3-pro-preview` | `google/gemini-3-pro-preview` | SOTA frontier |
| `gemini-2.5-flash-lite` | `google/gemini-2.5-flash-lite` | |
| `kimi-k2.5` | `moonshotai/kimi-k2.5` | |
| `deepseek-v3.2` | `deepseek/deepseek-v3.2` | |
| `minimax-m2.1` | `minimax/minimax-m2.1` | |
| `grok-4.1-fast` | `x-ai/grok-4.1-fast` | |
| `qwen3-235b` | `qwen/qwen3-235b-a22b-2507` | |

**To add a model:** add an entry to the `MODELS` dict with the OpenRouter model ID (browse https://openrouter.ai/models).

**`LLMClient`**: lazy-initialises a single `openai.OpenAI` client. `generate(messages, model_key)` sends a chat completion request and returns an `LLMResponse` containing content, usage stats, and latency. Retries up to 3 times with exponential backoff (2s, 5s, 15s) on transient errors. Returns empty content (not an exception) if all retries fail, so the experiment loop continues.

### 4.7 `run_essay_experiment.py` — CLI Entry Point

Orchestrates the full experiment. Loads data, iterates over essays, calls LLMs, parses outputs, evaluates, and saves results.

**CLI interface:**

```bash
# List available models and methods
python run_essay_experiment.py --list

# Run all 7 methods on one model
python run_essay_experiment.py --model gpt-4o

# Run a specific method
python run_essay_experiment.py --model gpt-4o --method fs_e2e

# Run method groups
python run_essay_experiment.py --model gpt-4o --method e2e    # zs_e2e, fs_e2e, fs_cot_e2e
python run_essay_experiment.py --model gpt-4o --method pipe   # zs_pipe, fs_pipe
python run_essay_experiment.py --model gpt-4o --method gold   # gold_zs, gold_fs

# Resume a partial run (skip essays with existing results)
python run_essay_experiment.py --model gpt-4o --method zs_e2e --resume

# Custom dataset path
python run_essay_experiment.py --model gpt-4o --dataset /path/to/data/PersuasiveEssaysV2
```

**Data flow per essay:**

1. `build_messages()` dispatches to the appropriate prompt template function.
2. `LLMClient.generate()` sends the request via OpenRouter and returns the raw response.
3. `parse_llm_output()` (or `parse_relations_only()` for gold-arg) converts the raw text into a `BAF` object.
4. The raw output + parsed BAF are saved to `results/{dataset_name}/raw/{model}/{method}/{essay_id}.json`.
5. After all essays, `evaluate_dataset()` computes metrics and saves to `results/{dataset_name}/metrics/{model}_{method}.json`.

**Pipeline methods** execute step 1 first, then feed its raw output into step 2. Parse errors in step 1 propagate naturally (an empty argument list means step 2 has nothing to work with).

**Resume mode** (`--resume`): checks for existing result files before making API calls. Allows re-running after interruptions without re-processing completed essays.

### 4.8 `src/neural_metrics.py` — Neural Evaluation Metrics

Reference-free neural metrics that compare a generated BAF against its source essay. Runs as a post-processing step — no GPU dependency for the main experiment loop.

**Serialization functions:**

- `serialize_arguments(baf) -> str`: Concatenates argument texts sorted by ID (natural sort: a2 < a10), newline-separated. Used as the BERTScore candidate.
- `serialize_af_statements(baf) -> list[str]`: Returns individual statement strings for NLI. Argument statements (raw text) followed by relation statements (`"{source.text}. This supports/challenges the argument that {target.text}"`).

**Model loading:**

- `_get_nli_model(model_name, device)`: Lazy singleton loader for NLI model (tokenizer + model). Loaded on first use, cached for subsequent calls. Avoids loading both models simultaneously.

**Metric functions:**

- `bertscore_af_coverage(baf, essay_text, ...)`: BERTScore between concatenated argument texts and essay. Returns precision (faithfulness), recall (coverage), F1, and per-argument precision variant. Uses `microsoft/deberta-xlarge-mnli` with `rescale_with_baseline=True`.
- `nli_af_faithfulness(baf, essay_text, ...)`: NLI-based faithfulness. Serializes AF into statements, computes P(entailment) for each against the essay. Returns overall mean, argument-only mean, relation-only mean, min, std, fraction above 0.5. Uses `microsoft/deberta-v2-xlarge-mnli`.

**Aggregation:**

- `evaluate_essay_neural(pred_baf, gold_baf, essay_text, ...)`: Combined evaluation for one essay. Evaluates both prediction and gold (ceiling).
- `evaluate_dataset_neural(predictions, golds, essay_texts, ...)`: Dataset-level aggregation of neural metrics across essays.

**Edge cases:**

- Empty BAF (no arguments): returns all-zero metrics without errors.
- Missing predictions: treated as empty BAF.
- Arguments with IDs referenced in relations but not present in argument list: relation statements are silently skipped.

### 4.9 `run_essay_neural_eval.py` — Neural Evaluation CLI

Post-hoc neural evaluation of saved BAF extraction results. Reads per-essay JSONs from `results/{dataset_name}/raw/` and writes aggregated neural metrics to `results/{dataset_name}/neural_metrics/`.

**CLI interface:**

```bash
# Evaluate one (model, method) pair
python run_essay_neural_eval.py --model gemini-3-flash-preview --method zs_e2e

# Evaluate all saved results
python run_essay_neural_eval.py --all

# List available results
python run_essay_neural_eval.py --list

# Force CPU
python run_essay_neural_eval.py --model gemini-3-flash-preview --method fs_e2e --device cpu
```

**Data flow:**

1. Load dataset via `load_persuasive_essays()` + `get_split("test")`.
2. Discover/load saved predictions from `results/{dataset_name}/raw/{model}/{method}/`.
3. Reconstruct predicted BAFs via `BAF.from_dict(data["pred_baf"])`.
4. Call `evaluate_dataset_neural()` with predictions, golds, and essay texts.
5. Save results to `results/{dataset_name}/neural_metrics/{model}_{method}.json`.
6. Print formatted summary table.

**Device auto-detection:** defaults to CUDA if available, falls back to CPU.

## 5. Output Format

### 5.1 Per-Essay Raw Results (`results/{dataset_name}/raw/{model}/{method}/{essay_id}.json`)

```json
{
  "essay_id": "essay001",
  "method": "fs_e2e",
  "model": "gpt-4o",
  "raw_output": "```json\n{\"arguments\": [...], ...}\n```",
  "usage": {"prompt_tokens": 2340, "completion_tokens": 512, "total_tokens": 2852},
  "latency_s": 3.21,
  "pred_baf": {
    "arguments": [{"id": "a1", "text": "...", "start": 503, "end": 575}, ...],
    "relations": [{"source": "a2", "target": "a1", "type": "support"}, ...]
  },
  "parse_success": true
}
```

Pipeline methods additionally include `step1_raw`, `step2_raw`, `step1_usage`, `step2_usage`.

### 5.2 Aggregate Metrics (`results/{dataset_name}/metrics/{model}_{method}.json`)

```json
{
  "micro": {
    "argument": {"precision": 0.85, "recall": 0.78, "f1": 0.81, "tp": 980, "fp": 173, "fn": 276},
    "relation": {
      "support": {"precision": 0.65, "recall": 0.58, "f1": 0.61, ...},
      "attack": {"precision": 0.12, "recall": 0.08, "f1": 0.10, ...},
      "macro": {"f1": 0.355},
      "micro": {"precision": 0.62, "recall": 0.55, "f1": 0.58, ...}
    }
  },
  "macro": {
    "argument_f1": 0.79,
    "support_f1": 0.59,
    "attack_f1": 0.09,
    "relation_macro_f1": 0.34
  },
  "parse_success_rate": 0.975,
  "per_essay_summary": {
    "essay003": {"arg_f1": 0.82, "sup_f1": 0.67, "att_f1": 0.0, "rel_macro_f1": 0.33},
    ...
  }
}
```

## 6. Extending the Framework

### Adding a new model

Edit `MODELS` in `src/llm_client.py`:

```python
MODELS["my-model"] = ModelConfig(
    name="My Model",
    model_id="provider/model-name",  # OpenRouter model ID
)
```

### Adding a new dataset

1. Write a loader function in `src/essay_data_loader.py` that returns `dict[str, {"baf": BAF, "text": str, "split": str}]`.
2. Call it from `run_essay_experiment.py` instead of (or in addition to) `load_persuasive_essays()`.
3. Everything downstream (evaluation, prompts, output parsing) works unchanged.

### Adding a new prompting method

1. Add a prompt-building function in `src/essay_prompts.py`.
2. Register it in the `METHODS` dict.
3. Add a dispatch case in `build_messages()` in `run_essay_experiment.py`.
4. Add the method key to the appropriate list (`E2E_METHODS`, `PIPE_METHODS`, etc.).
