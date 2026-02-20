#!/usr/bin/env python3
"""Main experiment runner for BAF extraction evaluation.

All LLM calls go through OpenRouter (https://openrouter.ai).
Set the OPENROUTER_API_KEY environment variable before running.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."

    # Run all methods on a single model
    python run_experiment.py --model gpt-4o --dataset ./PersuasiveEssaysV2

    # Run a specific method
    python run_experiment.py --model gpt-4o --method fs_e2e

    # Run gold-argument setting
    python run_experiment.py --model gpt-4o --method gold_fs

    # List available models and methods
    python run_experiment.py --list

    # Resume from a partial run (skips essays with existing results)
    python run_experiment.py --model gpt-4o --method zs_e2e --resume

    # --- Sensitivity experiments ---

    # Evaluate on validation split (40 held-out training essays)
    python run_experiment.py --model claude-haiku-4.5 --method fs_e2e --split val

    # Test a prompt variant on validation set
    python run_experiment.py --model claude-haiku-4.5 --method fs_e2e \\
        --split val --prompt-variant enhanced

    # Change number of few-shot examples
    python run_experiment.py --model claude-haiku-4.5 --method fs_e2e \\
        --split val --n-examples 5

    # Use a random example set (for sensitivity analysis)
    python run_experiment.py --model claude-haiku-4.5 --method fs_e2e \\
        --split val --example-seed 42

    # Tag to distinguish multiple runs (auto-generated if not provided)
    python run_experiment.py --model claude-haiku-4.5 --method fs_e2e \\
        --tag run2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add parent dir to path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baf import BAF
from src.data_loader import (
    create_val_split,
    get_split,
    load_persuasive_essays,
    select_fewshot_examples,
    select_fewshot_examples_random,
)
from src.evaluation import evaluate_dataset, format_results
from src.llm_client import LLMClient, MODELS, LLMResponse
from src.output_parser import parse_llm_output, parse_relations_only
from src import prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("experiment")

# All end-to-end and pipeline methods
E2E_METHODS = ["zs_e2e", "fs_e2e", "fs_cot_e2e"]
PIPE_METHODS = ["zs_pipe", "fs_pipe"]
GOLD_METHODS = ["gold_zs", "gold_fs"]
ALL_METHODS = E2E_METHODS + PIPE_METHODS + GOLD_METHODS


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def _results_dir(base: str, model_key: str, method: str, split: str = "test") -> str:
    if split == "val":
        d = os.path.join(base, "results_val", "raw", model_key, method)
    else:
        d = os.path.join(base, "results", "raw", model_key, method)
    os.makedirs(d, exist_ok=True)
    return d


def _save_raw(base: str, model_key: str, method: str, essay_id: str, data: dict,
              split: str = "test"):
    d = _results_dir(base, model_key, method, split)
    with open(os.path.join(d, f"{essay_id}.json"), "w") as f:
        json.dump(data, f, indent=2)


def _load_raw(base: str, model_key: str, method: str, essay_id: str,
              split: str = "test") -> Optional[dict]:
    if split == "val":
        path = os.path.join(base, "results_val", "raw", model_key, method, f"{essay_id}.json")
    else:
        path = os.path.join(base, "results", "raw", model_key, method, f"{essay_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Prompt dispatch
# ---------------------------------------------------------------------------


def build_messages(
    method: str,
    essay_text: str,
    examples: List[Dict],
    gold_baf: Optional[BAF] = None,
    step1_output: Optional[str] = None,
    variant: str = "default",
) -> List[Dict[str, str]]:
    """Build the prompt messages for a given method."""
    if method == "zs_e2e":
        return prompts.zs_e2e(essay_text, variant=variant)
    elif method == "fs_e2e":
        return prompts.fs_e2e(essay_text, examples, variant=variant)
    elif method == "fs_cot_e2e":
        return prompts.fs_cot_e2e(essay_text, examples, variant=variant)
    elif method == "zs_pipe":
        if step1_output is None:
            return prompts.zs_pipe_step1(essay_text, variant=variant)
        else:
            return prompts.zs_pipe_step2(essay_text, step1_output, variant=variant)
    elif method == "fs_pipe":
        if step1_output is None:
            return prompts.fs_pipe_step1(essay_text, examples, variant=variant)
        else:
            return prompts.fs_pipe_step2(essay_text, step1_output, examples, variant=variant)
    elif method == "gold_zs":
        return prompts.gold_arg_relations(essay_text, gold_baf, examples=None, variant=variant)
    elif method == "gold_fs":
        return prompts.gold_arg_relations(essay_text, gold_baf, examples=examples, variant=variant)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Per-essay execution
# ---------------------------------------------------------------------------


def run_essay(
    client: LLMClient,
    model_key: str,
    method: str,
    essay_id: str,
    essay_text: str,
    gold_baf: BAF,
    examples: List[Dict],
    variant: str = "default",
) -> Dict:
    """Run one method on one essay.  Returns a result dict."""
    is_pipe = method in PIPE_METHODS
    is_gold = method in GOLD_METHODS

    if is_pipe:
        # Step 1: identify arguments
        msgs1 = build_messages(method, essay_text, examples, variant=variant)
        resp1: LLMResponse = client.generate(msgs1, model_key)
        step1_output = resp1.content

        # Parse step 1 to check if we got any arguments
        pred_baf, stats = _parse_pipe_output(step1_output, None, essay_text)

        if len(pred_baf.arguments) == 0:
            # Skip step 2 if step 1 yielded nothing (saves an API call)
            return {
                "essay_id": essay_id,
                "method": method,
                "model": model_key,
                "step1_raw": resp1.content,
                "step2_raw": "",
                "step1_usage": resp1.usage,
                "step2_usage": {},
                "latency_s": resp1.latency_s,
                "pred_baf": pred_baf.to_dict(),
                "parse_stats": stats.to_dict(),
                "parse_success": False,
            }

        # Step 2: predict relations
        msgs2 = build_messages(method, essay_text, examples,
                               step1_output=step1_output, variant=variant)
        resp2: LLMResponse = client.generate(msgs2, model_key)

        # Parse combined result
        pred_baf, stats = _parse_pipe_output(step1_output, resp2.content, essay_text)

        return {
            "essay_id": essay_id,
            "method": method,
            "model": model_key,
            "step1_raw": resp1.content,
            "step2_raw": resp2.content,
            "step1_usage": resp1.usage,
            "step2_usage": resp2.usage,
            "latency_s": resp1.latency_s + resp2.latency_s,
            "pred_baf": pred_baf.to_dict(),
            "parse_stats": stats.to_dict(),
            "parse_success": len(pred_baf.arguments) > 0,
        }

    elif is_gold:
        msgs = build_messages(method, essay_text, examples, gold_baf=gold_baf,
                              variant=variant)
        resp: LLMResponse = client.generate(msgs, model_key)
        pred_baf, stats = parse_relations_only(resp.content, gold_baf)

        return {
            "essay_id": essay_id,
            "method": method,
            "model": model_key,
            "raw_output": resp.content,
            "usage": resp.usage,
            "latency_s": resp.latency_s,
            "pred_baf": pred_baf.to_dict(),
            "parse_stats": stats.to_dict(),
            "parse_success": stats.json_extracted,
        }

    else:
        # E2E methods
        msgs = build_messages(method, essay_text, examples, variant=variant)
        resp: LLMResponse = client.generate(msgs, model_key)
        pred_baf, stats = parse_llm_output(resp.content, essay_text)

        return {
            "essay_id": essay_id,
            "method": method,
            "model": model_key,
            "raw_output": resp.content,
            "usage": resp.usage,
            "latency_s": resp.latency_s,
            "pred_baf": pred_baf.to_dict(),
            "parse_stats": stats.to_dict(),
            "parse_success": len(pred_baf.arguments) > 0,
        }


def _parse_pipe_output(
    step1_raw: str, step2_raw: Optional[str], essay_text: str
) -> tuple:
    """Combine pipeline step outputs into a single BAF.

    Returns (BAF, ParseStats).  If step2_raw is None (step 2 not yet run),
    returns arguments only.
    """
    from src.output_parser import (
        ParseStats, _extract_json, _parse_arguments, _parse_relations,
    )

    stats = ParseStats()

    # Parse step 1: arguments
    data1 = _extract_json(step1_raw)
    if data1 is None:
        return BAF(), stats

    stats.json_extracted = True
    raw_args = data1.get("arguments", [])
    stats.args_in_json = len(raw_args)
    arguments = _parse_arguments(raw_args, essay_text, stats)
    arg_ids = {a.id for a in arguments}

    if step2_raw is None:
        return BAF(arguments=arguments, relations=[]), stats

    # Parse step 2: relations
    data2 = _extract_json(step2_raw)
    if data2 is None:
        return BAF(arguments=arguments, relations=[]), stats
    raw_rels = data2.get("relations", [])
    stats.rels_in_json = len(raw_rels)
    relations = _parse_relations(raw_rels, arg_ids, stats)

    return BAF(arguments=arguments, relations=relations), stats


# ---------------------------------------------------------------------------
# Tag computation
# ---------------------------------------------------------------------------


def _compute_tag(args) -> Optional[str]:
    """Auto-generate a result tag from non-default experimental settings.

    Returns None if all settings are at their defaults (standard run).
    """
    if args.tag is not None:
        return args.tag

    parts = []
    if args.prompt_variant != "default":
        parts.append(f"pv_{args.prompt_variant}")
    if args.n_examples != 3:
        parts.append(f"n{args.n_examples}")
    if args.example_seed is not None:
        parts.append(f"seed{args.example_seed}")
    if args.run_id is not None:
        parts.append(f"run{args.run_id}")

    return ".".join(parts) if parts else None


def _effective_method(method: str, tag: Optional[str]) -> str:
    """Append tag to method name for result paths."""
    return f"{method}.{tag}" if tag else method


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    dataset_dir: str,
    base_dir: str,
    model_key: str,
    methods: List[str],
    resume: bool = False,
    split: str = "test",
    prompt_variant: str = "default",
    n_examples: int = 3,
    example_seed: Optional[int] = None,
    tag: Optional[str] = None,
    val_size: int = 40,
):
    """Run the full experiment for one model across specified methods."""
    logger.info(f"Loading dataset from {dataset_dir}")
    dataset = load_persuasive_essays(dataset_dir)
    train_data = get_split(dataset, "train")
    logger.info(f"Loaded {len(train_data)} train essays")

    # Determine evaluation data and example source
    if split == "val":
        example_source, eval_data = create_val_split(train_data, val_size=val_size)
        logger.info(
            f"Val split: {len(example_source)} train-proper, "
            f"{len(eval_data)} validation essays"
        )
    else:
        eval_data = get_split(dataset, "test")
        example_source = train_data
        logger.info(f"Test split: {len(eval_data)} test essays")

    # Select few-shot examples
    if example_seed is not None:
        fs_ids = select_fewshot_examples_random(
            example_source, n=n_examples, seed=example_seed
        )
    else:
        fs_ids = select_fewshot_examples(example_source, n=n_examples)

    examples = [
        {"text": example_source[eid]["text"], "baf": example_source[eid]["baf"]}
        for eid in fs_ids
    ]
    logger.info(f"Few-shot examples ({len(fs_ids)}): {fs_ids}")

    # Save metadata
    if split == "val":
        meta_dir = os.path.join(base_dir, "results_val", "meta")
    else:
        meta_dir = os.path.join(base_dir, "results", "meta")
    os.makedirs(meta_dir, exist_ok=True)

    tag_label = tag or "default"
    meta_filename = f"fewshot_examples_{tag_label}.json" if tag else "fewshot_examples.json"
    with open(os.path.join(meta_dir, meta_filename), "w") as f:
        json.dump({
            "essay_ids": fs_ids,
            "n_examples": n_examples,
            "example_seed": example_seed,
            "prompt_variant": prompt_variant,
            "split": split,
            "tag": tag,
        }, f, indent=2)

    # Log experiment config
    logger.info(f"Config: split={split}, variant={prompt_variant}, "
                f"n_examples={n_examples}, seed={example_seed}, tag={tag}")

    client = LLMClient()

    for method in methods:
        eff_method = _effective_method(method, tag)
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_key} | Method: {method} | Tag: {tag or '(none)'}")
        logger.info(f"Split: {split} | Variant: {prompt_variant} | Effective: {eff_method}")
        logger.info(f"{'='*60}")

        predictions: Dict[str, BAF] = {}
        golds: Dict[str, BAF] = {}
        n_success = 0
        total_latency = 0.0
        consecutive_empty = 0  # circuit breaker: consecutive empty API responses
        CIRCUIT_BREAKER_LIMIT = 3  # abort method after this many consecutive empties

        eval_ids = sorted(eval_data.keys())
        aborted = False
        for i, essay_id in enumerate(eval_ids):
            essay = eval_data[essay_id]

            # Resume: skip if already done
            if resume:
                existing = _load_raw(base_dir, model_key, eff_method, essay_id, split)
                if existing is not None:
                    pred_baf = BAF.from_dict(existing["pred_baf"])
                    predictions[essay_id] = pred_baf
                    golds[essay_id] = essay["baf"]
                    n_success += existing.get("parse_success", False)
                    logger.info(f"  [{i+1}/{len(eval_ids)}] {essay_id}: loaded from cache")
                    consecutive_empty = 0  # cached result breaks the streak
                    continue

            logger.info(f"  [{i+1}/{len(eval_ids)}] {essay_id}: running...")

            result = run_essay(
                client=client,
                model_key=model_key,
                method=method,
                essay_id=essay_id,
                essay_text=essay["text"],
                gold_baf=essay["baf"],
                examples=examples,
                variant=prompt_variant,
            )

            # Save raw result
            _save_raw(base_dir, model_key, eff_method, essay_id, result, split)

            pred_baf = BAF.from_dict(result["pred_baf"])
            predictions[essay_id] = pred_baf
            golds[essay_id] = essay["baf"]
            total_latency += result.get("latency_s", 0)

            if result["parse_success"]:
                n_success += 1
                consecutive_empty = 0
            else:
                # Check if the API returned empty content (systematic failure)
                # vs. a non-empty response that just didn't parse (model issue)
                raw_content = result.get("raw_output", "") or result.get("step1_raw", "")
                if not raw_content.strip():
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0  # got content, just couldn't parse it

            logger.info(
                f"    -> {len(pred_baf.arguments)} args, "
                f"{len(pred_baf.relations)} rels, "
                f"parse_ok={result['parse_success']}"
            )

            # Circuit breaker: abort if API is systematically failing
            if consecutive_empty >= CIRCUIT_BREAKER_LIMIT:
                logger.error(
                    f"ABORTING {method} on {model_key}: "
                    f"{CIRCUIT_BREAKER_LIMIT} consecutive empty API responses. "
                    f"This likely indicates a systematic issue (API key, model "
                    f"availability, or incompatible parameters). "
                    f"Completed {i+1}/{len(eval_ids)} essays before abort."
                )
                aborted = True
                break

        # Evaluate
        if aborted:
            logger.warning(f"Run aborted — evaluating {len(predictions)}/{len(eval_ids)} essays")
        logger.info(f"\nEvaluating {eff_method} on {model_key} ({split} split)...")
        logger.info(f"Parse success rate: {n_success}/{len(eval_ids)}")

        eval_results = evaluate_dataset(predictions, golds)
        eval_results["parse_success_rate"] = n_success / len(eval_ids) if eval_ids else 0
        eval_results["aborted"] = aborted

        # Save evaluation results
        if split == "val":
            eval_dir = os.path.join(base_dir, "results_val", "metrics")
        else:
            eval_dir = os.path.join(base_dir, "results", "metrics")
        os.makedirs(eval_dir, exist_ok=True)
        eval_path = os.path.join(eval_dir, f"{model_key}_{eff_method}.json")

        # Convert for JSON serialization
        save_results = {k: v for k, v in eval_results.items() if k != "per_essay"}
        save_results["per_essay_summary"] = {
            eid: {
                "arg_f1": m["argument"]["f1"],
                "sup_f1": m["relation"]["support"]["f1"],
                "att_f1": m["relation"]["attack"]["f1"],
                "rel_macro_f1": m["relation"]["macro"]["f1"],
            }
            for eid, m in eval_results["per_essay"].items()
        }
        # Record experiment config in metrics for traceability
        save_results["config"] = {
            "split": split,
            "prompt_variant": prompt_variant,
            "n_examples": n_examples,
            "example_seed": example_seed,
            "fewshot_ids": fs_ids,
            "tag": tag,
        }
        with open(eval_path, "w") as f:
            json.dump(save_results, f, indent=2)

        # Print results
        print(f"\n{prompts.METHODS.get(method, method)} | {MODELS[model_key].name}")
        if tag:
            print(f"[tag={tag}, split={split}, variant={prompt_variant}]")
        print(format_results(eval_results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run BAF extraction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="./PersuasiveEssaysV2",
        help="Path to the PersuasiveEssaysV2 directory",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for saving results (default: current dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to run (use --list to see available models)",
    )
    parser.add_argument(
        "--method",
        choices=ALL_METHODS + ["all", "e2e", "pipe", "gold"],
        default="all",
        help="Method(s) to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip essays that already have results",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and methods",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify API key and model availability before running",
    )

    # --- Sensitivity experiment parameters ---
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Evaluation split: 'test' (80 official test essays) or "
             "'val' (held-out from training). Default: test",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=prompts.PROMPT_VARIANTS,
        default="default",
        help="Prompt variant to use. Default: 'default'",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of few-shot examples. Default: 3",
    )
    parser.add_argument(
        "--example-seed",
        type=int,
        default=None,
        help="Random seed for few-shot example selection. "
             "If not set, uses the deterministic heuristic.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=40,
        help="Number of essays in validation split. Default: 40",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag appended to method name in result paths. "
             "Auto-generated from non-default settings if not provided.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run identifier for repeated-run variance experiments.",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for k, v in MODELS.items():
            print(f"  {k:25s} {v.name:25s} ({v.model_id})")
        print("\nAvailable methods:")
        for k, v in prompts.METHODS.items():
            print(f"  {k:15s} {v}")
        print("\nPrompt variants:")
        for v in prompts.PROMPT_VARIANTS:
            print(f"  {v}")
        return

    if args.check:
        print("Pre-flight check...")
        from src.llm_client import suggest_model
        client = LLMClient()
        models_to_check = [args.model] if args.model else list(MODELS.keys())
        for mk in models_to_check:
            if mk not in MODELS:
                suggestions = suggest_model(mk)
                hint = f"  Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                print(f"  {mk:25s} UNKNOWN MODEL{hint}")
                continue
            cfg = MODELS[mk]
            ok = client.check_model(mk)
            status = "OK" if ok else "FAILED"
            print(f"  {mk:25s} {cfg.model_id:45s} {status}")
        return

    if not args.model:
        parser.error("--model is required (use --list to see options)")

    # Validate model key with helpful suggestions on typo
    if args.model not in MODELS:
        from src.llm_client import suggest_model
        suggestions = suggest_model(args.model)
        msg = f"Unknown model '{args.model}'."
        if suggestions:
            msg += f"\n  Did you mean one of: {', '.join(suggestions)}?"
        msg += f"\n  Use --list to see all available models."
        parser.error(msg)

    # Resolve method groups
    if args.method == "all":
        methods = ALL_METHODS
    elif args.method == "e2e":
        methods = E2E_METHODS
    elif args.method == "pipe":
        methods = PIPE_METHODS
    elif args.method == "gold":
        methods = GOLD_METHODS
    else:
        methods = [args.method]

    # Compute effective tag
    tag = _compute_tag(args)

    # Resolve dataset path relative to base_dir if needed
    dataset_dir = args.dataset
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(args.base_dir, dataset_dir)

    run_experiment(
        dataset_dir=dataset_dir,
        base_dir=args.base_dir,
        model_key=args.model,
        methods=methods,
        resume=args.resume,
        split=args.split,
        prompt_variant=args.prompt_variant,
        n_examples=args.n_examples,
        example_seed=args.example_seed,
        tag=tag,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
