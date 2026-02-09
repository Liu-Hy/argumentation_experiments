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
    get_split,
    load_persuasive_essays,
    select_fewshot_examples,
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


def _results_dir(base: str, model_key: str, method: str) -> str:
    d = os.path.join(base, "results", "raw", model_key, method)
    os.makedirs(d, exist_ok=True)
    return d


def _save_raw(base: str, model_key: str, method: str, essay_id: str, data: dict):
    d = _results_dir(base, model_key, method)
    with open(os.path.join(d, f"{essay_id}.json"), "w") as f:
        json.dump(data, f, indent=2)


def _load_raw(base: str, model_key: str, method: str, essay_id: str) -> Optional[dict]:
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
) -> List[Dict[str, str]]:
    """Build the prompt messages for a given method."""
    if method == "zs_e2e":
        return prompts.zs_e2e(essay_text)
    elif method == "fs_e2e":
        return prompts.fs_e2e(essay_text, examples)
    elif method == "fs_cot_e2e":
        return prompts.fs_cot_e2e(essay_text, examples)
    elif method == "zs_pipe":
        if step1_output is None:
            return prompts.zs_pipe_step1(essay_text)
        else:
            return prompts.zs_pipe_step2(essay_text, step1_output)
    elif method == "fs_pipe":
        if step1_output is None:
            return prompts.fs_pipe_step1(essay_text, examples)
        else:
            return prompts.fs_pipe_step2(essay_text, step1_output, examples)
    elif method == "gold_zs":
        return prompts.gold_arg_relations(essay_text, gold_baf, examples=None)
    elif method == "gold_fs":
        return prompts.gold_arg_relations(essay_text, gold_baf, examples=examples)
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
) -> Dict:
    """Run one method on one essay.  Returns a result dict."""
    is_pipe = method in PIPE_METHODS
    is_gold = method in GOLD_METHODS

    if is_pipe:
        # Step 1: identify arguments
        msgs1 = build_messages(method, essay_text, examples)
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
        msgs2 = build_messages(method, essay_text, examples, step1_output=step1_output)
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
        msgs = build_messages(method, essay_text, examples, gold_baf=gold_baf)
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
        msgs = build_messages(method, essay_text, examples)
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
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    dataset_dir: str,
    base_dir: str,
    model_key: str,
    methods: List[str],
    resume: bool = False,
):
    """Run the full experiment for one model across specified methods."""
    logger.info(f"Loading dataset from {dataset_dir}")
    dataset = load_persuasive_essays(dataset_dir)
    train_data = get_split(dataset, "train")
    test_data = get_split(dataset, "test")
    logger.info(f"Loaded {len(train_data)} train, {len(test_data)} test essays")

    # Select few-shot examples
    fs_ids = select_fewshot_examples(train_data, n=3)
    examples = [{"text": train_data[eid]["text"], "baf": train_data[eid]["baf"]} for eid in fs_ids]
    logger.info(f"Few-shot examples: {fs_ids}")

    # Save few-shot example IDs for reproducibility
    meta_dir = os.path.join(base_dir, "results", "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "fewshot_examples.json"), "w") as f:
        json.dump({"essay_ids": fs_ids}, f, indent=2)

    client = LLMClient()

    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_key} | Method: {method}")
        logger.info(f"{'='*60}")

        predictions: Dict[str, BAF] = {}
        golds: Dict[str, BAF] = {}
        n_success = 0
        total_latency = 0.0

        test_ids = sorted(test_data.keys())
        for i, essay_id in enumerate(test_ids):
            essay = test_data[essay_id]

            # Resume: skip if already done
            if resume:
                existing = _load_raw(base_dir, model_key, method, essay_id)
                if existing is not None:
                    pred_baf = BAF.from_dict(existing["pred_baf"])
                    predictions[essay_id] = pred_baf
                    golds[essay_id] = essay["baf"]
                    n_success += existing.get("parse_success", False)
                    logger.info(f"  [{i+1}/{len(test_ids)}] {essay_id}: loaded from cache")
                    continue

            logger.info(f"  [{i+1}/{len(test_ids)}] {essay_id}: running...")

            result = run_essay(
                client=client,
                model_key=model_key,
                method=method,
                essay_id=essay_id,
                essay_text=essay["text"],
                gold_baf=essay["baf"],
                examples=examples,
            )

            # Save raw result
            _save_raw(base_dir, model_key, method, essay_id, result)

            pred_baf = BAF.from_dict(result["pred_baf"])
            predictions[essay_id] = pred_baf
            golds[essay_id] = essay["baf"]
            total_latency += result.get("latency_s", 0)

            if result["parse_success"]:
                n_success += 1

            logger.info(
                f"    -> {len(pred_baf.arguments)} args, "
                f"{len(pred_baf.relations)} rels, "
                f"parse_ok={result['parse_success']}"
            )

        # Evaluate
        logger.info(f"\nEvaluating {method} on {model_key}...")
        logger.info(f"Parse success rate: {n_success}/{len(test_ids)}")

        eval_results = evaluate_dataset(predictions, golds)
        eval_results["parse_success_rate"] = n_success / len(test_ids) if test_ids else 0

        # Save evaluation results
        eval_dir = os.path.join(base_dir, "results", "metrics")
        os.makedirs(eval_dir, exist_ok=True)
        eval_path = os.path.join(eval_dir, f"{model_key}_{method}.json")
        # Convert for JSON serialization (remove per_essay BAFs)
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
        with open(eval_path, "w") as f:
            json.dump(save_results, f, indent=2)

        # Print results
        print(f"\n{prompts.METHODS.get(method, method)} | {MODELS[model_key].name}")
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
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for k, v in MODELS.items():
            print(f"  {k:25s} {v.name:25s} ({v.model_id})")
        print("\nAvailable methods:")
        for k, v in prompts.METHODS.items():
            print(f"  {k:15s} {v}")
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
    )


if __name__ == "__main__":
    main()
