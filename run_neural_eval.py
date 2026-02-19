#!/usr/bin/env python3
"""Post-hoc neural evaluation of saved BAF extraction results.

Computes reference-free neural metrics (BERTScore AF-Coverage and NLI
AF-Faithfulness) on saved per-essay predictions from run_experiment.py.

Usage:
    # Evaluate one (model, method) pair
    python run_neural_eval.py --model gemini-3-flash-preview --method zs_e2e

    # Evaluate all saved results
    python run_neural_eval.py --all

    # Use CPU (default auto-detects GPU)
    python run_neural_eval.py --model gemini-3-flash-preview --method fs_e2e --device cpu

    # Custom paths
    python run_neural_eval.py --model gemini-3-flash-preview --method zs_e2e \
        --dataset ./PersuasiveEssaysV2 --base-dir .
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baf import BAF
from src.data_loader import get_split, load_persuasive_essays
from src.neural_metrics import evaluate_dataset_neural

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("neural_eval")


# ---------------------------------------------------------------------------
# Loading saved results
# ---------------------------------------------------------------------------


def _discover_results(base_dir: str) -> List[Tuple[str, str]]:
    """Discover all (model, method) pairs with saved raw results."""
    raw_dir = os.path.join(base_dir, "results", "raw")
    if not os.path.isdir(raw_dir):
        return []

    pairs = []
    for model_key in sorted(os.listdir(raw_dir)):
        model_dir = os.path.join(raw_dir, model_key)
        if not os.path.isdir(model_dir):
            continue
        for method in sorted(os.listdir(model_dir)):
            method_dir = os.path.join(model_dir, method)
            if not os.path.isdir(method_dir):
                continue
            # Check that there's at least one JSON file
            jsons = [f for f in os.listdir(method_dir) if f.endswith(".json")]
            if jsons:
                pairs.append((model_key, method))
    return pairs


def _load_predictions(
    base_dir: str, model_key: str, method: str
) -> Dict[str, BAF]:
    """Load predicted BAFs from saved per-essay JSON files."""
    raw_dir = os.path.join(base_dir, "results", "raw", model_key, method)
    predictions: Dict[str, BAF] = {}

    if not os.path.isdir(raw_dir):
        return predictions

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        essay_id = fname[:-5]  # strip .json
        with open(os.path.join(raw_dir, fname)) as f:
            data = json.load(f)
        pred_baf = BAF.from_dict(data["pred_baf"])
        predictions[essay_id] = pred_baf

    return predictions


# ---------------------------------------------------------------------------
# Auto-detect device
# ---------------------------------------------------------------------------


def _detect_device(requested: str) -> str:
    """Resolve device string.  'auto' -> 'cuda' if available, else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------


def format_neural_results(results: Dict) -> str:
    """Format neural evaluation results as a readable table."""
    lines = []
    lines.append("=" * 72)
    lines.append("NEURAL EVALUATION METRICS")
    lines.append("=" * 72)
    lines.append(
        f"  Essays: {results['n_essays']}  "
        f"(predictions: {results['n_predicted']})"
    )
    lines.append(f"  Device: {results['config']['device']}")
    lines.append("")

    macro = results["macro"]
    gold = results["gold_ceiling"]

    lines.append("--- BERTScore AF-Coverage ---")
    for k, label in [
        ("bertscore_precision", "Precision (faithfulness)"),
        ("bertscore_recall", "Recall (coverage)"),
        ("bertscore_f1", "F1"),
        ("bertscore_per_arg_precision", "Per-arg precision"),
    ]:
        lines.append(
            f"  {label:26s}: {macro[k]:.3f}  "
            f"(gold ceiling: {gold[k]:.3f})"
        )

    lines.append("")
    lines.append("--- NLI AF-Faithfulness ---")
    for k, label in [
        ("nli_faithfulness_mean", "Overall faithfulness"),
        ("nli_argument_faithfulness", "Argument faithfulness"),
        ("nli_relation_groundedness", "Relation groundedness"),
        ("nli_faithfulness_frac_above_50", "Frac above 0.5"),
    ]:
        lines.append(
            f"  {label:26s}: {macro[k]:.3f}  "
            f"(gold ceiling: {gold[k]:.3f})"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def run_neural_eval(
    dataset_dir: str,
    base_dir: str,
    model_key: str,
    method: str,
    device: str = "auto",
    batch_size: int = 16,
):
    """Run neural evaluation for one (model, method) pair."""
    device = _detect_device(device)
    logger.info(f"Neural evaluation: {model_key} / {method} on {device}")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}")
    dataset = load_persuasive_essays(dataset_dir)
    test_data = get_split(dataset, "test")
    logger.info(f"Loaded {len(test_data)} test essays")

    # Load predictions
    predictions = _load_predictions(base_dir, model_key, method)
    logger.info(f"Loaded {len(predictions)} predictions for {model_key}/{method}")

    if not predictions:
        logger.warning(f"No predictions found for {model_key}/{method}. Skipping.")
        return None

    # Prepare inputs
    golds = {eid: test_data[eid]["baf"] for eid in test_data}
    essay_texts = {eid: test_data[eid]["text"] for eid in test_data}

    # Run evaluation
    results = evaluate_dataset_neural(
        predictions=predictions,
        golds=golds,
        essay_texts=essay_texts,
        device=device,
        batch_size=batch_size,
    )

    # Save results
    out_dir = os.path.join(base_dir, "results", "neural_metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_key}_{method}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")

    # Print summary
    print(f"\n{model_key} | {method}")
    print(format_neural_results(results))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc neural evaluation of BAF extraction results",
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
        help="Base directory for results (default: current dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all saved (model, method) pairs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto' (default), 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for NLI inference (default: 16)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available (model, method) pairs with saved results",
    )
    args = parser.parse_args()

    # Resolve dataset path
    dataset_dir = args.dataset
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(args.base_dir, dataset_dir)

    if args.list:
        pairs = _discover_results(args.base_dir)
        if not pairs:
            print("No saved results found.")
            return
        print("Available (model, method) pairs:")
        for model_key, method in pairs:
            raw_dir = os.path.join(
                args.base_dir, "results", "raw", model_key, method
            )
            n_files = len([
                f for f in os.listdir(raw_dir) if f.endswith(".json")
            ])
            print(f"  {model_key:30s} {method:15s} ({n_files} essays)")
        return

    if args.all:
        pairs = _discover_results(args.base_dir)
        if not pairs:
            print("No saved results found.")
            return
        logger.info(f"Evaluating {len(pairs)} (model, method) pairs")
        for model_key, method in pairs:
            run_neural_eval(
                dataset_dir=dataset_dir,
                base_dir=args.base_dir,
                model_key=model_key,
                method=method,
                device=args.device,
                batch_size=args.batch_size,
            )
        return

    if not args.model or not args.method:
        parser.error("--model and --method are required (or use --all)")

    run_neural_eval(
        dataset_dir=dataset_dir,
        base_dir=args.base_dir,
        model_key=args.model,
        method=args.method,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
