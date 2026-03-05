#!/usr/bin/env python3
"""Experiment runner for JTD argumentation framework baselines.

Implements baselines B0-B6 and BH from the experiment plan.
All LLM calls go through OpenRouter.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."

    # Run a single baseline
    python run_jtd_experiment.py --model claude-haiku-4.5 --method B2

    # Run all baselines
    python run_jtd_experiment.py --model claude-haiku-4.5 --method all

    # Development run (200-case subsample)
    python run_jtd_experiment.py --model claude-haiku-4.5 --method B2 --subsample 200

    # List available models
    python run_jtd_experiment.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.af_solver import BipolarAF, DungAF, QBAF, baf_to_dung, df_quad
from src.jtd_data_loader import (
    TortCase,
    format_case_for_prompt,
    get_argument_ids,
    load_jtd,
    stratified_kfold,
    stratified_subsample,
)
from src.jtd_evaluation import (
    TAU_GRID,
    aggregate_folds,
    evaluate_fold,
    format_results,
    tune_threshold,
)
from src.jtd_output_parser import (
    AFParseResult,
    DirectPredictionResult,
    parse_baf,
    parse_direct_prediction,
    parse_dung_af,
    parse_qbaf,
    parse_quaf,
)
from src.jtd_prediction import (
    compute_baf_predictions,
    compute_quantitative_predictions,
    compute_standard_predictions,
    construct_heuristic_qbaf,
    re_from_strengths,
    re_strengths_for_claims,
)
from src.jtd_prompts import JTD_METHODS, build_af_prompt
from src.llm_client import MODELS, LLMClient, LLMResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("jtd_experiment")

ALL_METHODS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "BH"]
AF_METHODS = ["B2", "B3", "B4", "B5", "B6"]  # methods that construct AFs via LLM
QUANTITATIVE_METHODS = ["B5", "B6", "BH"]
DEFAULT_DATASET = "./data/japanese-tort-case/Japanese_tort_training.jsonl"


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def _results_dir(base: str, model_key: str, method: str) -> str:
    d = os.path.join(base, "results_jtd", "raw", model_key, method)
    os.makedirs(d, exist_ok=True)
    return d


def _save_raw(base: str, model_key: str, method: str, tort_id: str, data: dict):
    d = _results_dir(base, model_key, method)
    with open(os.path.join(d, f"{tort_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_raw(
    base: str, model_key: str, method: str, tort_id: str
) -> Optional[dict]:
    path = os.path.join(
        base, "results_jtd", "raw", model_key, method, f"{tort_id}.json"
    )
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------


def _get_valid_ids(case: TortCase, method: str) -> Tuple[Set[str], Set[str]]:
    """Get valid argument IDs and undisputed fact IDs for a method."""
    include_u = method in ("B4", "B6")
    all_ids = set(get_argument_ids(case, include_undisputed=include_u))
    u_ids = {f"u{i}" for i in range(len(case.undisputed_facts))}
    return all_ids, u_ids


def run_case_b0(
    client: LLMClient, model_key: str, case: TortCase
) -> dict:
    """B0: Direct LLM prediction."""
    msgs = build_af_prompt("B0", case)
    resp: LLMResponse = client.generate(msgs, model_key)

    parsed = parse_direct_prediction(
        resp.content, len(case.plaintiff_claims), len(case.defendant_claims)
    )

    return {
        "tort_id": case.tort_id,
        "method": "B0",
        "model": model_key,
        "raw_output": resp.content,
        "usage": resp.usage,
        "latency_s": resp.latency_s,
        "parse_success": parsed.success,
        "parse_errors": parsed.errors,
        "re_predictions": {
            **parsed.plaintiff_predictions,
            **parsed.defendant_predictions,
        },
        "tp_a": parsed.tort_decision,
        "tp_b": parsed.tort_decision,  # B0 has only one TP prediction
    }


def run_case_b1(
    client: LLMClient, model_key: str, case: TortCase
) -> dict:
    """B1: AF-guided LLM prediction (two-step)."""
    # Step 1: Construct AF
    msgs1 = build_af_prompt("B1", case)
    resp1: LLMResponse = client.generate(msgs1, model_key)
    af_json = resp1.content

    # Step 2: Predict with AF context
    msgs2 = build_af_prompt("B1", case, af_json=af_json)
    resp2: LLMResponse = client.generate(msgs2, model_key)

    parsed = parse_direct_prediction(
        resp2.content, len(case.plaintiff_claims), len(case.defendant_claims)
    )

    return {
        "tort_id": case.tort_id,
        "method": "B1",
        "model": model_key,
        "step1_raw": resp1.content,
        "step2_raw": resp2.content,
        "step1_usage": resp1.usage,
        "step2_usage": resp2.usage,
        "latency_s": resp1.latency_s + resp2.latency_s,
        "parse_success": parsed.success,
        "parse_errors": parsed.errors,
        "re_predictions": {
            **parsed.plaintiff_predictions,
            **parsed.defendant_predictions,
        },
        "tp_a": parsed.tort_decision,
        "tp_b": parsed.tort_decision,
    }


def run_case_af(
    client: LLMClient, model_key: str, method: str, case: TortCase
) -> dict:
    """Run AF construction methods (B2-B6): LLM call + parse."""
    msgs = build_af_prompt(method, case)
    resp: LLMResponse = client.generate(msgs, model_key)

    valid_ids, u_ids = _get_valid_ids(case, method)

    # Parse based on method
    if method in ("B2", "B3"):
        # Dung AF (attacks only), valid IDs exclude undisputed facts
        claim_ids = valid_ids - u_ids
        parsed = parse_dung_af(resp.content, claim_ids)
    elif method == "B4":
        parsed = parse_baf(resp.content, valid_ids, u_ids)
    elif method == "B5":
        claim_ids = valid_ids - u_ids
        parsed = parse_quaf(resp.content, claim_ids)
    elif method == "B6":
        parsed = parse_qbaf(resp.content, valid_ids, u_ids)
    else:
        raise ValueError(f"Unknown AF method: {method}")

    return {
        "tort_id": case.tort_id,
        "method": method,
        "model": model_key,
        "raw_output": resp.content,
        "usage": resp.usage,
        "latency_s": resp.latency_s,
        "parse_success": parsed.success,
        "parse_errors": parsed.errors,
        "af": {
            "attacks": [list(a) for a in parsed.attacks],
            "supports": [list(s) for s in parsed.supports],
            "argument_strengths": parsed.argument_strengths,
        },
    }


def run_case_bh(case: TortCase) -> dict:
    """BH: Heuristic QBAF (no LLM call)."""
    qbaf = construct_heuristic_qbaf(case)
    return {
        "tort_id": case.tort_id,
        "method": "BH",
        "model": "heuristic",
        "raw_output": "",
        "usage": {},
        "latency_s": 0.0,
        "parse_success": True,
        "parse_errors": [],
        "af": {
            "attacks": [list(a) for a in qbaf.attacks],
            "supports": [list(s) for s in qbaf.supports],
            "argument_strengths": qbaf.base_strengths,
        },
    }


# ---------------------------------------------------------------------------
# AF reconstruction from saved results
# ---------------------------------------------------------------------------


def reconstruct_af(result: dict, case: TortCase, method: str):
    """Reconstruct AF object from saved result dict.

    Returns the appropriate AF object (DungAF, BipolarAF, or QBAF).
    """
    af_data = result.get("af", {})
    attacks = {tuple(a) for a in af_data.get("attacks", [])}
    supports = {tuple(s) for s in af_data.get("supports", [])}
    strengths = af_data.get("argument_strengths", {})

    if method in ("B2", "B3"):
        # Dung AF: arguments are P and D claims only
        args = set(get_argument_ids(case, include_undisputed=False))
        return DungAF(arguments=args, attacks=attacks)

    elif method == "B4":
        # BAF: includes undisputed facts
        args = set(get_argument_ids(case, include_undisputed=True))
        return BipolarAF(arguments=args, attacks=attacks, supports=supports)

    elif method in ("B5",):
        # QuAF (attack-only): P and D claims with strengths
        args = set(get_argument_ids(case, include_undisputed=False))
        # Fill missing strengths
        for a in args:
            if a not in strengths:
                strengths[a] = 0.5
        return QBAF(arguments=args, base_strengths=strengths, attacks=attacks)

    elif method in ("B6", "BH"):
        # QBAF: includes undisputed facts
        args = set(get_argument_ids(case, include_undisputed=True))
        # Fill missing and enforce U strengths
        for a in args:
            if a.startswith("u"):
                strengths[a] = 1.0
            elif a not in strengths:
                strengths[a] = 0.5
        return QBAF(
            arguments=args,
            base_strengths=strengths,
            attacks=attacks,
            supports=supports,
        )

    else:
        raise ValueError(f"Cannot reconstruct AF for method: {method}")


# ---------------------------------------------------------------------------
# Compute predictions from AF
# ---------------------------------------------------------------------------


def compute_predictions(
    result: dict, case: TortCase, method: str, threshold: float = 0.5
) -> dict:
    """Compute RE/TP predictions from a saved result.

    Returns prediction dict with 're_predictions', 'tp_a', 'tp_b', etc.
    """
    if method in ("B0", "B1"):
        # Direct predictions already in result
        return {
            "re_predictions": result.get("re_predictions", {}),
            "tp_a": result.get("tp_a", False),
            "tp_b": result.get("tp_b", False),
        }

    if not result.get("parse_success", False):
        # Parse failed: return empty predictions (all rejected, tort denied)
        p_ids = [f"p{i}" for i in range(len(case.plaintiff_claims))]
        d_ids = [f"d{i}" for i in range(len(case.defendant_claims))]
        re_preds = {k: False for k in p_ids + d_ids}
        return {"re_predictions": re_preds, "tp_a": False, "tp_b": False}

    af = reconstruct_af(result, case, method)

    if method == "B2":
        return compute_standard_predictions(af, case, semantics="grounded")
    elif method == "B3":
        return compute_standard_predictions(af, case, semantics="preferred")
    elif method == "B4":
        return compute_baf_predictions(af, case)
    elif method in ("B5", "B6", "BH"):
        return compute_quantitative_predictions(af, case, threshold=threshold)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    dataset_path: str,
    base_dir: str,
    model_key: str,
    methods: List[str],
    n_folds: int = 5,
    subsample: Optional[int] = None,
    resume: bool = False,
    seed: int = 42,
):
    """Run the full JTD experiment."""
    logger.info(f"Loading dataset from {dataset_path}")
    all_cases = load_jtd(dataset_path)
    logger.info(f"Loaded {len(all_cases)} cases")

    # Subsample if requested
    if subsample is not None and subsample < len(all_cases):
        indices = stratified_subsample(all_cases, subsample, seed=seed)
        cases = [all_cases[i] for i in indices]
        logger.info(f"Subsampled to {len(cases)} cases (stratified)")
    else:
        cases = all_cases

    # Log class distribution
    n_pos = sum(1 for c in cases if c.court_decision)
    n_neg = len(cases) - n_pos
    logger.info(
        f"Class distribution: {n_pos} affirmed ({100*n_pos/len(cases):.1f}%), "
        f"{n_neg} denied ({100*n_neg/len(cases):.1f}%)"
    )

    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Method: {method} ({JTD_METHODS.get(method, '?')})")
        logger.info(f"Model: {model_key}")
        logger.info(f"{'='*60}")

        # Phase 1: AF Construction (LLM calls)
        if method not in ("BH",):
            _phase1_llm_calls(
                cases, base_dir, model_key, method, resume
            )
        else:
            _phase1_heuristic(cases, base_dir, method)

        # Phase 2: Compute predictions and evaluate per fold
        folds = stratified_kfold(cases, n_folds=n_folds, seed=seed)

        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"\n--- Fold {fold_idx+1}/{n_folds} ---")
            train_cases = [cases[i] for i in train_idx]
            test_cases = [cases[i] for i in test_idx]

            # Determine threshold for quantitative methods
            threshold = 0.5
            if method in QUANTITATIVE_METHODS:
                threshold = _tune_threshold_fold(
                    train_cases, base_dir, model_key, method
                )
                logger.info(f"  Tuned threshold: {threshold}")

            # Compute predictions for test cases
            test_predictions = []
            for case in test_cases:
                raw = _load_raw(base_dir, model_key, method, case.tort_id)
                if raw is None:
                    logger.warning(f"  Missing result for {case.tort_id}")
                    p_ids = [f"p{i}" for i in range(len(case.plaintiff_claims))]
                    d_ids = [f"d{i}" for i in range(len(case.defendant_claims))]
                    preds = {
                        "re_predictions": {k: False for k in p_ids + d_ids},
                        "tp_a": False,
                        "tp_b": False,
                    }
                else:
                    preds = compute_predictions(raw, case, method, threshold)
                test_predictions.append(preds)

            # Evaluate
            for tp_strategy in ["tp_a", "tp_b"]:
                fold_result = evaluate_fold(
                    test_cases, test_predictions, tp_strategy=tp_strategy
                )
                fold_result["fold"] = fold_idx
                fold_result["threshold"] = threshold
                fold_result["tp_strategy"] = tp_strategy

                if tp_strategy == "tp_b":  # primary strategy for fold aggregation
                    fold_results.append(fold_result)

                logger.info(
                    f"  Fold {fold_idx+1} ({tp_strategy}): "
                    f"RE-F1={fold_result['re']['micro_f1']['f1']:.4f}, "
                    f"TP-Acc={fold_result['tp']['accuracy']:.4f}, "
                    f"TP-F1={fold_result['tp']['macro_f1']:.4f}"
                )

        # Aggregate across folds
        summary = aggregate_folds(fold_results)

        # Save summary
        metrics_dir = os.path.join(base_dir, "results_jtd", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        summary_path = os.path.join(metrics_dir, f"{model_key}_{method}.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "method": method,
                    "model": model_key,
                    "n_cases": len(cases),
                    "n_folds": n_folds,
                    "subsample": subsample,
                    "summary": summary,
                    "per_fold": fold_results,
                },
                f,
                indent=2,
            )

        # Print results
        print(f"\n{JTD_METHODS.get(method, method)} | {MODELS[model_key].name}")
        print(format_results(summary))


def _phase1_llm_calls(
    cases: List[TortCase],
    base_dir: str,
    model_key: str,
    method: str,
    resume: bool,
):
    """Phase 1: Run LLM calls for AF construction / direct prediction."""
    client = LLMClient()
    consecutive_empty = 0
    CIRCUIT_LIMIT = 3

    for i, case in enumerate(cases):
        # Resume: skip if already done
        if resume:
            existing = _load_raw(base_dir, model_key, method, case.tort_id)
            if existing is not None:
                logger.info(
                    f"  [{i+1}/{len(cases)}] tort_{case.tort_id}: cached"
                )
                consecutive_empty = 0
                continue

        logger.info(f"  [{i+1}/{len(cases)}] tort_{case.tort_id}: running...")

        if method == "B0":
            result = run_case_b0(client, model_key, case)
        elif method == "B1":
            result = run_case_b1(client, model_key, case)
        elif method in AF_METHODS:
            result = run_case_af(client, model_key, method, case)
        else:
            raise ValueError(f"Unknown LLM method: {method}")

        _save_raw(base_dir, model_key, method, case.tort_id, result)

        if result.get("parse_success"):
            consecutive_empty = 0
        else:
            raw = result.get("raw_output", "") or result.get("step2_raw", "")
            if not raw.strip():
                consecutive_empty += 1
            else:
                consecutive_empty = 0

        if consecutive_empty >= CIRCUIT_LIMIT:
            logger.error(
                f"ABORTING {method}: {CIRCUIT_LIMIT} consecutive empty responses. "
                f"Completed {i+1}/{len(cases)} cases."
            )
            break

        # Log parse result
        n_attacks = len(result.get("af", {}).get("attacks", []))
        n_supports = len(result.get("af", {}).get("supports", []))
        ok = result.get("parse_success", False)
        if method in AF_METHODS:
            logger.info(
                f"    -> {n_attacks} attacks, {n_supports} supports, parse_ok={ok}"
            )
        else:
            logger.info(f"    -> parse_ok={ok}")


def _phase1_heuristic(cases: List[TortCase], base_dir: str, method: str):
    """Phase 1 for BH: construct heuristic QBAFs (no LLM)."""
    for i, case in enumerate(cases):
        existing = _load_raw(base_dir, "heuristic", method, case.tort_id)
        if existing is not None:
            continue

        result = run_case_bh(case)
        _save_raw(base_dir, "heuristic", method, case.tort_id, result)

    logger.info(f"  BH: {len(cases)} heuristic QBAFs constructed")


def _tune_threshold_fold(
    train_cases: List[TortCase],
    base_dir: str,
    model_key: str,
    method: str,
) -> float:
    """Tune threshold on training fold for quantitative methods."""
    effective_model = "heuristic" if method == "BH" else model_key
    case_strengths = []

    for case in train_cases:
        raw = _load_raw(base_dir, effective_model, method, case.tort_id)
        if raw is None or not raw.get("parse_success", False):
            # Missing or failed: use zeros
            p_ids = [f"p{i}" for i in range(len(case.plaintiff_claims))]
            d_ids = [f"d{i}" for i in range(len(case.defendant_claims))]
            case_strengths.append({k: 0.0 for k in p_ids + d_ids})
            continue

        af = reconstruct_af(raw, case, method)

        # Compute DF-QuAD to get strengths
        from src.af_solver import df_quad as _df_quad

        final = _df_quad(af)
        p_ids = [f"p{i}" for i in range(len(case.plaintiff_claims))]
        d_ids = [f"d{i}" for i in range(len(case.defendant_claims))]
        claim_str = re_strengths_for_claims(final, p_ids, d_ids)
        case_strengths.append(claim_str)

    best_tau, tau_results = tune_threshold(train_cases, case_strengths)
    return best_tau


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run JTD argumentation framework experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to JTD JSONL file",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model key (use --list to see available)",
    )
    parser.add_argument(
        "--method",
        choices=ALL_METHODS + ["all", "af_only", "quantitative"],
        default="all",
        help="Method(s) to run",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample size for development (default: all cases)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases with existing results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and methods",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for k, v in MODELS.items():
            print(f"  {k:25s} {v.name:25s} ({v.model_id})")
        print("\nJTD methods:")
        for k, v in JTD_METHODS.items():
            print(f"  {k:5s} {v}")
        return

    if not args.model:
        parser.error("--model is required (use --list to see options)")

    if args.model not in MODELS:
        from src.llm_client import suggest_model

        suggestions = suggest_model(args.model)
        msg = f"Unknown model '{args.model}'."
        if suggestions:
            msg += f"\n  Did you mean: {', '.join(suggestions)}?"
        parser.error(msg)

    # Resolve method groups
    if args.method == "all":
        methods = ALL_METHODS
    elif args.method == "af_only":
        methods = AF_METHODS
    elif args.method == "quantitative":
        methods = QUANTITATIVE_METHODS
    else:
        methods = [args.method]

    # Resolve dataset path
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(args.base_dir, dataset_path)

    run_experiment(
        dataset_path=dataset_path,
        base_dir=args.base_dir,
        model_key=args.model,
        methods=methods,
        n_folds=args.n_folds,
        subsample=args.subsample,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
