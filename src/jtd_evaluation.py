"""Evaluation metrics for JTD argumentation framework experiments.

Metrics:
  RE: Claim-level Micro-F1, Case-averaged F1, Per-party Micro-F1, Accuracy
  TP: Accuracy, Macro-F1, Precision, Recall
  Joint: Joint accuracy, RE->TP consistency
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .jtd_data_loader import TortCase


# ---------------------------------------------------------------------------
# Low-level P/R/F1
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


# ---------------------------------------------------------------------------
# RE: Rationale Extraction metrics
# ---------------------------------------------------------------------------


def re_micro_f1(
    all_pred: List[bool], all_gold: List[bool]
) -> Dict[str, float]:
    """Claim-level Micro-F1 for the 'accepted' class.

    Pools all claims across all cases.
    """
    tp = sum(1 for p, g in zip(all_pred, all_gold) if p and g)
    fp = sum(1 for p, g in zip(all_pred, all_gold) if p and not g)
    fn = sum(1 for p, g in zip(all_pred, all_gold) if not p and g)
    return _prf(tp, fp, fn)


def re_accuracy(all_pred: List[bool], all_gold: List[bool]) -> float:
    """Claim-level accuracy: fraction of claims correctly predicted."""
    if not all_pred:
        return 0.0
    correct = sum(1 for p, g in zip(all_pred, all_gold) if p == g)
    return correct / len(all_pred)


def re_case_averaged_f1(
    cases: List[TortCase],
    case_predictions: List[Dict[str, bool]],
) -> float:
    """Per-case F1 for accepted class, averaged across cases."""
    f1s = []
    for case, preds in zip(cases, case_predictions):
        gold = _case_gold_re(case)
        pred_list = [preds.get(k, False) for k in sorted(gold.keys())]
        gold_list = [gold[k] for k in sorted(gold.keys())]

        tp = sum(1 for p, g in zip(pred_list, gold_list) if p and g)
        fp = sum(1 for p, g in zip(pred_list, gold_list) if p and not g)
        fn = sum(1 for p, g in zip(pred_list, gold_list) if not p and g)

        prf = _prf(tp, fp, fn)
        f1s.append(prf["f1"])

    return sum(f1s) / len(f1s) if f1s else 0.0


def re_per_party_micro_f1(
    cases: List[TortCase],
    case_predictions: List[Dict[str, bool]],
) -> Dict[str, Dict[str, float]]:
    """Separate Micro-F1 for plaintiff and defendant claims."""
    p_pred, p_gold = [], []
    d_pred, d_gold = [], []

    for case, preds in zip(cases, case_predictions):
        for i, claim in enumerate(case.plaintiff_claims):
            key = f"p{i}"
            p_pred.append(preds.get(key, False))
            p_gold.append(claim.is_accepted)
        for i, claim in enumerate(case.defendant_claims):
            key = f"d{i}"
            d_pred.append(preds.get(key, False))
            d_gold.append(claim.is_accepted)

    return {
        "plaintiff": re_micro_f1(p_pred, p_gold),
        "defendant": re_micro_f1(d_pred, d_gold),
    }


# ---------------------------------------------------------------------------
# TP: Tort Prediction metrics
# ---------------------------------------------------------------------------


def tp_accuracy(pred: List[bool], gold: List[bool]) -> float:
    """TP accuracy: fraction of cases correctly predicted."""
    if not pred:
        return 0.0
    return sum(1 for p, g in zip(pred, gold) if p == g) / len(pred)


def tp_macro_f1(pred: List[bool], gold: List[bool]) -> Dict[str, float]:
    """TP Macro-F1: average of F1 for affirmed and denied classes."""
    # Affirmed class (True)
    tp_aff = sum(1 for p, g in zip(pred, gold) if p and g)
    fp_aff = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn_aff = sum(1 for p, g in zip(pred, gold) if not p and g)
    f1_aff = _prf(tp_aff, fp_aff, fn_aff)

    # Denied class (False)
    tp_den = sum(1 for p, g in zip(pred, gold) if not p and not g)
    fp_den = sum(1 for p, g in zip(pred, gold) if not p and g)
    fn_den = sum(1 for p, g in zip(pred, gold) if p and not g)
    f1_den = _prf(tp_den, fp_den, fn_den)

    macro_f1 = (f1_aff["f1"] + f1_den["f1"]) / 2

    return {
        "macro_f1": macro_f1,
        "affirmed_precision": f1_aff["precision"],
        "affirmed_recall": f1_aff["recall"],
        "affirmed_f1": f1_aff["f1"],
        "denied_f1": f1_den["f1"],
    }


# ---------------------------------------------------------------------------
# Joint metrics
# ---------------------------------------------------------------------------


def joint_accuracy(
    cases: List[TortCase],
    case_re_predictions: List[Dict[str, bool]],
    case_tp_predictions: List[bool],
) -> float:
    """Fraction of cases where BOTH TP and ALL RE predictions are correct."""
    correct = 0
    for case, re_preds, tp_pred in zip(
        cases, case_re_predictions, case_tp_predictions
    ):
        # Check TP
        if tp_pred != case.court_decision:
            continue
        # Check all RE
        gold = _case_gold_re(case)
        all_re_correct = all(
            re_preds.get(k, False) == gold[k] for k in gold
        )
        if all_re_correct:
            correct += 1

    return correct / len(cases) if cases else 0.0


def re_tp_consistency(
    case_re_predictions: List[Dict[str, bool]],
    case_tp_from_re: List[bool],
    case_tp_direct: List[bool],
) -> float:
    """Fraction of cases where TP derived from RE matches direct TP."""
    if not case_tp_from_re:
        return 0.0
    matches = sum(
        1 for a, b in zip(case_tp_from_re, case_tp_direct) if a == b
    )
    return matches / len(case_tp_from_re)


# ---------------------------------------------------------------------------
# Threshold tuning for quantitative methods
# ---------------------------------------------------------------------------

TAU_GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def tune_threshold(
    cases: List[TortCase],
    case_strengths: List[Dict[str, float]],
) -> Tuple[float, Dict[str, float]]:
    """Grid search for optimal RE threshold on training data.

    Returns (best_tau, {tau: micro_f1}).
    """
    results = {}

    for tau in TAU_GRID:
        all_pred = []
        all_gold = []

        for case, strengths in zip(cases, case_strengths):
            gold = _case_gold_re(case)
            for key in sorted(gold.keys()):
                all_gold.append(gold[key])
                all_pred.append(strengths.get(key, 0.0) > tau)

        metrics = re_micro_f1(all_pred, all_gold)
        results[tau] = metrics["f1"]

    best_tau = max(results, key=results.get)
    return best_tau, results


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def evaluate_fold(
    cases: List[TortCase],
    case_predictions: List[Dict],
    tp_strategy: str = "tp_b",
) -> Dict:
    """Compute all metrics for one fold.

    case_predictions: list of prediction dicts (from compute_*_predictions),
                      one per case, with keys 're_predictions', 'tp_a', 'tp_b', etc.
    tp_strategy: which TP strategy to use ('tp_a' or 'tp_b').
    """
    # Collect RE predictions
    all_re_pred = []
    all_re_gold = []
    case_re_preds = []

    for case, preds in zip(cases, case_predictions):
        re_pred = preds.get("re_predictions", {})
        case_re_preds.append(re_pred)
        gold = _case_gold_re(case)
        for key in sorted(gold.keys()):
            all_re_gold.append(gold[key])
            all_re_pred.append(re_pred.get(key, False))

    # RE metrics
    re_micro = re_micro_f1(all_re_pred, all_re_gold)
    re_acc = re_accuracy(all_re_pred, all_re_gold)
    re_case_f1 = re_case_averaged_f1(cases, case_re_preds)
    re_party = re_per_party_micro_f1(cases, case_re_preds)

    # Collect TP predictions
    tp_pred = []
    tp_gold = []
    for case, preds in zip(cases, case_predictions):
        tp_gold.append(case.court_decision)
        tp_val = preds.get(tp_strategy)
        if tp_val is None:
            tp_val = preds.get("tp_b", False)
        tp_pred.append(bool(tp_val))

    # TP metrics
    tp_acc = tp_accuracy(tp_pred, tp_gold)
    tp_f1 = tp_macro_f1(tp_pred, tp_gold)

    # Joint metrics
    j_acc = joint_accuracy(cases, case_re_preds, tp_pred)

    # TP from RE (Strategy B) for consistency check
    tp_from_re = []
    for preds in case_predictions:
        tp_b = preds.get("tp_b", False)
        tp_from_re.append(bool(tp_b))

    tp_direct = tp_pred
    consistency = re_tp_consistency(case_re_preds, tp_from_re, tp_direct)

    return {
        "re": {
            "micro_f1": re_micro,
            "accuracy": re_acc,
            "case_averaged_f1": re_case_f1,
            "per_party": re_party,
        },
        "tp": {
            "accuracy": tp_acc,
            **tp_f1,
            "strategy": tp_strategy,
        },
        "joint": {
            "joint_accuracy": j_acc,
            "re_tp_consistency": consistency,
        },
        "n_cases": len(cases),
    }


def aggregate_folds(fold_results: List[Dict]) -> Dict:
    """Aggregate metrics across CV folds (mean +/- std)."""
    import numpy as np

    def _collect(path):
        """Extract a scalar metric from each fold result using dot-path."""
        parts = path.split(".")
        values = []
        for fr in fold_results:
            v = fr
            for p in parts:
                v = v[p]
            values.append(float(v))
        return values

    metrics_paths = [
        ("re.micro_f1.f1", "RE Micro-F1"),
        ("re.micro_f1.precision", "RE Micro-Precision"),
        ("re.micro_f1.recall", "RE Micro-Recall"),
        ("re.accuracy", "RE Accuracy"),
        ("re.case_averaged_f1", "RE Case-averaged F1"),
        ("re.per_party.plaintiff.f1", "RE Plaintiff F1"),
        ("re.per_party.defendant.f1", "RE Defendant F1"),
        ("tp.accuracy", "TP Accuracy"),
        ("tp.macro_f1", "TP Macro-F1"),
        ("tp.affirmed_precision", "TP Affirmed Precision"),
        ("tp.affirmed_recall", "TP Affirmed Recall"),
        ("tp.affirmed_f1", "TP Affirmed F1"),
        ("joint.joint_accuracy", "Joint Accuracy"),
        ("joint.re_tp_consistency", "RE->TP Consistency"),
    ]

    summary = {}
    for path, label in metrics_paths:
        try:
            values = _collect(path)
            summary[label] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_fold": values,
            }
        except (KeyError, TypeError):
            pass

    return summary


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _case_gold_re(case: TortCase) -> Dict[str, bool]:
    """Get gold RE labels for a case: {claim_id: is_accepted}."""
    gold = {}
    for i, c in enumerate(case.plaintiff_claims):
        gold[f"p{i}"] = c.is_accepted
    for i, c in enumerate(case.defendant_claims):
        gold[f"d{i}"] = c.is_accepted
    return gold


def format_results(results: Dict) -> str:
    """Format aggregated fold results as a readable table."""
    lines = ["=" * 60, "EVALUATION RESULTS", "=" * 60]

    for label, data in results.items():
        if isinstance(data, dict) and "mean" in data:
            lines.append(f"  {label:30s}: {data['mean']:.4f} +/- {data['std']:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)
