"""Evaluation metrics for Bipolar Argumentation Framework extraction.

All metrics are dataset-agnostic: they take predicted and gold BAFs as input.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Tuple

from .baf import Argument, BAF, Relation

# We avoid a hard scipy dependency by providing a fallback for the
# Hungarian algorithm.  scipy is *much* faster on large instances, but
# for typical essay sizes (< 30 arguments) the naive implementation is fine.
try:
    from scipy.optimize import linear_sum_assignment as _lsa

    def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _lsa(cost)

except ImportError:

    def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy fallback (not optimal but acceptable for small n)."""
        n, m = cost.shape
        row_ind, col_ind = [], []
        used_cols = set()
        # Greedy: pick the lowest-cost cell repeatedly
        flat = [(cost[i, j], i, j) for i in range(n) for j in range(m)]
        flat.sort()
        used_rows = set()
        for val, i, j in flat:
            if i not in used_rows and j not in used_cols:
                row_ind.append(i)
                col_ind.append(j)
                used_rows.add(i)
                used_cols.add(j)
        return np.array(row_ind), np.array(col_ind)


# ---------------------------------------------------------------------------
# Argument span matching
# ---------------------------------------------------------------------------


def match_arguments(
    pred_args: List[Argument],
    gold_args: List[Argument],
    iou_threshold: float = 0.5,
) -> Dict[str, str]:
    """Optimally match predicted argument spans to gold spans.

    Uses the Hungarian algorithm on a cost matrix of -IoU values.

    Returns
    -------
    dict : pred_arg_id -> gold_arg_id  (only for pairs with IoU >= threshold)
    """
    if not pred_args or not gold_args:
        return {}

    n_pred = len(pred_args)
    n_gold = len(gold_args)
    cost = np.zeros((n_pred, n_gold))
    for i, p in enumerate(pred_args):
        for j, g in enumerate(gold_args):
            cost[i, j] = -p.iou(g)

    row_ind, col_ind = _hungarian(cost)

    matching: Dict[str, str] = {}
    for i, j in zip(row_ind, col_ind):
        if -cost[i, j] >= iou_threshold:
            matching[pred_args[i].id] = gold_args[j].id
    return matching


# ---------------------------------------------------------------------------
# P / R / F1 helpers
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Argument-level evaluation
# ---------------------------------------------------------------------------


def compute_argument_metrics(
    pred_baf: BAF, gold_baf: BAF, iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Precision / Recall / F1 for argument span identification."""
    matching = match_arguments(pred_baf.arguments, gold_baf.arguments, iou_threshold)
    tp = len(matching)
    fp = len(pred_baf.arguments) - tp
    fn = len(gold_baf.arguments) - tp
    return _prf(tp, fp, fn)


# ---------------------------------------------------------------------------
# Relation-level evaluation
# ---------------------------------------------------------------------------


def compute_relation_metrics(
    pred_baf: BAF,
    gold_baf: BAF,
    arg_matching: Optional[Dict[str, str]] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Per-type and aggregate relation P / R / F1.

    A predicted relation (src, tgt, type) is a true positive iff:
      - src matches a gold argument via arg_matching
      - tgt matches a gold argument via arg_matching
      - a gold relation exists between those gold arguments with the same type
    """
    if arg_matching is None:
        arg_matching = match_arguments(
            pred_baf.arguments, gold_baf.arguments, iou_threshold
        )

    gold_rel_set = {(r.source, r.target, r.type) for r in gold_baf.relations}

    # Count gold relations by type
    gold_counts: Dict[str, int] = {}
    for r in gold_baf.relations:
        gold_counts[r.type] = gold_counts.get(r.type, 0) + 1

    # Evaluate predicted relations
    tp_by_type: Dict[str, int] = {"support": 0, "attack": 0}
    fp_by_type: Dict[str, int] = {"support": 0, "attack": 0}

    for r in pred_baf.relations:
        rtype = r.type
        if rtype not in tp_by_type:
            # Unknown relation type treated as FP under both types?
            # Safest: count it as a generic FP
            fp_by_type.setdefault(rtype, 0)
            fp_by_type[rtype] += 1
            continue

        gold_src = arg_matching.get(r.source)
        gold_tgt = arg_matching.get(r.target)
        if gold_src and gold_tgt and (gold_src, gold_tgt, rtype) in gold_rel_set:
            tp_by_type[rtype] += 1
        else:
            fp_by_type[rtype] += 1

    results: Dict[str, Dict[str, float]] = {}
    for rtype in ["support", "attack"]:
        tp = tp_by_type.get(rtype, 0)
        fp = fp_by_type.get(rtype, 0)
        fn = gold_counts.get(rtype, 0) - tp
        results[rtype] = _prf(tp, fp, fn)

    # Macro F1 (average of per-type F1)
    type_f1s = [results[t]["f1"] for t in ["support", "attack"]]
    results["macro"] = {"f1": float(np.mean(type_f1s))}

    # Micro F1 (pool all TP/FP/FN across types)
    total_tp = sum(results[t]["tp"] for t in ["support", "attack"])
    total_fp = sum(results[t]["fp"] for t in ["support", "attack"])
    total_fn = sum(results[t]["fn"] for t in ["support", "attack"])
    results["micro"] = _prf(int(total_tp), int(total_fp), int(total_fn))

    return results


# ---------------------------------------------------------------------------
# Single-essay evaluation
# ---------------------------------------------------------------------------


def evaluate_essay(
    pred_baf: BAF, gold_baf: BAF, iou_threshold: float = 0.5
) -> Dict:
    """Full evaluation for one essay.  Returns nested dict of metrics."""
    arg_matching = match_arguments(
        pred_baf.arguments, gold_baf.arguments, iou_threshold
    )
    return {
        "argument": compute_argument_metrics(pred_baf, gold_baf, iou_threshold),
        "relation": compute_relation_metrics(
            pred_baf, gold_baf, arg_matching, iou_threshold
        ),
        "n_pred_args": len(pred_baf.arguments),
        "n_gold_args": len(gold_baf.arguments),
        "n_pred_rels": len(pred_baf.relations),
        "n_gold_rels": len(gold_baf.relations),
    }


# ---------------------------------------------------------------------------
# Dataset-level evaluation with bootstrap CIs
# ---------------------------------------------------------------------------


def evaluate_dataset(
    predictions: Dict[str, BAF],
    golds: Dict[str, BAF],
    iou_threshold: float = 0.5,
    bootstrap_n: int = 10000,
    bootstrap_seed: int = 42,
) -> Dict:
    """Aggregate evaluation across all essays.

    Returns per-essay results, micro-aggregated metrics, and bootstrap 95% CIs.
    """
    per_essay: Dict[str, Dict] = {}
    for essay_id, gold_baf in golds.items():
        if essay_id in predictions:
            per_essay[essay_id] = evaluate_essay(
                predictions[essay_id], gold_baf, iou_threshold
            )
        else:
            # Missing prediction => everything is a false negative
            n_gold_support = sum(
                1 for r in gold_baf.relations if r.type == "support"
            )
            n_gold_attack = sum(
                1 for r in gold_baf.relations if r.type == "attack"
            )
            per_essay[essay_id] = {
                "argument": _prf(0, 0, len(gold_baf.arguments)),
                "relation": {
                    "support": _prf(0, 0, n_gold_support),
                    "attack": _prf(0, 0, n_gold_attack),
                    "macro": {"f1": 0.0},
                    "micro": _prf(0, 0, len(gold_baf.relations)),
                },
                "n_pred_args": 0,
                "n_gold_args": len(gold_baf.arguments),
                "n_pred_rels": 0,
                "n_gold_rels": len(gold_baf.relations),
            }

    # ---- Micro-aggregate (pool TP/FP/FN across all essays) ----
    micro = _micro_aggregate(per_essay)

    # ---- Macro-average of per-essay F1 scores ----
    macro = _macro_average(per_essay)

    # ---- Bootstrap 95% CIs on macro-average F1 ----
    cis = _bootstrap_cis(per_essay, bootstrap_n, bootstrap_seed)

    return {
        "per_essay": per_essay,
        "micro": micro,
        "macro": macro,
        "bootstrap_95ci": cis,
        "n_essays": len(golds),
        "n_predicted": len(predictions),
    }


def _micro_aggregate(per_essay: Dict[str, Dict]) -> Dict:
    """Pool TP/FP/FN across all essays, then compute P/R/F1."""
    arg_tp = arg_fp = arg_fn = 0
    rel_tp: Dict[str, int] = {"support": 0, "attack": 0}
    rel_fp: Dict[str, int] = {"support": 0, "attack": 0}
    rel_fn: Dict[str, int] = {"support": 0, "attack": 0}

    for m in per_essay.values():
        arg_tp += m["argument"]["tp"]
        arg_fp += m["argument"]["fp"]
        arg_fn += m["argument"]["fn"]
        for rtype in ["support", "attack"]:
            rel_tp[rtype] += m["relation"][rtype]["tp"]
            rel_fp[rtype] += m["relation"][rtype]["fp"]
            rel_fn[rtype] += m["relation"][rtype]["fn"]

    result = {"argument": _prf(arg_tp, arg_fp, arg_fn), "relation": {}}
    for rtype in ["support", "attack"]:
        result["relation"][rtype] = _prf(
            rel_tp[rtype], rel_fp[rtype], rel_fn[rtype]
        )

    total_tp = sum(rel_tp.values())
    total_fp = sum(rel_fp.values())
    total_fn = sum(rel_fn.values())
    result["relation"]["micro"] = _prf(total_tp, total_fp, total_fn)

    f1s = [result["relation"][t]["f1"] for t in ["support", "attack"]]
    result["relation"]["macro"] = {"f1": float(np.mean(f1s))}

    return result


def _macro_average(per_essay: Dict[str, Dict]) -> Dict:
    """Average per-essay F1 scores."""
    if not per_essay:
        return {}
    arg_f1s = [m["argument"]["f1"] for m in per_essay.values()]
    sup_f1s = [m["relation"]["support"]["f1"] for m in per_essay.values()]
    att_f1s = [m["relation"]["attack"]["f1"] for m in per_essay.values()]
    rel_macro_f1s = [m["relation"]["macro"]["f1"] for m in per_essay.values()]

    return {
        "argument_f1": float(np.mean(arg_f1s)),
        "support_f1": float(np.mean(sup_f1s)),
        "attack_f1": float(np.mean(att_f1s)),
        "relation_macro_f1": float(np.mean(rel_macro_f1s)),
    }


def _bootstrap_cis(
    per_essay: Dict[str, Dict],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict:
    """Bootstrap 95% confidence intervals for macro-average F1 scores."""
    if not per_essay:
        return {}
    rng = np.random.RandomState(seed)
    essay_ids = list(per_essay.keys())
    n = len(essay_ids)

    keys = ["argument_f1", "support_f1", "attack_f1", "relation_macro_f1"]
    samples: Dict[str, List[float]] = {k: [] for k in keys}

    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        resampled = [per_essay[essay_ids[i]] for i in indices]
        samples["argument_f1"].append(
            float(np.mean([m["argument"]["f1"] for m in resampled]))
        )
        samples["support_f1"].append(
            float(np.mean([m["relation"]["support"]["f1"] for m in resampled]))
        )
        samples["attack_f1"].append(
            float(np.mean([m["relation"]["attack"]["f1"] for m in resampled]))
        )
        samples["relation_macro_f1"].append(
            float(np.mean([m["relation"]["macro"]["f1"] for m in resampled]))
        )

    cis = {}
    for k in keys:
        arr = np.array(samples[k])
        cis[k] = {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }
    return cis


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------


def format_results(results: Dict) -> str:
    """Format evaluation results as a readable table."""
    lines = []
    lines.append("=" * 72)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 72)

    micro = results["micro"]
    macro = results["macro"]
    cis = results["bootstrap_95ci"]

    lines.append(
        f"  Essays evaluated: {results['n_essays']}  "
        f"(predictions received: {results['n_predicted']})"
    )
    lines.append("")

    # Micro-aggregated (pooled TP/FP/FN)
    lines.append("--- Micro-aggregated (pooled across essays) ---")
    am = micro["argument"]
    lines.append(
        f"  Argument spans:  P={am['precision']:.3f}  R={am['recall']:.3f}  "
        f"F1={am['f1']:.3f}"
    )
    for rtype in ["support", "attack"]:
        rm = micro["relation"][rtype]
        lines.append(
            f"  {rtype.capitalize():8s} rels:  P={rm['precision']:.3f}  "
            f"R={rm['recall']:.3f}  F1={rm['f1']:.3f}"
        )
    rm = micro["relation"]["micro"]
    lines.append(
        f"  All rels (micro):P={rm['precision']:.3f}  R={rm['recall']:.3f}  "
        f"F1={rm['f1']:.3f}"
    )
    lines.append(
        f"  Rel macro F1:    {micro['relation']['macro']['f1']:.3f}"
    )

    lines.append("")
    lines.append("--- Macro-averaged (per-essay F1, then averaged) ---")
    for k, label in [
        ("argument_f1", "Argument span F1"),
        ("support_f1", "Support rel F1"),
        ("attack_f1", "Attack rel F1"),
        ("relation_macro_f1", "Relation macro F1"),
    ]:
        ci = cis.get(k, {})
        lines.append(
            f"  {label:22s}: {macro[k]:.3f}  "
            f"[{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}]"
        )

    lines.append("=" * 72)
    return "\n".join(lines)
