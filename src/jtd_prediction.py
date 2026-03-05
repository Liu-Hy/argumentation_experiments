"""Map argumentation framework results to RE/TP predictions for JTD.

Handles:
- Extension-based predictions (standard AF: B2, B3, B4)
- Strength-based predictions (quantitative AF: B5, B6, BH)
- TP Strategy A (meta-argument)
- TP Strategy B (aggregation over RE)
- Heuristic QBAF construction (BH)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .af_solver import (
    DungAF,
    BipolarAF,
    QBAF,
    baf_to_dung,
    df_quad,
    grounded_extension,
    preferred_extensions,
    credulous_acceptance,
    skeptical_acceptance,
)
from .jtd_data_loader import TortCase


# ---------------------------------------------------------------------------
# Rationale Extraction (RE) predictions
# ---------------------------------------------------------------------------


def re_from_extension(
    in_set: Set[str],
    p_ids: List[str],
    d_ids: List[str],
) -> Dict[str, bool]:
    """RE predictions from standard AF extension membership.

    A claim is accepted iff it is in the IN set.
    Undecided claims are predicted as rejected (default convention).

    Returns dict: claim_id -> accepted (bool).
    """
    predictions = {}
    for pid in p_ids:
        predictions[pid] = pid in in_set
    for did in d_ids:
        predictions[did] = did in in_set
    return predictions


def re_from_strengths(
    strengths: Dict[str, float],
    p_ids: List[str],
    d_ids: List[str],
    threshold: float = 0.5,
) -> Dict[str, bool]:
    """RE predictions from quantitative AF final strengths.

    A claim is accepted iff its final strength > threshold.
    """
    predictions = {}
    for pid in p_ids:
        predictions[pid] = strengths.get(pid, 0.0) > threshold
    for did in d_ids:
        predictions[did] = strengths.get(did, 0.0) > threshold
    return predictions


def re_strengths_for_claims(
    strengths: Dict[str, float],
    p_ids: List[str],
    d_ids: List[str],
) -> Dict[str, float]:
    """Extract final strengths for claims only (for threshold tuning)."""
    result = {}
    for pid in p_ids:
        result[pid] = strengths.get(pid, 0.0)
    for did in d_ids:
        result[did] = strengths.get(did, 0.0)
    return result


# ---------------------------------------------------------------------------
# Tort Prediction (TP) -- Strategy A: Meta-argument
# ---------------------------------------------------------------------------


def tp_meta_argument_grounded(
    af: DungAF,
    p_ids: List[str],
    d_ids: List[str],
) -> bool:
    """TP Strategy A for standard Dung AF (B2) using grounded semantics.

    Augments AF with meta-argument T:
    - All D-claims attack T
    - All P-claims attack all D-claims (defense of T)
    T in grounded extension => tort affirmed.
    """
    augmented = _build_meta_dung(af, p_ids, d_ids)
    in_set, _, _ = grounded_extension(augmented)
    return "T" in in_set


def tp_meta_argument_preferred(
    af: DungAF,
    p_ids: List[str],
    d_ids: List[str],
    timeout_s: float = 30.0,
) -> Tuple[Optional[bool], Optional[bool], bool]:
    """TP Strategy A for standard Dung AF (B3) using preferred semantics.

    Returns (credulous, skeptical, timed_out).
    If timed out, returns (None, None, True).
    """
    augmented = _build_meta_dung(af, p_ids, d_ids)
    exts = preferred_extensions(augmented, timeout_s=timeout_s)
    if exts is None:
        return None, None, True
    cred = credulous_acceptance(exts)
    skep = skeptical_acceptance(exts)
    return "T" in cred, "T" in skep, False


def tp_meta_argument_baf(
    baf: BipolarAF,
    p_ids: List[str],
    d_ids: List[str],
) -> bool:
    """TP Strategy A for BAF (B4) using grounded semantics via reduction.

    Augments BAF with T:
    - P-claims support T
    - D-claims attack T
    Reduces to Dung AF and computes grounded extension.
    """
    args = set(baf.arguments) | {"T"}
    attacks = set(baf.attacks)
    supports = set(baf.supports)

    for d in d_ids:
        attacks.add((d, "T"))
    for p in p_ids:
        supports.add((p, "T"))

    augmented_baf = BipolarAF(arguments=args, attacks=attacks, supports=supports)
    augmented_dung = baf_to_dung(augmented_baf)
    in_set, _, _ = grounded_extension(augmented_dung)
    return "T" in in_set


def tp_meta_argument_quantitative(
    qbaf: QBAF,
    p_ids: List[str],
    d_ids: List[str],
) -> Tuple[bool, float]:
    """TP Strategy A for quantitative AF (B5, B6, BH).

    Augments QBAF with T (base strength 0.5):
    - P-claims support T
    - D-claims attack T
    T's final strength > 0.5 => affirmed.

    Returns (affirmed, T_strength).
    """
    args = set(qbaf.arguments) | {"T"}
    strengths = dict(qbaf.base_strengths)
    strengths["T"] = 0.5
    attacks = set(qbaf.attacks)
    supports = set(qbaf.supports)

    for d in d_ids:
        attacks.add((d, "T"))
    for p in p_ids:
        supports.add((p, "T"))

    augmented = QBAF(
        arguments=args,
        base_strengths=strengths,
        attacks=attacks,
        supports=supports,
    )
    final = df_quad(augmented)
    t_strength = final.get("T", 0.5)
    return t_strength > 0.5, t_strength


def _build_meta_dung(
    af: DungAF, p_ids: List[str], d_ids: List[str]
) -> DungAF:
    """Build augmented Dung AF with meta-argument T for TP Strategy A.

    - All D attack T
    - All P attack all D (defense of T)
    """
    args = set(af.arguments) | {"T"}
    attacks = set(af.attacks)

    for d in d_ids:
        attacks.add((d, "T"))
    for p in p_ids:
        for d in d_ids:
            attacks.add((p, d))

    return DungAF(arguments=args, attacks=attacks)


# ---------------------------------------------------------------------------
# Tort Prediction (TP) -- Strategy B: Aggregation over RE
# ---------------------------------------------------------------------------


def tp_aggregation_standard(
    re_predictions: Dict[str, bool], p_ids: List[str]
) -> bool:
    """TP Strategy B for standard AF: affirm if majority of P-claims accepted.

    Tort affirmed if (# accepted P-claims) / (# total P-claims) > 0.5.
    """
    if not p_ids:
        return False
    n_accepted = sum(1 for pid in p_ids if re_predictions.get(pid, False))
    return n_accepted / len(p_ids) > 0.5


def tp_aggregation_quantitative(
    strengths: Dict[str, float], p_ids: List[str]
) -> Tuple[bool, float]:
    """TP Strategy B for quantitative AF: affirm if mean P-claim strength > 0.5.

    Returns (affirmed, mean_strength).
    """
    if not p_ids:
        return False, 0.0
    mean_s = sum(strengths.get(pid, 0.0) for pid in p_ids) / len(p_ids)
    return mean_s > 0.5, mean_s


# ---------------------------------------------------------------------------
# BH: Heuristic QBAF construction (no LLM)
# ---------------------------------------------------------------------------


def construct_heuristic_qbaf(case: TortCase) -> QBAF:
    """Construct BH heuristic QBAF.

    - Arguments: all P, D, U
    - Base strengths: 0.5 for P/D, 1.0 for U
    - Attacks: complete bipartite P <-> D
    - Supports: each U supports each P and each D
    """
    args = set()
    strengths = {}

    # Undisputed facts
    for i in range(len(case.undisputed_facts)):
        uid = f"u{i}"
        args.add(uid)
        strengths[uid] = 1.0

    # Plaintiff claims
    p_ids = []
    for i in range(len(case.plaintiff_claims)):
        pid = f"p{i}"
        args.add(pid)
        strengths[pid] = 0.5
        p_ids.append(pid)

    # Defendant claims
    d_ids = []
    for i in range(len(case.defendant_claims)):
        did = f"d{i}"
        args.add(did)
        strengths[did] = 0.5
        d_ids.append(did)

    # Complete bipartite attacks
    attacks = set()
    for p in p_ids:
        for d in d_ids:
            attacks.add((p, d))
            attacks.add((d, p))

    # U supports all claims
    supports = set()
    u_ids = [f"u{i}" for i in range(len(case.undisputed_facts))]
    for u in u_ids:
        for p in p_ids:
            supports.add((u, p))
        for d in d_ids:
            supports.add((u, d))

    return QBAF(
        arguments=args,
        base_strengths=strengths,
        attacks=attacks,
        supports=supports,
    )


# ---------------------------------------------------------------------------
# Full pipeline helpers
# ---------------------------------------------------------------------------


def compute_standard_predictions(
    af: DungAF,
    case: TortCase,
    semantics: str = "grounded",
    timeout_s: float = 30.0,
) -> Dict:
    """Run full standard AF prediction pipeline.

    Returns dict with all predictions and intermediate results.
    """
    p_ids = [f"p{i}" for i in range(len(case.plaintiff_claims))]
    d_ids = [f"d{i}" for i in range(len(case.defendant_claims))]

    result = {"semantics_type": semantics}

    if semantics == "grounded":
        in_set, out_set, undecided = grounded_extension(af)
        result["in_set"] = sorted(in_set)
        result["out_set"] = sorted(out_set)
        result["undecided"] = sorted(undecided)

        re_preds = re_from_extension(in_set, p_ids, d_ids)
        result["re_predictions"] = re_preds

        # TP Strategy A
        result["tp_a"] = tp_meta_argument_grounded(af, p_ids, d_ids)

        # TP Strategy B
        result["tp_b"] = tp_aggregation_standard(re_preds, p_ids)

    elif semantics == "preferred":
        exts = preferred_extensions(af, timeout_s=timeout_s)
        if exts is None:
            # Timeout: fall back to grounded
            result["timeout"] = True
            result["fallback"] = "grounded"
            in_set, out_set, undecided = grounded_extension(af)
            result["in_set"] = sorted(in_set)
            result["out_set"] = sorted(out_set)
            result["undecided"] = sorted(undecided)

            re_preds = re_from_extension(in_set, p_ids, d_ids)
            result["re_predictions"] = re_preds
            result["re_predictions_credulous"] = re_preds
            result["re_predictions_skeptical"] = re_preds
            result["tp_a_credulous"] = tp_meta_argument_grounded(af, p_ids, d_ids)
            result["tp_a_skeptical"] = result["tp_a_credulous"]
            result["tp_b"] = tp_aggregation_standard(re_preds, p_ids)
        else:
            result["timeout"] = False
            result["n_extensions"] = len(exts)
            result["extensions"] = [sorted(e) for e in exts]

            cred = credulous_acceptance(exts)
            skep = skeptical_acceptance(exts)
            result["credulous"] = sorted(cred)
            result["skeptical"] = sorted(skep)

            # RE: credulous and skeptical
            re_cred = re_from_extension(cred, p_ids, d_ids)
            re_skep = re_from_extension(skep, p_ids, d_ids)
            result["re_predictions_credulous"] = re_cred
            result["re_predictions_skeptical"] = re_skep
            # Default RE uses credulous
            result["re_predictions"] = re_cred

            # TP Strategy A
            tp_cred, tp_skep, tp_timeout = tp_meta_argument_preferred(
                af, p_ids, d_ids, timeout_s=timeout_s
            )
            if tp_timeout:
                # Fall back to grounded for TP meta-argument
                result["tp_a_credulous"] = tp_meta_argument_grounded(
                    af, p_ids, d_ids
                )
                result["tp_a_skeptical"] = result["tp_a_credulous"]
                result["tp_a_timeout"] = True
            else:
                result["tp_a_credulous"] = tp_cred
                result["tp_a_skeptical"] = tp_skep
                result["tp_a_timeout"] = False

            # TP Strategy B (from credulous RE)
            result["tp_b"] = tp_aggregation_standard(re_cred, p_ids)

    return result


def compute_baf_predictions(
    baf: BipolarAF, case: TortCase
) -> Dict:
    """Run full BAF prediction pipeline (B4).

    Reduces to Dung AF, computes grounded extension.
    """
    p_ids = [f"p{i}" for i in range(len(case.plaintiff_claims))]
    d_ids = [f"d{i}" for i in range(len(case.defendant_claims))]

    dung_af = baf_to_dung(baf)
    in_set, out_set, undecided = grounded_extension(dung_af)

    re_preds = re_from_extension(in_set, p_ids, d_ids)

    return {
        "semantics_type": "grounded_via_reduction",
        "in_set": sorted(in_set),
        "out_set": sorted(out_set),
        "undecided": sorted(undecided),
        "re_predictions": re_preds,
        "tp_a": tp_meta_argument_baf(baf, p_ids, d_ids),
        "tp_b": tp_aggregation_standard(re_preds, p_ids),
    }


def compute_quantitative_predictions(
    qbaf: QBAF,
    case: TortCase,
    threshold: float = 0.5,
) -> Dict:
    """Run full quantitative AF prediction pipeline (B5, B6, BH).

    Returns dict with final strengths, RE predictions, and TP predictions.
    """
    p_ids = [f"p{i}" for i in range(len(case.plaintiff_claims))]
    d_ids = [f"d{i}" for i in range(len(case.defendant_claims))]

    final_strengths = df_quad(qbaf)

    re_preds = re_from_strengths(final_strengths, p_ids, d_ids, threshold=threshold)
    claim_strengths = re_strengths_for_claims(final_strengths, p_ids, d_ids)

    tp_a, t_strength = tp_meta_argument_quantitative(qbaf, p_ids, d_ids)
    tp_b, mean_p_strength = tp_aggregation_quantitative(final_strengths, p_ids)

    return {
        "semantics_type": "df_quad",
        "final_strengths": {k: round(v, 6) for k, v in final_strengths.items()},
        "claim_strengths": {k: round(v, 6) for k, v in claim_strengths.items()},
        "threshold": threshold,
        "re_predictions": re_preds,
        "tp_a": tp_a,
        "tp_a_t_strength": round(t_strength, 6),
        "tp_b": tp_b,
        "tp_b_mean_p_strength": round(mean_p_strength, 6),
    }
