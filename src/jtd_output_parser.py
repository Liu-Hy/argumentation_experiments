"""Parse LLM outputs for JTD argumentation framework experiments.

Handles common LLM output issues: code fences, trailing commas, etc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction (adapted from output_parser.py)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _try_parse_json(s: str) -> Optional[dict]:
    """Attempt to parse JSON with tolerance for common issues."""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    return None


def _find_brace_blocks(text: str) -> List[str]:
    """Find all top-level balanced { ... } blocks."""
    blocks = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        blocks.append(text[start : j + 1])
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return blocks


def extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM output.

    Tries code fences first, then brace blocks, then full text.
    Prefers the last valid JSON block (final answer after reasoning).
    """
    # 1. Code fences (prefer last valid one)
    fences = list(_CODE_FENCE_RE.finditer(text))
    for m in reversed(fences):
        parsed = _try_parse_json(m.group(1).strip())
        if parsed is not None:
            return parsed

    # 2. Brace blocks (prefer last one)
    candidates = _find_brace_blocks(text)
    for c in reversed(candidates):
        parsed = _try_parse_json(c)
        if parsed is not None:
            return parsed

    # 3. Full text
    return _try_parse_json(text)


# ---------------------------------------------------------------------------
# Parse results dataclass
# ---------------------------------------------------------------------------


@dataclass
class AFParseResult:
    """Result of parsing an LLM response for AF construction."""

    success: bool = False
    attacks: List[Tuple[str, str]] = field(default_factory=list)
    supports: List[Tuple[str, str]] = field(default_factory=list)
    argument_strengths: Dict[str, float] = field(default_factory=dict)
    raw_json: Optional[dict] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class DirectPredictionResult:
    """Result of parsing B0/B1 direct prediction output."""

    success: bool = False
    plaintiff_predictions: Dict[str, bool] = field(default_factory=dict)
    defendant_predictions: Dict[str, bool] = field(default_factory=dict)
    tort_decision: Optional[bool] = None
    raw_json: Optional[dict] = None
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Relation parsing helpers
# ---------------------------------------------------------------------------


def _parse_relation_list(
    items: list, valid_ids: Set[str], relation_type: str
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Parse a list of relation dicts, validating IDs.

    Returns (valid_relations, error_messages).
    """
    relations = []
    errors = []
    for item in items:
        if not isinstance(item, dict):
            errors.append(f"Non-dict item in {relation_type}: {item}")
            continue
        src = str(item.get("source", ""))
        tgt = str(item.get("target", ""))
        if src not in valid_ids:
            errors.append(f"Unknown source ID in {relation_type}: {src}")
            continue
        if tgt not in valid_ids:
            errors.append(f"Unknown target ID in {relation_type}: {tgt}")
            continue
        if src == tgt:
            errors.append(f"Self-{relation_type}: {src}")
            continue
        relations.append((src, tgt))
    return relations, errors


def _parse_strengths(
    strengths_dict: dict, valid_ids: Set[str]
) -> Tuple[Dict[str, float], List[str]]:
    """Parse argument strengths, validating IDs and values.

    Returns (strengths, error_messages).
    """
    strengths = {}
    errors = []
    for arg_id, val in strengths_dict.items():
        arg_id = str(arg_id)
        if arg_id not in valid_ids:
            errors.append(f"Unknown argument ID in strengths: {arg_id}")
            continue
        try:
            s = float(val)
            strengths[arg_id] = max(0.0, min(1.0, s))
        except (ValueError, TypeError):
            errors.append(f"Invalid strength for {arg_id}: {val}")
    return strengths, errors


def _coerce_bool(val) -> Optional[bool]:
    """Coerce common JSON/LLM boolean representations to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"true", "t", "1", "yes", "y"}:
            return True
        if low in {"false", "f", "0", "no", "n"}:
            return False
    return None


# ---------------------------------------------------------------------------
# Method-specific parsers
# ---------------------------------------------------------------------------


def parse_dung_af(
    raw_output: str, valid_ids: Set[str]
) -> AFParseResult:
    """Parse Dung AF output (B2/B3): attacks only.

    valid_ids: set of valid argument IDs (p0, p1, ..., d0, d1, ...).
    """
    result = AFParseResult()
    data = extract_json(raw_output)
    if data is None:
        result.errors.append("Could not extract JSON from output")
        return result

    result.raw_json = data
    attacks_raw = data.get("attacks", [])
    if not isinstance(attacks_raw, list):
        result.errors.append(f"'attacks' is not a list: {type(attacks_raw)}")
        return result

    result.attacks, errs = _parse_relation_list(attacks_raw, valid_ids, "attack")
    result.errors.extend(errs)
    result.success = True
    return result


def parse_baf(
    raw_output: str, valid_ids: Set[str], undisputed_ids: Set[str]
) -> AFParseResult:
    """Parse BAF output (B4): attacks + supports.

    Enforces constraints on undisputed facts:
    - no attack can target or originate from an undisputed fact
    - undisputed facts can only be support sources (never targets)
    """
    result = AFParseResult()
    data = extract_json(raw_output)
    if data is None:
        result.errors.append("Could not extract JSON from output")
        return result

    result.raw_json = data

    # Parse attacks
    attacks_raw = data.get("attacks", [])
    if isinstance(attacks_raw, list):
        all_attacks, errs = _parse_relation_list(attacks_raw, valid_ids, "attack")
        result.errors.extend(errs)
        # Undisputed facts are axioms: cannot attack or be attacked.
        for src, tgt in all_attacks:
            if src in undisputed_ids or tgt in undisputed_ids:
                result.errors.append(
                    f"Dropped invalid attack with undisputed fact: {src} -> {tgt}"
                )
            else:
                result.attacks.append((src, tgt))

    # Parse supports
    supports_raw = data.get("supports", [])
    if isinstance(supports_raw, list):
        all_supports, errs = _parse_relation_list(supports_raw, valid_ids, "support")
        result.errors.extend(errs)
        # Undisputed facts may only be support sources.
        for src, tgt in all_supports:
            if tgt in undisputed_ids:
                result.errors.append(
                    f"Dropped invalid support targeting undisputed fact: {src} -> {tgt}"
                )
            else:
                result.supports.append((src, tgt))

    result.success = True
    return result


def parse_quaf(
    raw_output: str, valid_ids: Set[str]
) -> AFParseResult:
    """Parse Quantitative AF output (B5): attacks + strengths."""
    result = AFParseResult()
    data = extract_json(raw_output)
    if data is None:
        result.errors.append("Could not extract JSON from output")
        return result

    result.raw_json = data

    # Parse attacks
    attacks_raw = data.get("attacks", [])
    if isinstance(attacks_raw, list):
        result.attacks, errs = _parse_relation_list(attacks_raw, valid_ids, "attack")
        result.errors.extend(errs)

    # Parse strengths
    strengths_raw = data.get("argument_strengths", {})
    if isinstance(strengths_raw, dict):
        result.argument_strengths, errs = _parse_strengths(strengths_raw, valid_ids)
        result.errors.extend(errs)
    else:
        result.errors.append(f"'argument_strengths' is not a dict")

    # Fill missing strengths with default 0.5
    for arg_id in valid_ids:
        if arg_id not in result.argument_strengths:
            result.argument_strengths[arg_id] = 0.5
            result.errors.append(f"Missing strength for {arg_id}, defaulting to 0.5")

    result.success = True
    return result


def parse_qbaf(
    raw_output: str, valid_ids: Set[str], undisputed_ids: Set[str]
) -> AFParseResult:
    """Parse Quantitative BAF output (B6): attacks + supports + strengths.

    Enforces constraints on undisputed facts.
    """
    result = AFParseResult()
    data = extract_json(raw_output)
    if data is None:
        result.errors.append("Could not extract JSON from output")
        return result

    result.raw_json = data

    # Parse attacks (undisputed facts cannot attack or be attacked)
    attacks_raw = data.get("attacks", [])
    if isinstance(attacks_raw, list):
        all_attacks, errs = _parse_relation_list(attacks_raw, valid_ids, "attack")
        result.errors.extend(errs)
        for src, tgt in all_attacks:
            if src in undisputed_ids or tgt in undisputed_ids:
                result.errors.append(
                    f"Dropped invalid attack with undisputed fact: {src} -> {tgt}"
                )
            else:
                result.attacks.append((src, tgt))

    # Parse supports
    supports_raw = data.get("supports", [])
    if isinstance(supports_raw, list):
        all_supports, errs = _parse_relation_list(supports_raw, valid_ids, "support")
        result.errors.extend(errs)
        # Undisputed facts may only be support sources.
        for src, tgt in all_supports:
            if tgt in undisputed_ids:
                result.errors.append(
                    f"Dropped invalid support targeting undisputed fact: {src} -> {tgt}"
                )
            else:
                result.supports.append((src, tgt))

    # Parse strengths
    strengths_raw = data.get("argument_strengths", {})
    if isinstance(strengths_raw, dict):
        result.argument_strengths, errs = _parse_strengths(strengths_raw, valid_ids)
        result.errors.extend(errs)

    # Enforce undisputed facts have strength 1.0, fill missing
    for arg_id in valid_ids:
        if arg_id in undisputed_ids:
            result.argument_strengths[arg_id] = 1.0
        elif arg_id not in result.argument_strengths:
            result.argument_strengths[arg_id] = 0.5
            result.errors.append(f"Missing strength for {arg_id}, defaulting to 0.5")

    result.success = True
    return result


def parse_direct_prediction(
    raw_output: str, n_plaintiff: int, n_defendant: int
) -> DirectPredictionResult:
    """Parse B0/B1 direct prediction output.

    Expected format:
    {
      "plaintiff_claims": {"p0": true, "p1": false, ...},
      "defendant_claims": {"d0": false, ...},
      "tort_decision": true
    }
    """
    result = DirectPredictionResult()
    data = extract_json(raw_output)
    if data is None:
        result.errors.append("Could not extract JSON from output")
        return result

    result.raw_json = data

    # Parse plaintiff predictions
    p_preds = data.get("plaintiff_claims", {})
    if isinstance(p_preds, dict):
        for i in range(n_plaintiff):
            key = f"p{i}"
            val = p_preds.get(key)
            if val is not None:
                parsed = _coerce_bool(val)
                if parsed is None:
                    result.plaintiff_predictions[key] = False
                    result.errors.append(f"Invalid boolean for {key}: {val}")
                else:
                    result.plaintiff_predictions[key] = parsed
            else:
                result.plaintiff_predictions[key] = False
                result.errors.append(f"Missing prediction for {key}")
    elif isinstance(p_preds, list):
        # Handle list format: [true, false, ...]
        for i, val in enumerate(p_preds):
            if i < n_plaintiff:
                parsed = _coerce_bool(val)
                if parsed is None:
                    result.plaintiff_predictions[f"p{i}"] = False
                    result.errors.append(f"Invalid boolean for p{i}: {val}")
                else:
                    result.plaintiff_predictions[f"p{i}"] = parsed

    # Parse defendant predictions
    d_preds = data.get("defendant_claims", {})
    if isinstance(d_preds, dict):
        for i in range(n_defendant):
            key = f"d{i}"
            val = d_preds.get(key)
            if val is not None:
                parsed = _coerce_bool(val)
                if parsed is None:
                    result.defendant_predictions[key] = False
                    result.errors.append(f"Invalid boolean for {key}: {val}")
                else:
                    result.defendant_predictions[key] = parsed
            else:
                result.defendant_predictions[key] = False
                result.errors.append(f"Missing prediction for {key}")
    elif isinstance(d_preds, list):
        for i, val in enumerate(d_preds):
            if i < n_defendant:
                parsed = _coerce_bool(val)
                if parsed is None:
                    result.defendant_predictions[f"d{i}"] = False
                    result.errors.append(f"Invalid boolean for d{i}: {val}")
                else:
                    result.defendant_predictions[f"d{i}"] = parsed

    # Parse tort decision
    td = data.get("tort_decision")
    if td is not None:
        parsed_td = _coerce_bool(td)
        if parsed_td is None:
            result.errors.append(f"Invalid tort_decision: {td}")
        else:
            result.tort_decision = parsed_td
    else:
        result.errors.append("Missing tort_decision")

    result.success = len(result.plaintiff_predictions) > 0 or len(
        result.defendant_predictions
    ) > 0
    return result
