"""Parse LLM outputs into BAF structures.

Handles common failure modes: markdown-wrapped JSON, paraphrased quotes,
missing offsets, malformed JSON, etc.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from .baf import Argument, BAF, Relation

logger = logging.getLogger(__name__)


def parse_llm_output(raw_output: str, essay_text: str) -> BAF:
    """Parse an LLM response string into a BAF.

    1. Extract JSON from the raw output (handles markdown code fences).
    2. Build Argument objects, resolving character offsets via fuzzy matching
       against the essay text when they are missing or incorrect.
    3. Build Relation objects.
    """
    data = _extract_json(raw_output)
    if data is None:
        logger.warning("Could not extract valid JSON from LLM output")
        return BAF()

    arguments = _parse_arguments(data.get("arguments", []), essay_text)
    arg_ids = {a.id for a in arguments}
    relations = _parse_relations(data.get("relations", []), arg_ids)

    return BAF(arguments=arguments, relations=relations)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)```", re.DOTALL
)


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from LLM output.

    Tries in order:
    1. Content inside ```json ... ``` fences
    2. First { ... } block in the text
    3. The full text as JSON
    """
    # Try code fence first
    m = _CODE_FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            return parsed

    # Try to find outermost { ... }
    brace_start = text.find("{")
    if brace_start != -1:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    parsed = _try_parse_json(candidate)
                    if parsed is not None:
                        return parsed
                    break

    # Last resort: try the whole thing
    return _try_parse_json(text)


def _try_parse_json(s: str) -> Optional[dict]:
    """Attempt to parse JSON, with some tolerance for common issues."""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try fixing trailing commas (common LLM mistake)
    cleaned = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Argument parsing with fuzzy offset resolution
# ---------------------------------------------------------------------------


def _parse_arguments(arg_list: list, essay_text: str) -> List[Argument]:
    """Parse the 'arguments' array from the LLM JSON."""
    arguments = []
    for item in arg_list:
        if not isinstance(item, dict):
            continue
        arg_id = str(item.get("id", f"a{len(arguments)+1}"))
        text = item.get("text", "")
        if not text:
            continue

        start = item.get("start")
        end = item.get("end")

        # Validate provided offsets
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(essay_text)
        ):
            # Check if the text at these offsets is a reasonable match
            actual_text = essay_text[start:end]
            ratio = difflib.SequenceMatcher(None, text, actual_text).ratio()
            if ratio >= 0.8:
                arguments.append(Argument(id=arg_id, text=actual_text, start=start, end=end))
                continue

        # Offsets missing or wrong -> fuzzy match the text in the essay
        resolved = _fuzzy_find(text, essay_text)
        if resolved is not None:
            rstart, rend = resolved
            arguments.append(
                Argument(
                    id=arg_id,
                    text=essay_text[rstart:rend],
                    start=rstart,
                    end=rend,
                )
            )
        else:
            logger.debug(f"Could not locate argument text in essay: {text[:60]}...")

    return arguments


def _fuzzy_find(query: str, text: str) -> Optional[Tuple[int, int]]:
    """Find the best approximate location of `query` within `text`.

    Uses difflib.SequenceMatcher to find the longest contiguous match,
    then extends it to roughly the length of the query.
    Returns (start, end) character offsets or None if no good match.
    """
    if not query or not text:
        return None

    # First try exact substring match
    idx = text.find(query)
    if idx != -1:
        return (idx, idx + len(query))

    # Try case-insensitive exact match
    idx = text.lower().find(query.lower())
    if idx != -1:
        return (idx, idx + len(query))

    # Fuzzy matching: slide a window roughly the size of the query over the text
    query_len = len(query)
    best_ratio = 0.0
    best_start = 0
    best_end = 0

    # Use a coarse search first (step by word boundaries)
    words_starts = [0] + [m.start() for m in re.finditer(r"\s+", text)]

    for ws in words_starts:
        # Try windows of varying size around query_len
        for factor in [1.0, 0.8, 1.2]:
            wlen = max(1, int(query_len * factor))
            candidate = text[ws : ws + wlen]
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = ws
                best_end = ws + wlen

    if best_ratio >= 0.6:
        # Clamp to text bounds
        best_end = min(best_end, len(text))
        return (best_start, best_end)

    return None


# ---------------------------------------------------------------------------
# Relation parsing
# ---------------------------------------------------------------------------

_VALID_TYPES = {"support", "attack"}


def _parse_relations(rel_list: list, valid_arg_ids: set) -> List[Relation]:
    """Parse the 'relations' array from the LLM JSON."""
    relations = []
    for item in rel_list:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        target = str(item.get("target", ""))
        rtype = str(item.get("type", "")).lower().strip()

        # Normalise common LLM variations
        if rtype in ("supports", "supporting"):
            rtype = "support"
        elif rtype in ("attacks", "attacking"):
            rtype = "attack"

        if rtype not in _VALID_TYPES:
            logger.debug(f"Skipping relation with unknown type: {rtype}")
            continue
        if source not in valid_arg_ids or target not in valid_arg_ids:
            logger.debug(
                f"Skipping relation with unknown arg id: {source} -> {target}"
            )
            continue

        relations.append(Relation(source=source, target=target, type=rtype))

    return relations


# ---------------------------------------------------------------------------
# Gold-argument setting: parse relations-only output
# ---------------------------------------------------------------------------


def parse_relations_only(raw_output: str, gold_baf: BAF) -> BAF:
    """Parse LLM output for the gold-argument setting.

    The model was given gold arguments and asked to predict only relations.
    We re-use the gold arguments and parse just the relations from the output.
    """
    data = _extract_json(raw_output)
    if data is None:
        logger.warning("Could not extract valid JSON from LLM output (gold-arg)")
        return BAF(arguments=list(gold_baf.arguments), relations=[])

    gold_ids = {a.id for a in gold_baf.arguments}
    relations = _parse_relations(data.get("relations", []), gold_ids)
    return BAF(arguments=list(gold_baf.arguments), relations=relations)
