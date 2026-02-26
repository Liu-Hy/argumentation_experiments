"""Parse LLM outputs into BAF structures.

Handles common failure modes: markdown-wrapped JSON, paraphrased quotes,
missing offsets, malformed JSON, etc.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .baf import Argument, BAF, Relation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parse statistics (for transparency about dropped predictions)
# ---------------------------------------------------------------------------


@dataclass
class ParseStats:
    """Counters for what the parser kept vs dropped."""

    json_extracted: bool = False
    args_in_json: int = 0
    args_resolved: int = 0
    args_dropped: int = 0  # could not locate in essay text
    rels_in_json: int = 0
    rels_kept: int = 0
    rels_dropped_bad_type: int = 0
    rels_dropped_bad_id: int = 0

    def to_dict(self) -> dict:
        return {
            "json_extracted": self.json_extracted,
            "args_in_json": self.args_in_json,
            "args_resolved": self.args_resolved,
            "args_dropped": self.args_dropped,
            "rels_in_json": self.rels_in_json,
            "rels_kept": self.rels_kept,
            "rels_dropped_bad_type": self.rels_dropped_bad_type,
            "rels_dropped_bad_id": self.rels_dropped_bad_id,
        }


def parse_llm_output(raw_output: str, essay_text: str) -> Tuple[BAF, ParseStats]:
    """Parse an LLM response string into a BAF.

    Returns (BAF, ParseStats) so callers can inspect drop rates.
    """
    stats = ParseStats()
    data = _extract_json(raw_output)
    if data is None:
        logger.warning("Could not extract valid JSON from LLM output")
        return BAF(), stats

    stats.json_extracted = True
    raw_args = data.get("arguments", [])
    raw_rels = data.get("relations", [])
    stats.args_in_json = len(raw_args)
    stats.rels_in_json = len(raw_rels)

    arguments = _parse_arguments(raw_args, essay_text, stats)
    arg_ids = {a.id for a in arguments}
    relations = _parse_relations(raw_rels, arg_ids, stats)

    return BAF(arguments=arguments, relations=relations), stats


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)

_EXPECTED_KEYS = {"arguments", "relations"}


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from LLM output.

    Tries in order:
    1. Content inside ```json ... ``` fences (last fence wins, to handle
       CoT outputs that may have earlier fences with partial data).
    2. All top-level { ... } blocks, preferring those containing expected
       keys ("arguments" or "relations").
    3. The full text as JSON.
    """
    # 1. Try code fences (prefer the last valid one — CoT may have earlier junk)
    fences = list(_CODE_FENCE_RE.finditer(text))
    for m in reversed(fences):
        candidate = m.group(1).strip()
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            return parsed

    # 2. Try all top-level { ... } blocks
    candidates = _find_brace_blocks(text)
    # Prefer candidates that contain expected keys
    with_keys = []
    without_keys = []
    for c in candidates:
        parsed = _try_parse_json(c)
        if parsed is not None:
            if _EXPECTED_KEYS & set(parsed.keys()):
                with_keys.append(parsed)
            else:
                without_keys.append(parsed)
    if with_keys:
        return with_keys[-1]  # prefer last (likely the final answer after reasoning)
    if without_keys:
        return without_keys[-1]

    # 3. Last resort: try the whole thing
    return _try_parse_json(text)


def _find_brace_blocks(text: str) -> List[str]:
    """Find all top-level balanced { ... } blocks in text."""
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
                break  # unbalanced
        else:
            i += 1
    return blocks


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


def _parse_arguments(
    arg_list: list, essay_text: str, stats: ParseStats
) -> List[Argument]:
    """Parse the 'arguments' array from the LLM JSON."""
    arguments = []
    for item in arg_list:
        if not isinstance(item, dict):
            stats.args_dropped += 1
            continue
        arg_id = str(item.get("id", f"a{len(arguments)+1}"))
        text = item.get("text", "")
        if not text:
            stats.args_dropped += 1
            continue

        start = item.get("start")
        end = item.get("end")

        # Validate provided offsets
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(essay_text)
        ):
            actual_text = essay_text[start:end]
            ratio = difflib.SequenceMatcher(None, text, actual_text).ratio()
            if ratio >= 0.8:
                arguments.append(
                    Argument(id=arg_id, text=actual_text, start=start, end=end)
                )
                stats.args_resolved += 1
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
            stats.args_resolved += 1
        else:
            stats.args_dropped += 1
            logger.debug(
                f"Could not locate argument text in essay: {text[:60]}..."
            )

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

    # Word-start positions: m.end() gives the char after whitespace = start of next word
    word_starts = [0] + [m.end() for m in re.finditer(r"\s+", text)]

    for ws in word_starts:
        # Try windows of varying size around query_len
        for factor in [1.0, 0.8, 1.2]:
            wlen = max(1, int(query_len * factor))
            candidate = text[ws : ws + wlen]
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(
                None, query.lower(), candidate.lower()
            ).ratio()
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


def _parse_relations(
    rel_list: list, valid_arg_ids: set, stats: Optional[ParseStats] = None
) -> List[Relation]:
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
            if stats:
                stats.rels_dropped_bad_type += 1
            logger.debug(f"Skipping relation with unknown type: {rtype}")
            continue
        if source not in valid_arg_ids or target not in valid_arg_ids:
            if stats:
                stats.rels_dropped_bad_id += 1
            logger.debug(
                f"Skipping relation with unknown arg id: {source} -> {target}"
            )
            continue

        relations.append(Relation(source=source, target=target, type=rtype))
        if stats:
            stats.rels_kept += 1

    return relations


# ---------------------------------------------------------------------------
# Gold-argument setting: parse relations-only output
# ---------------------------------------------------------------------------


def parse_relations_only(
    raw_output: str, gold_baf: BAF
) -> Tuple[BAF, ParseStats]:
    """Parse LLM output for the gold-argument setting.

    The model was given gold arguments renumbered as a1, a2, ... (in
    enumeration order).  We remap the model's a* IDs back to the
    original gold IDs before validation.
    """
    stats = ParseStats()

    data = _extract_json(raw_output)
    if data is None:
        logger.warning("Could not extract valid JSON from LLM output (gold-arg)")
        return BAF(arguments=list(gold_baf.arguments), relations=[]), stats

    stats.json_extracted = True

    # Build reverse map: a{i} -> original gold ID (same order as the prompt)
    reverse_map: Dict[str, str] = {}
    for i, arg in enumerate(gold_baf.arguments, 1):
        reverse_map[f"a{i}"] = arg.id

    # Remap relation IDs from a* back to original gold IDs
    raw_rels = data.get("relations", [])
    stats.rels_in_json = len(raw_rels)
    remapped_rels = []
    for item in raw_rels:
        if not isinstance(item, dict):
            continue
        src_raw = str(item.get("source", ""))
        tgt_raw = str(item.get("target", ""))
        src = reverse_map.get(src_raw, src_raw)  # remap or keep as-is
        tgt = reverse_map.get(tgt_raw, tgt_raw)
        remapped_rels.append(
            {"source": src, "target": tgt, "type": item.get("type", "")}
        )

    gold_ids = {a.id for a in gold_baf.arguments}
    relations = _parse_relations(remapped_rels, gold_ids, stats)

    return BAF(arguments=list(gold_baf.arguments), relations=relations), stats


# ---------------------------------------------------------------------------
# Pairwise relation classification parsing
# ---------------------------------------------------------------------------


def parse_pairwise_response(
    raw_output: str, arg1_id: str, arg2_id: str
) -> Optional[Relation]:
    """Parse a pairwise relation classification response.

    The LLM was asked to classify using placeholder IDs 'arg1' and 'arg2'.
    This function remaps them to the actual argument IDs.

    Returns a Relation if support/attack was identified, or None for
    "no relation" or parse failure.
    """
    data = _extract_json(raw_output)
    if data is None:
        return None

    # "none" responses
    if data.get("relation") == "none" or data.get("type") == "none":
        return None

    source = str(data.get("source", ""))
    target = str(data.get("target", ""))
    rtype = str(data.get("type", "")).lower().strip()

    # Normalise common LLM variations
    if rtype in ("supports", "supporting"):
        rtype = "support"
    elif rtype in ("attacks", "attacking"):
        rtype = "attack"

    if rtype not in ("support", "attack"):
        return None

    # Remap placeholder IDs -> actual IDs
    remap = {"arg1": arg1_id, "arg2": arg2_id}
    actual_src = remap.get(source, source)
    actual_tgt = remap.get(target, target)

    valid_ids = {arg1_id, arg2_id}
    if actual_src not in valid_ids or actual_tgt not in valid_ids:
        return None
    if actual_src == actual_tgt:
        return None

    return Relation(source=actual_src, target=actual_tgt, type=rtype)
