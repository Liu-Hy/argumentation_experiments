"""Prompt templates for BAF extraction experiments.

Five methods:
  1. ZS-E2E    -- Zero-shot end-to-end
  2. FS-E2E    -- Few-shot (3-shot) end-to-end
  3. ZS-Pipe   -- Zero-shot two-step pipeline
  4. FS-Pipe   -- Few-shot two-step pipeline
  5. FS-CoT-E2E -- Few-shot + chain-of-thought end-to-end

Plus gold-argument variants (relation prediction only).
"""

from __future__ import annotations

from typing import Dict, List

from .baf import BAF

# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

TASK_DESCRIPTION = """\
You are an expert in argumentation theory. Your task is to analyze a \
persuasive essay and extract its Bipolar Argumentation Framework (BAF).

A BAF consists of:
- **Arguments**: text spans from the essay that express argumentative points \
(claims, premises, main theses — all treated uniformly as "arguments").
- **Relations**: directed edges between arguments. Each relation is either:
  - "support": the source argument provides evidence or reasoning that \
supports the target argument.
  - "attack": the source argument provides evidence or reasoning that \
undermines or opposes the target argument.\
"""

OUTPUT_SCHEMA = """\
Return your answer as a JSON object with this exact schema:
{
  "arguments": [
    {"id": "a1", "text": "<exact verbatim quote from the essay>"},
    {"id": "a2", "text": "<exact verbatim quote from the essay>"}
  ],
  "relations": [
    {"source": "a1", "target": "a2", "type": "support"}
  ]
}

IMPORTANT RULES:
- The "text" field of each argument MUST be an exact, verbatim substring \
copied from the essay. Do not paraphrase or summarize.
- Argument IDs should be "a1", "a2", "a3", etc.
- Every relation must reference valid argument IDs from your arguments list.
- Only include "support" and "attack" as relation types.
- Return ONLY the JSON object, no other text.\
"""

OUTPUT_SCHEMA_COT = """\
First, reason step by step about the essay's argumentative structure:
1. Identify the author's main thesis or central claim.
2. Identify arguments that support this thesis and arguments that oppose it.
3. For each argument, determine what evidence or reasoning supports or \
attacks it.
4. Trace the full network of support and attack relations.

After your reasoning, return your answer as a JSON object with this schema:
{
  "arguments": [
    {"id": "a1", "text": "<exact verbatim quote from the essay>"},
    {"id": "a2", "text": "<exact verbatim quote from the essay>"}
  ],
  "relations": [
    {"source": "a1", "target": "a2", "type": "support"}
  ]
}

IMPORTANT RULES:
- The "text" field of each argument MUST be an exact, verbatim substring \
copied from the essay. Do not paraphrase or summarize.
- Argument IDs should be "a1", "a2", "a3", etc.
- Every relation must reference valid argument IDs from your arguments list.
- Only include "support" and "attack" as relation types.\
"""


def _format_example(essay_text: str, baf: BAF) -> str:
    """Format a single essay + BAF as a few-shot example string."""
    # Re-assign sequential IDs for clarity
    id_map = {}
    args_json = []
    for i, arg in enumerate(baf.arguments, 1):
        new_id = f"a{i}"
        id_map[arg.id] = new_id
        args_json.append({"id": new_id, "text": arg.text})

    rels_json = []
    for rel in baf.relations:
        src = id_map.get(rel.source)
        tgt = id_map.get(rel.target)
        if src and tgt:
            rels_json.append({"source": src, "target": tgt, "type": rel.type})

    import json

    baf_str = json.dumps(
        {"arguments": args_json, "relations": rels_json}, indent=2
    )
    return f"Essay:\n{essay_text.strip()}\n\nBAF:\n{baf_str}"


# ---------------------------------------------------------------------------
# Method 1: ZS-E2E (Zero-shot End-to-End)
# ---------------------------------------------------------------------------


def zs_e2e(essay_text: str) -> List[Dict[str, str]]:
    """Zero-shot end-to-end prompt. Returns list of messages."""
    return [
        {
            "role": "system",
            "content": f"{TASK_DESCRIPTION}\n\n{OUTPUT_SCHEMA}",
        },
        {
            "role": "user",
            "content": f"Analyze the following essay and extract its BAF.\n\n"
            f"Essay:\n{essay_text.strip()}",
        },
    ]


# ---------------------------------------------------------------------------
# Method 2: FS-E2E (Few-shot End-to-End)
# ---------------------------------------------------------------------------


def fs_e2e(
    essay_text: str,
    examples: List[Dict],  # [{"text": str, "baf": BAF}, ...]
) -> List[Dict[str, str]]:
    """Few-shot end-to-end prompt."""
    examples_str = "\n\n---\n\n".join(
        _format_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {
            "role": "system",
            "content": f"{TASK_DESCRIPTION}\n\n{OUTPUT_SCHEMA}",
        },
        {
            "role": "user",
            "content": (
                f"Here are some examples of essays and their BAFs:\n\n"
                f"{examples_str}\n\n"
                f"---\n\n"
                f"Now analyze this essay and extract its BAF.\n\n"
                f"Essay:\n{essay_text.strip()}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Method 3: ZS-Pipe (Zero-shot Pipeline)
# ---------------------------------------------------------------------------


_PIPE_STEP1_SYSTEM = """\
You are an expert in argumentation theory. Your task is to identify all \
arguments in a persuasive essay.

An argument is any text span that expresses an argumentative point: a claim, \
a piece of evidence, a thesis statement, a counter-argument, etc.

Return your answer as a JSON object:
{
  "arguments": [
    {"id": "a1", "text": "<exact verbatim quote from the essay>"},
    {"id": "a2", "text": "<exact verbatim quote from the essay>"}
  ]
}

IMPORTANT: The "text" MUST be an exact, verbatim substring from the essay. \
Return ONLY the JSON object.\
"""

_PIPE_STEP2_SYSTEM = """\
You are an expert in argumentation theory. You are given a persuasive essay \
and a list of arguments that have been identified in it.

Your task is to determine the relations between these arguments. Each \
relation is directed (source -> target) and is either:
- "support": the source provides evidence/reasoning that supports the target.
- "attack": the source provides evidence/reasoning that undermines the target.

Only create a relation when there is a clear argumentative connection. Most \
argument pairs will have no relation.

Return your answer as a JSON object:
{
  "relations": [
    {"source": "a1", "target": "a2", "type": "support"}
  ]
}

Only include "support" and "attack" as relation types. Only reference \
argument IDs from the provided list. Return ONLY the JSON object.\
"""


def zs_pipe_step1(essay_text: str) -> List[Dict[str, str]]:
    """Pipeline step 1: identify arguments (zero-shot)."""
    return [
        {"role": "system", "content": _PIPE_STEP1_SYSTEM},
        {
            "role": "user",
            "content": f"Identify all arguments in this essay.\n\n"
            f"Essay:\n{essay_text.strip()}",
        },
    ]


def zs_pipe_step2(
    essay_text: str, arguments_json: str
) -> List[Dict[str, str]]:
    """Pipeline step 2: predict relations (zero-shot)."""
    return [
        {"role": "system", "content": _PIPE_STEP2_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Essay:\n{essay_text.strip()}\n\n"
                f"Identified arguments:\n{arguments_json}\n\n"
                f"Determine the support and attack relations between these "
                f"arguments."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Method 4: FS-Pipe (Few-shot Pipeline)
# ---------------------------------------------------------------------------


def _format_step1_example(essay_text: str, baf: BAF) -> str:
    import json
    args = []
    for i, arg in enumerate(baf.arguments, 1):
        args.append({"id": f"a{i}", "text": arg.text})
    return (
        f"Essay:\n{essay_text.strip()}\n\n"
        f"Arguments:\n{json.dumps({'arguments': args}, indent=2)}"
    )


def _format_step2_example(essay_text: str, baf: BAF) -> str:
    import json
    id_map = {}
    args = []
    for i, arg in enumerate(baf.arguments, 1):
        new_id = f"a{i}"
        id_map[arg.id] = new_id
        args.append({"id": new_id, "text": arg.text})

    rels = []
    for rel in baf.relations:
        src = id_map.get(rel.source)
        tgt = id_map.get(rel.target)
        if src and tgt:
            rels.append({"source": src, "target": tgt, "type": rel.type})

    return (
        f"Essay:\n{essay_text.strip()}\n\n"
        f"Arguments:\n{json.dumps({'arguments': args}, indent=2)}\n\n"
        f"Relations:\n{json.dumps({'relations': rels}, indent=2)}"
    )


def fs_pipe_step1(
    essay_text: str, examples: List[Dict]
) -> List[Dict[str, str]]:
    """Pipeline step 1 with few-shot examples."""
    examples_str = "\n\n---\n\n".join(
        _format_step1_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {"role": "system", "content": _PIPE_STEP1_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Here are some examples:\n\n{examples_str}\n\n---\n\n"
                f"Now identify all arguments in this essay.\n\n"
                f"Essay:\n{essay_text.strip()}"
            ),
        },
    ]


def fs_pipe_step2(
    essay_text: str, arguments_json: str, examples: List[Dict]
) -> List[Dict[str, str]]:
    """Pipeline step 2 with few-shot examples."""
    examples_str = "\n\n---\n\n".join(
        _format_step2_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {"role": "system", "content": _PIPE_STEP2_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Here are some examples:\n\n{examples_str}\n\n---\n\n"
                f"Now determine relations for this essay.\n\n"
                f"Essay:\n{essay_text.strip()}\n\n"
                f"Identified arguments:\n{arguments_json}\n\n"
                f"Determine the support and attack relations."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Method 5: FS-CoT-E2E (Few-shot + Chain-of-Thought End-to-End)
# ---------------------------------------------------------------------------


def fs_cot_e2e(
    essay_text: str,
    examples: List[Dict],
) -> List[Dict[str, str]]:
    """Few-shot + CoT end-to-end prompt."""
    examples_str = "\n\n---\n\n".join(
        _format_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {
            "role": "system",
            "content": f"{TASK_DESCRIPTION}\n\n{OUTPUT_SCHEMA_COT}",
        },
        {
            "role": "user",
            "content": (
                f"Here are some examples of essays and their BAFs:\n\n"
                f"{examples_str}\n\n"
                f"---\n\n"
                f"Now analyze the following essay. Think step by step about "
                f"its argumentative structure, then output the BAF as JSON.\n\n"
                f"Essay:\n{essay_text.strip()}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Gold-argument setting: relation prediction only
# ---------------------------------------------------------------------------


def gold_arg_relations(
    essay_text: str,
    gold_baf: BAF,
    examples: List[Dict] | None = None,
) -> List[Dict[str, str]]:
    """Prompt for the gold-argument evaluation setting.

    Provides gold argument spans; model predicts only relations.
    If examples are given, uses few-shot; otherwise zero-shot.
    """
    import json

    # Format gold arguments with sequential IDs
    id_map = {}
    args = []
    for i, arg in enumerate(gold_baf.arguments, 1):
        new_id = f"a{i}"
        id_map[arg.id] = new_id
        args.append({"id": new_id, "text": arg.text})

    args_json = json.dumps({"arguments": args}, indent=2)

    user_content = (
        f"Essay:\n{essay_text.strip()}\n\n"
        f"The following arguments have been identified:\n{args_json}\n\n"
        f"Determine which pairs of arguments have support or attack "
        f"relations."
    )

    if examples:
        examples_str = "\n\n---\n\n".join(
            _format_step2_example(ex["text"], ex["baf"]) for ex in examples
        )
        user_content = (
            f"Here are some examples:\n\n{examples_str}\n\n---\n\n"
            + user_content
        )

    return [
        {"role": "system", "content": _PIPE_STEP2_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Registry for easy access
# ---------------------------------------------------------------------------

METHODS = {
    "zs_e2e": "Zero-shot End-to-End",
    "fs_e2e": "Few-shot End-to-End",
    "zs_pipe": "Zero-shot Pipeline",
    "fs_pipe": "Few-shot Pipeline",
    "fs_cot_e2e": "Few-shot CoT End-to-End",
    "gold_zs": "Gold-argument Zero-shot",
    "gold_fs": "Gold-argument Few-shot",
}
