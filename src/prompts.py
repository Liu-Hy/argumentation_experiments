"""Prompt templates for BAF extraction experiments.

Five core methods:
  1. ZS-E2E    -- Zero-shot end-to-end
  2. FS-E2E    -- Few-shot (3-shot) end-to-end
  3. ZS-Pipe   -- Zero-shot two-step pipeline
  4. FS-Pipe   -- Few-shot two-step pipeline
  5. FS-CoT-E2E -- Few-shot + chain-of-thought end-to-end

Plus gold-argument variants (relation prediction only) and pairwise
relation classification variants.

Prompt variants:
  - "default"  -- Standard prompts (used for main results)
  - "enhanced" -- More specific guidance (granularity, direction, attacks)
  - "minimal"  -- Stripped-down prompts (bare task definition + JSON schema)
"""

from __future__ import annotations

from typing import Dict, List

from .baf import BAF

# ---------------------------------------------------------------------------
# Shared components — "default" variant
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
1. Read through the essay and identify all argumentative text spans — any \
claim, premise, thesis, evidence, or counter-argument. They are all treated \
uniformly as "arguments". Note each as a verbatim quote from the essay.
2. For each pair of arguments, consider whether one provides evidence or \
reasoning that supports or undermines the other. Only note relations where \
there is a clear argumentative connection.
3. Pay special attention to opposing viewpoints, counterpoints, and \
concessions — these often form attack relations.
4. Compile your analysis into the JSON format below.

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

# ---------------------------------------------------------------------------
# Pipeline system prompts — "default" variant
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


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = ["default", "enhanced", "minimal"]

# --- Enhanced variant: more specific guidance ---

_TASK_DESC_ENHANCED = """\
You are an expert in argumentation theory. Your task is to analyze a \
persuasive essay and extract its Bipolar Argumentation Framework (BAF).

A BAF consists of:
- **Arguments**: text spans from the essay that express argumentative \
points (claims, premises, main theses — all treated uniformly as \
"arguments"). Each argument should be a clause-length or sentence-length \
span expressing a single coherent point. A typical essay contains between \
5 and 20 arguments.
- **Relations**: directed edges between arguments. The source argument \
provides evidence or reasoning about the target argument. Each relation is \
either:
  - "support": the source provides evidence or reasoning that supports the \
target (e.g., a premise backing a claim).
  - "attack": the source provides evidence or reasoning that undermines or \
opposes the target (e.g., a counter-argument against a claim, a concession \
that weakens a point).

Pay special attention to opposing viewpoints, counterpoints, and \
concessions — these form attack relations, which are less common but \
important to capture.\
"""

_OUT_SCHEMA_ENHANCED = """\
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
- Not all argument pairs have relations. Only create a relation when there \
is a clear argumentative connection.
- Return ONLY the JSON object, no other text.\
"""

_OUT_SCHEMA_COT_ENHANCED = """\
First, reason step by step about the essay's argumentative structure:
1. Read through the essay and identify all argumentative text spans — any \
claim, premise, thesis, evidence, or counter-argument. They are all treated \
uniformly as "arguments". Each should be a clause-length or sentence-length \
span expressing one point. Note each as a verbatim quote.
2. For each pair of arguments, consider whether one provides evidence or \
reasoning that supports or undermines the other. Only note relations where \
there is a clear argumentative connection.
3. Pay special attention to opposing viewpoints, counterpoints, and \
concessions — these often form attack relations, which are less common but \
important to capture.
4. Compile your arguments and relations into the JSON format below.

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

_PIPE1_SYSTEM_ENHANCED = """\
You are an expert in argumentation theory. Your task is to identify all \
arguments in a persuasive essay.

An argument is any clause-length or sentence-length text span that expresses \
an argumentative point: a claim, a piece of evidence, a thesis statement, \
a counter-argument, a concession, etc. A typical essay contains between \
5 and 20 arguments.

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

_PIPE2_SYSTEM_ENHANCED = """\
You are an expert in argumentation theory. You are given a persuasive essay \
and a list of arguments that have been identified in it.

Your task is to determine the relations between these arguments. Each \
relation is directed (source -> target):
- "support": the source provides evidence/reasoning that supports the target \
(e.g., a premise backing a claim).
- "attack": the source provides evidence/reasoning that undermines the \
target (e.g., a counter-argument opposing a claim, a concession weakening \
a point).

Pay special attention to opposing viewpoints and concessions — these form \
attack relations. Only create a relation when there is a clear argumentative \
connection. Most argument pairs will have no relation.

Return your answer as a JSON object:
{
  "relations": [
    {"source": "a1", "target": "a2", "type": "support"}
  ]
}

Only include "support" and "attack" as relation types. Only reference \
argument IDs from the provided list. Return ONLY the JSON object.\
"""

# --- Minimal variant: stripped-down prompts ---

_TASK_DESC_MINIMAL = """\
Extract the Bipolar Argumentation Framework (BAF) from the following essay. \
A BAF consists of arguments (verbatim text spans from the essay) and \
directed relations between them (either "support" or "attack").\
"""

_OUT_SCHEMA_MINIMAL = """\
Return your answer as JSON:
{
  "arguments": [{"id": "a1", "text": "<verbatim quote from essay>"}, ...],
  "relations": [{"source": "a1", "target": "a2", "type": "support"}, ...]
}

Text must be exact verbatim quotes from the essay. Only use "support" and \
"attack" as relation types. Return ONLY the JSON.\
"""

_OUT_SCHEMA_COT_MINIMAL = """\
Think step by step about the essay's argumentative structure, then return \
your answer as JSON:
{
  "arguments": [{"id": "a1", "text": "<verbatim quote from essay>"}, ...],
  "relations": [{"source": "a1", "target": "a2", "type": "support"}, ...]
}

Text must be exact verbatim quotes. Only use "support" and "attack" types.\
"""

_PIPE1_SYSTEM_MINIMAL = """\
Identify all arguments in the following persuasive essay. An argument is \
any text span expressing an argumentative point.

Return as JSON:
{
  "arguments": [{"id": "a1", "text": "<verbatim quote from essay>"}, ...]
}

Text must be exact verbatim quotes. Return ONLY the JSON.\
"""

_PIPE2_SYSTEM_MINIMAL = """\
Given a persuasive essay and identified arguments, determine the support \
and attack relations between them.

Return as JSON:
{
  "relations": [{"source": "a1", "target": "a2", "type": "support"}, ...]
}

Only use "support" and "attack" types. Only reference provided argument \
IDs. Return ONLY the JSON.\
"""

# --- Variant lookup dicts ---

_TASK_DESCS = {
    "default": TASK_DESCRIPTION,
    "enhanced": _TASK_DESC_ENHANCED,
    "minimal": _TASK_DESC_MINIMAL,
}

_OUT_SCHEMAS = {
    "default": OUTPUT_SCHEMA,
    "enhanced": _OUT_SCHEMA_ENHANCED,
    "minimal": _OUT_SCHEMA_MINIMAL,
}

_OUT_SCHEMAS_COT = {
    "default": OUTPUT_SCHEMA_COT,
    "enhanced": _OUT_SCHEMA_COT_ENHANCED,
    "minimal": _OUT_SCHEMA_COT_MINIMAL,
}

_PIPE1_SYSTEMS = {
    "default": _PIPE_STEP1_SYSTEM,
    "enhanced": _PIPE1_SYSTEM_ENHANCED,
    "minimal": _PIPE1_SYSTEM_MINIMAL,
}

_PIPE2_SYSTEMS = {
    "default": _PIPE_STEP2_SYSTEM,
    "enhanced": _PIPE2_SYSTEM_ENHANCED,
    "minimal": _PIPE2_SYSTEM_MINIMAL,
}


# ---------------------------------------------------------------------------
# Example formatting helpers (variant-independent — these format BAF data)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Method 1: ZS-E2E (Zero-shot End-to-End)
# ---------------------------------------------------------------------------


def zs_e2e(
    essay_text: str, variant: str = "default"
) -> List[Dict[str, str]]:
    """Zero-shot end-to-end prompt. Returns list of messages."""
    return [
        {
            "role": "system",
            "content": f"{_TASK_DESCS[variant]}\n\n{_OUT_SCHEMAS[variant]}",
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
    variant: str = "default",
) -> List[Dict[str, str]]:
    """Few-shot end-to-end prompt."""
    examples_str = "\n\n---\n\n".join(
        _format_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {
            "role": "system",
            "content": f"{_TASK_DESCS[variant]}\n\n{_OUT_SCHEMAS[variant]}",
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


def zs_pipe_step1(
    essay_text: str, variant: str = "default"
) -> List[Dict[str, str]]:
    """Pipeline step 1: identify arguments (zero-shot)."""
    return [
        {"role": "system", "content": _PIPE1_SYSTEMS[variant]},
        {
            "role": "user",
            "content": f"Identify all arguments in this essay.\n\n"
            f"Essay:\n{essay_text.strip()}",
        },
    ]


def zs_pipe_step2(
    essay_text: str, arguments_json: str, variant: str = "default"
) -> List[Dict[str, str]]:
    """Pipeline step 2: predict relations (zero-shot)."""
    return [
        {"role": "system", "content": _PIPE2_SYSTEMS[variant]},
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


def fs_pipe_step1(
    essay_text: str, examples: List[Dict], variant: str = "default"
) -> List[Dict[str, str]]:
    """Pipeline step 1 with few-shot examples."""
    examples_str = "\n\n---\n\n".join(
        _format_step1_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {"role": "system", "content": _PIPE1_SYSTEMS[variant]},
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
    essay_text: str, arguments_json: str, examples: List[Dict],
    variant: str = "default",
) -> List[Dict[str, str]]:
    """Pipeline step 2 with few-shot examples."""
    examples_str = "\n\n---\n\n".join(
        _format_step2_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {"role": "system", "content": _PIPE2_SYSTEMS[variant]},
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
    variant: str = "default",
) -> List[Dict[str, str]]:
    """Few-shot + CoT end-to-end prompt."""
    examples_str = "\n\n---\n\n".join(
        _format_example(ex["text"], ex["baf"]) for ex in examples
    )
    return [
        {
            "role": "system",
            "content": f"{_TASK_DESCS[variant]}\n\n{_OUT_SCHEMAS_COT[variant]}",
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
    variant: str = "default",
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
        {"role": "system", "content": _PIPE2_SYSTEMS[variant]},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Pairwise relation classification
# ---------------------------------------------------------------------------

_PAIRWISE_SYSTEM = """\
You are an expert in argumentation theory. Given a persuasive essay and two \
specific arguments extracted from it, determine if there is a direct \
argumentative relation between them.

A "support" relation means one argument provides evidence or reasoning that \
backs or strengthens the other.
An "attack" relation means one argument provides evidence or reasoning that \
undermines, opposes, or weakens the other.

Return EXACTLY one of these JSON responses:
- {"source": "arg1", "target": "arg2", "type": "support"} — Argument 1 supports Argument 2
- {"source": "arg2", "target": "arg1", "type": "support"} — Argument 2 supports Argument 1
- {"source": "arg1", "target": "arg2", "type": "attack"} — Argument 1 attacks Argument 2
- {"source": "arg2", "target": "arg1", "type": "attack"} — Argument 2 attacks Argument 1
- {"relation": "none"} — no direct argumentative relation

IMPORTANT: Most argument pairs have NO direct relation. Only identify a \
relation when there is a clear, direct argumentative connection between \
the two specific arguments. Return ONLY the JSON.\
"""


def extract_pairwise_examples(examples: List[Dict]) -> List[Dict]:
    """Extract representative argument pairs from few-shot BAF examples.

    Returns a list of dicts with keys:
        essay_text, arg1_text, arg2_text, answer

    Tries to find one support, one attack, and one no-relation pair,
    preferably from one essay that has both relation types.
    """
    if not examples:
        return []

    # Prefer an essay with both support and attack relations
    best_essay = None
    for ex in examples:
        has_attacks = any(r.type == "attack" for r in ex["baf"].relations)
        has_supports = any(r.type == "support" for r in ex["baf"].relations)
        if has_attacks and has_supports:
            best_essay = ex
            break
    if best_essay is None:
        best_essay = examples[0]

    baf = best_essay["baf"]
    text = best_essay["text"]
    arg_map = {a.id: a for a in baf.arguments}

    # Build relation lookup for no-relation search
    rel_set = set()
    for rel in baf.relations:
        rel_set.add((rel.source, rel.target))

    pair_examples = []

    # One support pair
    for rel in baf.relations:
        if rel.type == "support" and rel.source in arg_map and rel.target in arg_map:
            pair_examples.append({
                "essay_text": text,
                "arg1_text": arg_map[rel.source].text,
                "arg2_text": arg_map[rel.target].text,
                "answer": {"source": "arg1", "target": "arg2", "type": "support"},
            })
            break

    # One attack pair
    for rel in baf.relations:
        if rel.type == "attack" and rel.source in arg_map and rel.target in arg_map:
            pair_examples.append({
                "essay_text": text,
                "arg1_text": arg_map[rel.source].text,
                "arg2_text": arg_map[rel.target].text,
                "answer": {"source": "arg1", "target": "arg2", "type": "attack"},
            })
            break

    # One no-relation pair
    args_list = list(baf.arguments)
    found_none = False
    for i in range(len(args_list)):
        if found_none:
            break
        for j in range(i + 1, len(args_list)):
            a, b = args_list[i], args_list[j]
            if (a.id, b.id) not in rel_set and (b.id, a.id) not in rel_set:
                pair_examples.append({
                    "essay_text": text,
                    "arg1_text": a.text,
                    "arg2_text": b.text,
                    "answer": {"relation": "none"},
                })
                found_none = True
                break

    return pair_examples


def _format_pairwise_block(pair_examples: List[Dict]) -> str:
    """Format pairwise examples into a prompt block.

    All examples are assumed to come from the same essay (shown once).
    """
    import json as _json

    if not pair_examples:
        return ""

    essay_text = pair_examples[0]["essay_text"]
    parts = [f"Essay:\n{essay_text.strip()}\n"]

    for i, ex in enumerate(pair_examples, 1):
        answer_str = _json.dumps(ex["answer"])
        parts.append(
            f'Pair {i}:\n'
            f'Argument 1 (arg1): "{ex["arg1_text"]}"\n'
            f'Argument 2 (arg2): "{ex["arg2_text"]}"\n'
            f'Answer: {answer_str}'
        )

    return "\n\n".join(parts)


def pairwise_classify(
    essay_text: str,
    arg1_text: str,
    arg2_text: str,
    pair_examples: List[Dict] | None = None,
) -> List[Dict[str, str]]:
    """Build prompt for classifying the relation between two arguments.

    Uses placeholder IDs 'arg1' and 'arg2'; the caller remaps to real IDs.
    If pair_examples is provided, includes few-shot demonstration.
    """
    user_parts = []

    if pair_examples:
        examples_block = _format_pairwise_block(pair_examples)
        user_parts.append(
            f"Here are examples of pairwise relation classification:\n\n"
            f"{examples_block}\n\n"
            f"---\n\n"
        )

    user_parts.append(
        f"Now classify the following pair:\n\n"
        f"Essay:\n{essay_text.strip()}\n\n"
        f'Argument 1 (arg1): "{arg1_text}"\n'
        f'Argument 2 (arg2): "{arg2_text}"\n\n'
        f"Classify the argumentative relation between these two arguments."
    )

    return [
        {"role": "system", "content": _PAIRWISE_SYSTEM},
        {"role": "user", "content": "".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Graph-context pairwise classification
# ---------------------------------------------------------------------------


def extract_graph_example(examples: List[Dict]) -> Dict | None:
    """Extract a full training essay BAF as graph context for pairwise prompts.

    Returns a dict with:
        essay_text, arguments (list of (seq_id, text)),
        relations (list of (src_id, tgt_id, type)),
        n_args, n_pairs, n_rels

    Prefers an essay with both support and attack relations.
    Returns None if examples is empty.
    """
    if not examples:
        return None

    # Prefer an essay with both support and attack relations
    best_essay = None
    for ex in examples:
        has_attacks = any(r.type == "attack" for r in ex["baf"].relations)
        has_supports = any(r.type == "support" for r in ex["baf"].relations)
        if has_attacks and has_supports:
            best_essay = ex
            break
    if best_essay is None:
        best_essay = examples[0]

    baf = best_essay["baf"]

    # Build sequential ID mapping
    id_map = {}
    args = []
    for i, arg in enumerate(baf.arguments, 1):
        seq_id = f"a{i}"
        id_map[arg.id] = seq_id
        args.append((seq_id, arg.text))

    # Map relations
    rels = []
    for rel in baf.relations:
        src = id_map.get(rel.source)
        tgt = id_map.get(rel.target)
        if src and tgt:
            rels.append((src, tgt, rel.type))

    n_args = len(args)
    n_pairs = n_args * (n_args - 1) // 2

    return {
        "essay_text": best_essay["text"],
        "arguments": args,
        "relations": rels,
        "n_args": n_args,
        "n_pairs": n_pairs,
        "n_rels": len(rels),
    }


def _format_graph_block(graph_example: Dict) -> str:
    """Format a full training BAF as a prompt block."""
    parts = [f"Essay:\n{graph_example['essay_text'].strip()}\n"]

    # List all arguments
    parts.append(f"Arguments identified ({graph_example['n_args']} total):")
    for seq_id, text in graph_example["arguments"]:
        parts.append(f'  {seq_id}: "{text}"')

    # List all relations with sparsity context
    n_rels = graph_example["n_rels"]
    n_pairs = graph_example["n_pairs"]
    parts.append(f"\nRelations ({n_rels} out of {n_pairs} possible pairs):")
    for src, tgt, rtype in graph_example["relations"]:
        parts.append(f"  {src} -> {tgt}: {rtype}")

    no_rel = n_pairs - n_rels
    parts.append(
        f"\nThe remaining {no_rel} argument pairs have NO direct relation. "
        f"In argumentative essays, only a small fraction of argument pairs "
        f"are directly related."
    )

    return "\n".join(parts)


def pairwise_classify_with_graph(
    essay_text: str,
    arg1_text: str,
    arg2_text: str,
    graph_example: Dict,
) -> List[Dict[str, str]]:
    """Build pairwise classification prompt with full-graph training context.

    Shows the complete annotation of a training essay (all arguments and
    all relations) so the model can calibrate its predictions based on
    real relation density (~8% of pairs).
    """
    graph_block = _format_graph_block(graph_example)

    user_content = (
        f"Here is a complete annotation of a training essay, showing all "
        f"arguments and their relations:\n\n"
        f"{graph_block}\n\n"
        f"---\n\n"
        f"Now classify the relation between the following two arguments "
        f"from a new essay:\n\n"
        f"Essay:\n{essay_text.strip()}\n\n"
        f'Argument 1 (arg1): "{arg1_text}"\n'
        f'Argument 2 (arg2): "{arg2_text}"\n\n'
        f"Classify the argumentative relation between these two arguments."
    )

    return [
        {"role": "system", "content": _PAIRWISE_SYSTEM},
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
    "gold_fs_pairwise": "Gold-argument Few-shot Pairwise",
    "fs_pipe_pairwise": "Few-shot Pipeline Pairwise",
    "gold_fs_pw_graph": "Gold-arg Pairwise (Graph Context)",
    "fs_pipe_pw_graph": "Pipeline Pairwise (Graph Context)",
}
