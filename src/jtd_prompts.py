"""Prompt templates for JTD argumentation framework experiments.

Baselines:
  B0: Direct LLM prediction (no AF)
  B1: AF-guided LLM prediction (AF as structured CoT)
  B2/B3: Standard Dung AF construction (attacks only)
  B4: Standard BAF construction (attacks + supports)
  B5: Quantitative AF construction (attacks + strengths)
  B6: Quantitative BAF construction (attacks + supports + strengths)
  BH: Heuristic QBAF (no LLM -- constructed deterministically)
"""

from __future__ import annotations

from typing import Dict, List

from .jtd_data_loader import TortCase, format_case_for_prompt

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

JTD_METHODS = {
    "B0": "Direct LLM Prediction",
    "B1": "AF-Guided LLM Prediction",
    "B2": "Dung AF + Grounded Semantics",
    "B3": "Dung AF + Preferred Semantics",
    "B4": "BAF + Grounded Semantics (Coalition Reduction)",
    "B5": "Quantitative AF + DF-QuAD",
    "B6": "Quantitative BAF + DF-QuAD",
    "BH": "Heuristic QBAF + DF-QuAD",
}

# Methods that require LLM calls for AF construction
LLM_AF_METHODS = {"B1", "B2", "B3", "B4", "B5", "B6"}

# Methods that require LLM calls for direct prediction
LLM_DIRECT_METHODS = {"B0"}

# Methods that use the same AF construction (share LLM call)
SHARED_AF = {"B2": "dung", "B3": "dung"}

# ---------------------------------------------------------------------------
# B0: Direct LLM Prediction
# ---------------------------------------------------------------------------

_B0_SYSTEM = """\
You are an expert in Japanese civil law, specifically tort law under Article 709 \
of the Japanese Civil Code.

You are given a tort case with:
- Undisputed facts: facts agreed upon by both parties
- Plaintiff claims: arguments by the plaintiff
- Defendant claims: arguments by the defendant

Your task is to predict:
1. For each plaintiff claim (p0, p1, ...), whether the court accepts it (true/false)
2. For each defendant claim (d0, d1, ...), whether the court accepts it (true/false)
3. Whether the overall tort claim is affirmed (true/false)

Return your answer as JSON:
{
  "plaintiff_claims": {"p0": true, "p1": false},
  "defendant_claims": {"d0": false, "d1": true},
  "tort_decision": true
}

Include ALL claims with their predictions. Return ONLY the JSON.\
"""


def b0_direct(case: TortCase) -> List[Dict[str, str]]:
    """B0: Direct LLM prediction prompt."""
    case_text = format_case_for_prompt(case, include_undisputed=True)
    return [
        {"role": "system", "content": _B0_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Analyze the following tort case and predict the court's decisions.\n\n"
                f"{case_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# B1: AF-Guided LLM Prediction
# ---------------------------------------------------------------------------

_B1_STEP2_SYSTEM = """\
You are an expert in Japanese civil law, specifically tort law under Article 709 \
of the Japanese Civil Code.

You are given a tort case and an argumentation framework that was constructed \
from it, showing attack relations between the plaintiff and defendant claims. \
Use the argumentation framework as a structured guide for your reasoning. \
Consider which claims attack which, and how the structure of attacks determines \
which arguments ultimately prevail.

Predict:
1. For each plaintiff claim (p0, p1, ...), whether the court accepts it (true/false)
2. For each defendant claim (d0, d1, ...), whether the court accepts it (true/false)
3. Whether the overall tort claim is affirmed (true/false)

Return your answer as JSON:
{
  "plaintiff_claims": {"p0": true, "p1": false},
  "defendant_claims": {"d0": false, "d1": true},
  "tort_decision": true
}

Include ALL claims. Return ONLY the JSON.\
"""


def b1_step1(case: TortCase) -> List[Dict[str, str]]:
    """B1 Step 1: Construct Dung AF (same prompt as B2/B3)."""
    return _dung_af_prompt(case)


def b1_step2(case: TortCase, af_json: str) -> List[Dict[str, str]]:
    """B1 Step 2: Predict with AF as context."""
    case_text = format_case_for_prompt(case, include_undisputed=True)
    return [
        {"role": "system", "content": _B1_STEP2_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{case_text}\n\n"
                f"Argumentation Framework (attack relations):\n{af_json}\n\n"
                f"Using this framework, predict the court's decisions."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# B2/B3: Dung AF Construction (attack-only)
# ---------------------------------------------------------------------------

_DUNG_AF_SYSTEM = """\
You are an expert in argumentation theory and Japanese civil law. Your task is \
to construct a Dung Argumentation Framework from a tort case by identifying \
attack relations between claims.

Attack definition: Argument A attacks argument B if A provides a reason, \
evidence, or legal basis that undermines, contradicts, or rebuts the conclusion \
of B. Attacks are typically cross-party (defendant claims rebutting plaintiff \
claims, and vice versa), but may also occur within the same party if claims \
are logically inconsistent.

Undisputed facts are provided as context for identifying attacks but are NOT \
included as arguments in the framework.

Return your answer as JSON:
{
  "attacks": [
    {"source": "p0", "target": "d2"},
    {"source": "d1", "target": "p0"}
  ]
}

Only use argument IDs from the provided claims (p0, p1, ..., d0, d1, ...). \
Return ONLY the JSON.\
"""


def _dung_af_prompt(case: TortCase) -> List[Dict[str, str]]:
    """Shared Dung AF construction prompt for B1/B2/B3."""
    case_text = format_case_for_prompt(case, include_undisputed=True)
    return [
        {"role": "system", "content": _DUNG_AF_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Identify the attack relations between claims in this tort case.\n\n"
                f"{case_text}"
            ),
        },
    ]


def b2_construct(case: TortCase) -> List[Dict[str, str]]:
    """B2: Dung AF construction prompt."""
    return _dung_af_prompt(case)


def b3_construct(case: TortCase) -> List[Dict[str, str]]:
    """B3: Same AF construction as B2 (different semantics applied later)."""
    return _dung_af_prompt(case)


# ---------------------------------------------------------------------------
# B4: BAF Construction (attack + support)
# ---------------------------------------------------------------------------

_BAF_SYSTEM = """\
You are an expert in argumentation theory and Japanese civil law. Your task is \
to construct a Bipolar Argumentation Framework from a tort case by identifying \
both attack and support relations.

Attack: Argument A attacks argument B if A provides a reason, evidence, or \
legal basis that undermines, contradicts, or rebuts the conclusion of B. \
Attacks are typically cross-party (defendant claims rebutting plaintiff claims, \
and vice versa), but may also occur within the same party if claims are \
logically inconsistent.

Support: Argument A supports argument B if A provides evidence, reasoning, or \
factual basis that strengthens or substantiates B. Support commonly occurs: \
(1) from undisputed facts to claims they substantiate, and (2) between claims \
within the same party that reinforce each other.

Constraints:
- Undisputed facts (u0, u1, ...) are axioms. They may ONLY be sources of \
support relations. No argument may attack an undisputed fact.
- All listed arguments (u*, p*, d*) are valid IDs.

Return your answer as JSON:
{
  "attacks": [
    {"source": "d1", "target": "p0"}
  ],
  "supports": [
    {"source": "u0", "target": "p1"},
    {"source": "p0", "target": "p2"}
  ]
}

Return ONLY the JSON.\
"""


def b4_construct(case: TortCase) -> List[Dict[str, str]]:
    """B4: BAF construction prompt (includes undisputed facts as arguments)."""
    case_text = format_case_for_prompt(case, include_undisputed=True)
    return [
        {"role": "system", "content": _BAF_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Identify the attack and support relations in this tort case.\n\n"
                f"{case_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# B5: Quantitative AF (attack-only + strengths)
# ---------------------------------------------------------------------------

_QUAF_SYSTEM = """\
You are an expert in argumentation theory and Japanese civil law. Your task is \
to construct a Quantitative Argumentation Framework from a tort case.

You must:
1. Assign a base strength to each claim: a value in [0, 1] reflecting how \
convincing the argument is in isolation (before considering interactions with \
other arguments). Higher values indicate stronger, more well-founded arguments.
2. Identify attack relations between claims.

Attack definition: Argument A attacks argument B if A provides a reason, \
evidence, or legal basis that undermines, contradicts, or rebuts the conclusion \
of B.

Undisputed facts are provided as context but are NOT included as arguments.

Return your answer as JSON:
{
  "argument_strengths": {
    "p0": 0.8, "p1": 0.6, "d0": 0.7, "d1": 0.5
  },
  "attacks": [
    {"source": "d0", "target": "p0"}
  ]
}

Include strengths for ALL plaintiff and defendant claims. \
Only use claim IDs (p0, p1, ..., d0, d1, ...). Return ONLY the JSON.\
"""


def b5_construct(case: TortCase) -> List[Dict[str, str]]:
    """B5: Quantitative AF construction prompt."""
    case_text = format_case_for_prompt(case, include_undisputed=True)
    return [
        {"role": "system", "content": _QUAF_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Assign base strengths and identify attack relations.\n\n"
                f"{case_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# B6: Quantitative BAF (attack + support + strengths)
# ---------------------------------------------------------------------------

_QBAF_SYSTEM = """\
You are an expert in argumentation theory and Japanese civil law. Your task is \
to construct a Quantitative Bipolar Argumentation Framework from a tort case.

You must:
1. Assign a base strength to each argument: a value in [0, 1].
   - Undisputed facts MUST have strength 1.0 (they are axioms).
   - Claims should be assigned strengths reflecting their convincingness in \
isolation (before considering interactions).
2. Identify both attack and support relations.

Attack: Argument A attacks argument B if A provides a reason, evidence, or \
legal basis that undermines, contradicts, or rebuts the conclusion of B.

Support: Argument A supports argument B if A provides evidence, reasoning, or \
factual basis that strengthens or substantiates B. Support commonly occurs: \
(1) from undisputed facts to claims they substantiate, and (2) between claims \
within the same party that reinforce each other.

Constraints:
- Undisputed facts (u0, u1, ...) are axioms with strength 1.0. No argument \
may attack them. They may only be sources of support.

Return your answer as JSON:
{
  "argument_strengths": {
    "u0": 1.0, "p0": 0.8, "p1": 0.6, "d0": 0.7
  },
  "attacks": [
    {"source": "d0", "target": "p0"}
  ],
  "supports": [
    {"source": "u0", "target": "p1"}
  ]
}

Include strengths for ALL arguments (u*, p*, d*). Return ONLY the JSON.\
"""


def b6_construct(case: TortCase) -> List[Dict[str, str]]:
    """B6: Quantitative BAF construction prompt."""
    case_text = format_case_for_prompt(case, include_undisputed=True)
    return [
        {"role": "system", "content": _QBAF_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Assign base strengths and identify attack and support relations.\n\n"
                f"{case_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Prompt dispatch
# ---------------------------------------------------------------------------


def build_af_prompt(
    method: str, case: TortCase, af_json: str | None = None
) -> List[Dict[str, str]]:
    """Build the LLM prompt for a given method and case.

    For B1, af_json must be provided for step 2. If af_json is None, returns
    step 1 prompt.
    """
    if method == "B0":
        return b0_direct(case)
    elif method == "B1":
        if af_json is None:
            return b1_step1(case)
        else:
            return b1_step2(case, af_json)
    elif method in ("B2", "B3"):
        return b2_construct(case)
    elif method == "B4":
        return b4_construct(case)
    elif method == "B5":
        return b5_construct(case)
    elif method == "B6":
        return b6_construct(case)
    else:
        raise ValueError(f"Unknown method: {method}")
