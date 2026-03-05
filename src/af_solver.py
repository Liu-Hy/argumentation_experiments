"""Argumentation semantics computation.

Implements:
- Grounded semantics (iterative fixpoint)
- Preferred semantics (backtracking enumeration with timeout)
- BAF coalition reduction (Cayrol & Lagasquie-Schiex 2005)
- DF-QuAD (Rago et al. 2016; bipolar extension by Baroni et al. 2019)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DungAF:
    """Dung Argumentation Framework (attack-only)."""

    arguments: Set[str] = field(default_factory=set)
    attacks: Set[Tuple[str, str]] = field(default_factory=set)  # (source, target)

    def attackers_of(self, arg: str) -> Set[str]:
        return {src for src, tgt in self.attacks if tgt == arg}


@dataclass
class BipolarAF:
    """Bipolar Argumentation Framework (attack + support)."""

    arguments: Set[str] = field(default_factory=set)
    attacks: Set[Tuple[str, str]] = field(default_factory=set)
    supports: Set[Tuple[str, str]] = field(default_factory=set)


@dataclass
class QBAF:
    """Quantitative (Bipolar) Argumentation Framework."""

    arguments: Set[str] = field(default_factory=set)
    base_strengths: Dict[str, float] = field(default_factory=dict)
    attacks: Set[Tuple[str, str]] = field(default_factory=set)
    supports: Set[Tuple[str, str]] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Grounded semantics
# ---------------------------------------------------------------------------


def grounded_extension(af: DungAF) -> Tuple[Set[str], Set[str], Set[str]]:
    """Compute the grounded extension via iterative fixpoint.

    Returns (IN, OUT, UNDECIDED).
    """
    in_set: Set[str] = set()
    out_set: Set[str] = set()

    # Initial: unattacked arguments are IN
    for arg in af.arguments:
        if not af.attackers_of(arg):
            in_set.add(arg)

    changed = True
    while changed:
        changed = False
        for arg in af.arguments:
            if arg in in_set or arg in out_set:
                continue
            attackers = af.attackers_of(arg)
            # Any attacker IN => arg is OUT
            if any(a in in_set for a in attackers):
                out_set.add(arg)
                changed = True
            # All attackers OUT => arg is IN
            elif all(a in out_set for a in attackers):
                in_set.add(arg)
                changed = True

    undecided = af.arguments - in_set - out_set
    return in_set, out_set, undecided


# ---------------------------------------------------------------------------
# Preferred semantics
# ---------------------------------------------------------------------------


def _is_conflict_free(attacks: Set[Tuple[str, str]], s: Set[str]) -> bool:
    for a in s:
        for b in s:
            if (a, b) in attacks:
                return False
    return True


def _is_admissible(af: DungAF, s: Set[str]) -> bool:
    """Check if s is admissible (conflict-free and defends all its members)."""
    if not _is_conflict_free(af.attacks, s):
        return False
    for a in s:
        for attacker in af.attackers_of(a):
            if attacker in s:
                continue  # handled by conflict-free check
            if not any((defender, attacker) in af.attacks for defender in s):
                return False
    return True


def preferred_extensions(
    af: DungAF, timeout_s: float = 30.0
) -> Optional[List[Set[str]]]:
    """Enumerate all preferred extensions (maximal admissible sets).

    Returns None if timeout exceeded.
    """
    start_time = time.time()
    args_list = sorted(af.arguments)
    n = len(args_list)

    if n == 0:
        return [set()]

    admissible: List[frozenset] = []

    def search(idx: int, current: Set[str]):
        if time.time() - start_time > timeout_s:
            raise TimeoutError()

        if idx == n:
            if _is_admissible(af, current):
                admissible.append(frozenset(current))
            return

        arg = args_list[idx]

        # Branch: include arg (only if maintains conflict-freeness)
        can_include = all(
            (arg, c) not in af.attacks and (c, arg) not in af.attacks
            for c in current
        )
        if can_include:
            current.add(arg)
            search(idx + 1, current)
            current.remove(arg)

        # Branch: exclude arg
        search(idx + 1, current)

    try:
        search(0, set())
    except TimeoutError:
        return None

    # Filter to maximal admissible sets
    maximal = []
    for s in admissible:
        if not any(s < other for other in admissible):
            maximal.append(set(s))

    # Deduplicate
    seen: Set[frozenset] = set()
    result = []
    for s in maximal:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            result.append(s)

    return result if result else [set()]


def credulous_acceptance(extensions: List[Set[str]]) -> Set[str]:
    """Arguments in at least one preferred extension."""
    result: Set[str] = set()
    for ext in extensions:
        result |= ext
    return result


def skeptical_acceptance(extensions: List[Set[str]]) -> Set[str]:
    """Arguments in every preferred extension."""
    if not extensions:
        return set()
    result = set(extensions[0])
    for ext in extensions[1:]:
        result &= ext
    return result


# ---------------------------------------------------------------------------
# BAF coalition reduction
# ---------------------------------------------------------------------------


def baf_to_dung(baf: BipolarAF) -> DungAF:
    """Reduce BAF to Dung AF via Cayrol & Lagasquie-Schiex coalition reduction.

    For each support (a, b) and attack (c, a): add secondary attack (c, b).
    """
    attacks = set(baf.attacks)

    for sup_src, sup_tgt in baf.supports:
        for att_src, att_tgt in baf.attacks:
            if att_tgt == sup_src:
                attacks.add((att_src, sup_tgt))

    return DungAF(arguments=set(baf.arguments), attacks=attacks)


# ---------------------------------------------------------------------------
# DF-QuAD
# ---------------------------------------------------------------------------


def _product(values) -> float:
    """Product of an iterable of floats."""
    result = 1.0
    for v in values:
        result *= v
    return result


def df_quad(
    qbaf: QBAF, epsilon: float = 0.001, max_iterations: int = 100
) -> Dict[str, float]:
    """Compute final argument strengths using DF-QuAD.

    Handles both attack-only and bipolar QBAFs.
    Returns dict: argument ID -> final strength.
    """
    strengths = dict(qbaf.base_strengths)
    has_support = len(qbaf.supports) > 0

    for _ in range(max_iterations):
        new_strengths = {}
        max_change = 0.0

        for arg in qbaf.arguments:
            s0 = qbaf.base_strengths[arg]

            # Aggregate attackers
            attacker_strengths = [
                strengths[src] for src, tgt in qbaf.attacks if tgt == arg
            ]
            att_agg = (
                1.0 - _product(1.0 - s for s in attacker_strengths)
                if attacker_strengths
                else 0.0
            )

            if has_support:
                # Aggregate supporters
                supporter_strengths = [
                    strengths[src] for src, tgt in qbaf.supports if tgt == arg
                ]
                sup_agg = (
                    1.0 - _product(1.0 - s for s in supporter_strengths)
                    if supporter_strengths
                    else 0.0
                )

                # Combined formula (Baroni et al. 2019)
                if sup_agg >= att_agg:
                    new_s = s0 + (1.0 - s0) * (sup_agg - att_agg)
                else:
                    new_s = s0 * (1.0 - (att_agg - sup_agg))
            else:
                # Attack-only formula (Rago et al. 2016)
                if att_agg <= 0:
                    new_s = s0
                else:
                    new_s = s0 * (1.0 - att_agg)

            new_strengths[arg] = max(0.0, min(1.0, new_s))
            max_change = max(max_change, abs(new_strengths[arg] - strengths[arg]))

        strengths = new_strengths

        if max_change < epsilon:
            break

    return strengths
