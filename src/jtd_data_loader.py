"""Load the Japanese Tort-Case Dataset (JTD) and prepare CV splits.

Dataset: 6,508 annotated civil tort cases under Japanese Civil Code Article 709.
Each case has undisputed facts, plaintiff/defendant claims with acceptance labels,
and a court decision on whether the tort is affirmed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Claim:
    """A plaintiff or defendant claim."""

    id: str
    description: str
    is_accepted: bool


@dataclass
class UndisputedFact:
    """A fact agreed upon by both parties."""

    id: str
    description: str


@dataclass
class TortCase:
    """A single tort case."""

    tort_id: str
    undisputed_facts: List[UndisputedFact]
    plaintiff_claims: List[Claim]
    defendant_claims: List[Claim]
    court_decision: bool  # True = tort affirmed

    @property
    def n_claims(self) -> int:
        return len(self.plaintiff_claims) + len(self.defendant_claims)


def load_jtd(filepath: str) -> List[TortCase]:
    """Load JTD from a JSONL file.

    Returns list of TortCase objects.
    """
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            undisputed = [
                UndisputedFact(id=u["id"], description=u["description"])
                for u in obj.get("undisputed_facts", [])
            ]
            plaintiff = [
                Claim(
                    id=c["id"],
                    description=c["description"],
                    is_accepted=c["is_accepted"],
                )
                for c in obj.get("plaintiff_claims", [])
            ]
            defendant = [
                Claim(
                    id=c["id"],
                    description=c["description"],
                    is_accepted=c["is_accepted"],
                )
                for c in obj.get("defendant_claims", [])
            ]
            cases.append(
                TortCase(
                    tort_id=str(obj["tort_id"]),
                    undisputed_facts=undisputed,
                    plaintiff_claims=plaintiff,
                    defendant_claims=defendant,
                    court_decision=obj["court_decision"],
                )
            )
    return cases


def stratified_kfold(
    cases: List[TortCase], n_folds: int = 5, seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Create stratified k-fold splits, stratified on court_decision.

    Returns list of (train_indices, test_indices) tuples.
    """
    rng = random.Random(seed)

    pos_indices = [i for i, c in enumerate(cases) if c.court_decision]
    neg_indices = [i for i, c in enumerate(cases) if not c.court_decision]

    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    # Round-robin assignment to folds
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for i, idx in enumerate(pos_indices):
        folds[i % n_folds].append(idx)
    for i, idx in enumerate(neg_indices):
        folds[i % n_folds].append(idx)

    splits = []
    for fold_idx in range(n_folds):
        test = sorted(folds[fold_idx])
        train = sorted(
            [idx for j, fold in enumerate(folds) for idx in fold if j != fold_idx]
        )
        splits.append((train, test))

    return splits


def stratified_subsample(
    cases: List[TortCase], n: int, seed: int = 42
) -> List[int]:
    """Select a stratified subsample of n cases.

    Returns sorted list of indices.
    """
    rng = random.Random(seed)

    pos_indices = [i for i, c in enumerate(cases) if c.court_decision]
    neg_indices = [i for i, c in enumerate(cases) if not c.court_decision]

    pos_ratio = len(pos_indices) / len(cases)
    n_pos = round(n * pos_ratio)
    n_neg = n - n_pos

    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    selected = pos_indices[:n_pos] + neg_indices[:n_neg]
    return sorted(selected)


def stratified_train_val_test_split(
    cases: List[TortCase],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified train/val/test split, stratified on court_decision.

    Returns (train_indices, val_indices, test_indices).
    """
    rng = random.Random(seed)

    pos_indices = [i for i, c in enumerate(cases) if c.court_decision]
    neg_indices = [i for i, c in enumerate(cases) if not c.court_decision]

    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    def _split_list(indices, val_r, test_r):
        n = len(indices)
        n_test = round(n * test_r)
        n_val = round(n * val_r)
        test = indices[:n_test]
        val = indices[n_test : n_test + n_val]
        train = indices[n_test + n_val :]
        return train, val, test

    pos_train, pos_val, pos_test = _split_list(pos_indices, val_ratio, test_ratio)
    neg_train, neg_val, neg_test = _split_list(neg_indices, val_ratio, test_ratio)

    return (
        sorted(pos_train + neg_train),
        sorted(pos_val + neg_val),
        sorted(pos_test + neg_test),
    )


def format_case_for_prompt(case: TortCase, include_undisputed: bool = True) -> str:
    """Format a case as a structured text block for LLM prompts.

    Uses AF argument IDs: u{i}, p{i}, d{i}.
    """
    parts = []

    if include_undisputed and case.undisputed_facts:
        parts.append("Undisputed Facts:")
        for i, u in enumerate(case.undisputed_facts):
            parts.append(f"  u{i}: {u.description}")
        parts.append("")

    parts.append("Plaintiff Claims:")
    for i, p in enumerate(case.plaintiff_claims):
        parts.append(f"  p{i}: {p.description}")
    parts.append("")

    parts.append("Defendant Claims:")
    for i, d in enumerate(case.defendant_claims):
        parts.append(f"  d{i}: {d.description}")

    return "\n".join(parts)


def get_argument_ids(case: TortCase, include_undisputed: bool = False) -> List[str]:
    """Get the AF argument IDs for a case.

    Returns list of IDs: [u0, u1, ..., p0, p1, ..., d0, d1, ...].
    """
    ids = []
    if include_undisputed:
        ids.extend(f"u{i}" for i in range(len(case.undisputed_facts)))
    ids.extend(f"p{i}" for i in range(len(case.plaintiff_claims)))
    ids.extend(f"d{i}" for i in range(len(case.defendant_claims)))
    return ids
