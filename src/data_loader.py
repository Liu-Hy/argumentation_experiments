"""Load BRAT-annotated datasets and convert to unified BAF format.

Currently supports the Persuasive Essays v2 (AAEC) corpus.
Designed to be extended for other argumentation datasets.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Tuple

from .baf import Argument, BAF, Relation


# ---------------------------------------------------------------------------
# BRAT parsing
# ---------------------------------------------------------------------------

def parse_brat_ann(ann_path: str, txt_path: str) -> Tuple[BAF, str]:
    """Parse a BRAT .ann file and its corresponding .txt file into a BAF.

    Component types (MajorClaim, Claim, Premise) are merged into generic
    arguments.  Relation types are normalised: supports -> support,
    attacks -> attack.  Stance attributes are ignored.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        essay_text = f.read()

    arguments: List[Argument] = []
    relations: List[Relation] = []
    component_types = {"MajorClaim", "Claim", "Premise"}

    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("T"):
                # Format: T1\tMajorClaim 503 575\texact text
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                entity_id = parts[0]
                type_and_offsets = parts[1].split()
                entity_type = type_and_offsets[0]
                if entity_type not in component_types:
                    continue
                # Handle discontinuous spans: take outermost boundary
                start = int(type_and_offsets[1])
                end = int(type_and_offsets[-1])
                text = parts[2]
                arguments.append(
                    Argument(id=entity_id, text=text, start=start, end=end)
                )

            elif line.startswith("R"):
                # Format: R1\tsupports Arg1:T4 Arg2:T3
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                rel_parts = parts[1].split()
                rel_type_raw = rel_parts[0]
                type_map = {"supports": "support", "attacks": "attack"}
                rel_type = type_map.get(rel_type_raw)
                if rel_type is None:
                    continue
                source_id = rel_parts[1].split(":")[1]
                target_id = rel_parts[2].split(":")[1]
                relations.append(
                    Relation(source=source_id, target=target_id, type=rel_type)
                )
            # A-lines (stance) and other annotations are intentionally skipped

    return BAF(arguments=arguments, relations=relations), essay_text


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_persuasive_essays(dataset_dir: str) -> Dict[str, dict]:
    """Load the Persuasive Essays v2 dataset.

    Returns
    -------
    dict : essay_id -> {"baf": BAF, "text": str, "split": "train"|"test"}
    """
    brat_dir = os.path.join(dataset_dir, "brat-project-final")

    # Read train/test split
    split_map: Dict[str, str] = {}
    split_path = os.path.join(dataset_dir, "train-test-split.csv")
    with open(split_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 2:
                continue
            essay_id = row[0].strip().strip('"')
            split = row[1].strip().strip('"').lower()
            split_map[essay_id] = split

    dataset: Dict[str, dict] = {}
    for essay_id in sorted(split_map.keys()):
        ann_path = os.path.join(brat_dir, f"{essay_id}.ann")
        txt_path = os.path.join(brat_dir, f"{essay_id}.txt")
        if not (os.path.exists(ann_path) and os.path.exists(txt_path)):
            continue
        baf, text = parse_brat_ann(ann_path, txt_path)
        dataset[essay_id] = {"baf": baf, "text": text, "split": split_map[essay_id]}

    return dataset


def get_split(dataset: Dict[str, dict], split: str) -> Dict[str, dict]:
    """Filter dataset by split name ('train' or 'test')."""
    return {k: v for k, v in dataset.items() if v["split"] == split}


def select_fewshot_examples(
    train_data: Dict[str, dict], n: int = 3
) -> List[str]:
    """Select n diverse training essays for few-shot prompts.

    Strategy: pick one essay with attack relations, one short/simple essay,
    and one longer/complex essay.  Falls back to evenly-spaced selection if
    the heuristic fails.
    """
    items = sorted(train_data.items(), key=lambda x: x[0])

    # Partition by presence of attack relations
    has_attack = [
        (eid, d)
        for eid, d in items
        if any(r.type == "attack" for r in d["baf"].relations)
    ]
    no_attack = [
        (eid, d) for eid, d in items if (eid, d) not in has_attack
    ]

    selected: List[str] = []

    # 1. One essay with attack relations (prefer medium size)
    if has_attack:
        has_attack.sort(key=lambda x: len(x[1]["text"]))
        selected.append(has_attack[len(has_attack) // 2][0])

    # 2. One short/simple essay (fewest arguments, no attacks)
    if no_attack:
        no_attack.sort(key=lambda x: len(x[1]["baf"].arguments))
        selected.append(no_attack[0][0])

    # 3. One longer/complex essay (most arguments, no attacks)
    if no_attack and len(no_attack) > 1:
        candidate = no_attack[-1][0]
        if candidate not in selected:
            selected.append(candidate)

    # Fallback: fill remaining slots with evenly spaced essays
    remaining = [eid for eid, _ in items if eid not in selected]
    while len(selected) < n and remaining:
        idx = len(remaining) // (n - len(selected) + 1)
        selected.append(remaining.pop(idx))

    return selected[:n]


def create_val_split(
    train_data: Dict[str, dict], val_size: int = 40, seed: int = 42
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Split training data into train-proper and validation sets.

    Uses stratified sampling to ensure the validation set contains essays
    with attack relations in roughly the same proportion as the full
    training set.

    Returns
    -------
    (train_proper, val_data) : tuple of dicts with same format as input
    """
    items = sorted(train_data.items(), key=lambda x: x[0])

    has_attack = [
        (eid, d) for eid, d in items
        if any(r.type == "attack" for r in d["baf"].relations)
    ]
    no_attack = [
        (eid, d) for eid, d in items
        if not any(r.type == "attack" for r in d["baf"].relations)
    ]

    rng = random.Random(seed)
    rng.shuffle(has_attack)
    rng.shuffle(no_attack)

    # Proportional allocation
    attack_ratio = len(has_attack) / len(items) if items else 0.5
    n_attack_val = max(1, round(val_size * attack_ratio))
    n_no_attack_val = val_size - n_attack_val

    # Clamp to available
    n_attack_val = min(n_attack_val, len(has_attack))
    n_no_attack_val = min(n_no_attack_val, len(no_attack))

    val_items = has_attack[:n_attack_val] + no_attack[:n_no_attack_val]
    train_items = has_attack[n_attack_val:] + no_attack[n_no_attack_val:]

    return dict(train_items), dict(val_items)


def select_fewshot_examples_random(
    train_data: Dict[str, dict],
    n: int = 3,
    seed: int = 0,
    ensure_attacks: bool = True,
) -> List[str]:
    """Randomly select n training essays for few-shot prompts.

    If ensure_attacks is True and n >= 2, at least one selected essay will
    have attack relations (if any exist in the training data).

    Different seeds produce different example sets, enabling sensitivity
    analysis across example selections.
    """
    items = sorted(train_data.items(), key=lambda x: x[0])
    rng = random.Random(seed)

    has_attack = [
        (eid, d) for eid, d in items
        if any(r.type == "attack" for r in d["baf"].relations)
    ]
    no_attack = [
        (eid, d) for eid, d in items
        if not any(r.type == "attack" for r in d["baf"].relations)
    ]

    selected: List[str] = []

    if ensure_attacks and has_attack and n >= 2:
        # Pick one random essay with attacks
        attack_choice = rng.choice(has_attack)
        selected.append(attack_choice[0])
        remaining = [(eid, d) for eid, d in items if eid != attack_choice[0]]
    else:
        remaining = list(items)

    # Fill remaining slots randomly
    rng.shuffle(remaining)
    for eid, _d in remaining:
        if len(selected) >= n:
            break
        if eid not in selected:
            selected.append(eid)

    return selected[:n]
