"""Core data structures for Bipolar Argumentation Frameworks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Argument:
    """A single argument (node) in a BAF, grounded in a text span."""

    id: str
    text: str
    start: int  # character offset in source text
    end: int  # character offset in source text

    def iou(self, other: Argument) -> float:
        """Character-level Intersection-over-Union with another argument span."""
        inter_start = max(self.start, other.start)
        inter_end = min(self.end, other.end)
        intersection = max(0, inter_end - inter_start)
        union = (self.end - self.start) + (other.end - other.start) - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class Relation:
    """A directed relation (edge) in a BAF."""

    source: str  # argument id
    target: str  # argument id
    type: str  # "support" or "attack"


@dataclass
class BAF:
    """A Bipolar Argumentation Framework extracted from a text."""

    arguments: List[Argument] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def arg_by_id(self, arg_id: str) -> Optional[Argument]:
        for a in self.arguments:
            if a.id == arg_id:
                return a
        return None

    def to_dict(self) -> dict:
        return {
            "arguments": [
                {"id": a.id, "text": a.text, "start": a.start, "end": a.end}
                for a in self.arguments
            ],
            "relations": [
                {"source": r.source, "target": r.target, "type": r.type}
                for r in self.relations
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> BAF:
        arguments = [Argument(**a) for a in d.get("arguments", [])]
        relations = [Relation(**r) for r in d.get("relations", [])]
        return cls(arguments=arguments, relations=relations)

    @classmethod
    def from_json(cls, s: str) -> BAF:
        return cls.from_dict(json.loads(s))
