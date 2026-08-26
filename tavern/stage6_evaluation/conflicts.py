"""
Stage 6 - the conflict report (thesis Table tab:res-conflicts).

Intra-document conflicts indicate annotation error; inter-document conflicts
indicate disagreement between sources. Both fall out of the same machinery that
performs the ordering: unsatisfiability under closure, and the arcs the feedback
arc set removes. The mechanism needs no thresholds and cannot fail silently.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .consistency import known_conflicts


@dataclass
class ConflictReport:
    intra: int = 0
    inter: int = 0
    removed: int = 0
    documented: List[str] = field(default_factory=list)
    detail: str = ""
    examples: List[dict] = field(default_factory=list)

    def as_rows(self) -> List[dict]:
        return [
            {"class": "Intra-document (annotation error)", "count": self.intra,
             "documented": "---"},
            {"class": "Inter-document (source divergence)", "count": self.inter,
             "documented": ", ".join(self.documented) or "none"},
            {"class": "  of which removed by feedback arc set",
             "count": self.removed, "documented": "---"},
        ]


def report(structs, induced, clustering, units) -> ConflictReport:
    intra = sum(1 for s in structs.values() for c in s.conflicts
                if c.get("scope") == "intra-document")
    ordering = [c for c in induced.conflicts
                if c.get("kind") in ("ordering", "unsatisfiable")]
    found, detail = known_conflicts(induced.conflicts, clustering, units)

    examples = []
    for c in ordering[:20]:
        ex = {"kind": c.get("kind"), "books": c.get("books", [])}
        if c.get("units"):
            ex["episodes"] = [units[u].ref for u in c["units"] if u in units]
            ex["scores"] = c.get("scores")
        if c.get("clusters"):
            a, b = c["clusters"]
            ca, cb = clustering.by_id(a), clustering.by_id(b)
            ex["refs_a"] = [units[m].ref for m in (ca.members if ca else [])]
            ex["refs_b"] = [units[m].ref for m in (cb.members if cb else [])]
            ex["per_document"] = c.get("per_document", {})
        examples.append(ex)

    return ConflictReport(intra=intra, inter=len(ordering),
                          removed=len(induced.removed), documented=found,
                          detail=detail, examples=examples)
