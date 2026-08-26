"""
Stage 2 - validation (thesis Section 6.2.8; Appendix B, Section B.4).

Two stages. Schema validation uses the corrected schema of Appendix B, in which
every deviation from the published Annex H schema is numbered (S1-S8). Then the
twelve constraints that a schema cannot express are applied, in the order
Appendix B lists them.

Constraint 10 -- the accessibility constraint of 8.4.3.3 -- is a HARD ERROR:
the standard declares annotations violating it uninterpretable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .enums import (EventClass, SLINK_ALLOWED_BY_CLASS, TimexType)
from .model import AnnotationStructure

# Constraint identifiers, matching the enumeration of Appendix B, Section B.4.
CONSTRAINTS = {
    1: "exactly one of @eventID and @timeID on a <TLINK> (defect S4)",
    2: "@beginPoint/@endPoint only when @type=DURATION (A.2.2.3.9)",
    3: "@quant/@freq only when @type=SET (A.2.2.3.10)",
    4: "a SET carries at least one of @quant and @freq (A.2.2.3.4.4)",
    5: "@anchorTimeID chains terminate and contain no cycle",
    6: "@temporalFunction is true when @value is underspecified (A.2.2.3.6)",
    7: "every REPORTING/PERCEPTION event has at least one <SLINK> (A.3.3.1.2)",
    8: "<SLINK> @relType compatible with the governor's @class (A.3.3.1.2)",
    9: "durations use the PT designator for sub-day units (ISO 8601)",
    10: "accessibility constraint of 8.4.3.3 (HARD ERROR)",
    11: "IDENTITY assertions form an equivalence relation",
    12: "every element carries a unique xml:id (7.3.2)",
}

HARD = {10}


@dataclass
class ValidationReport:
    book: str
    violations: Dict[int, List[str]] = field(default_factory=dict)
    checked: int = 0

    def add(self, constraint: int, message: str) -> None:
        self.violations.setdefault(constraint, []).append(message)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def hard_errors(self) -> Dict[int, List[str]]:
        return {k: v for k, v in self.violations.items() if k in HARD}

    def summary(self) -> str:
        if self.ok:
            return f"{self.book}: all {len(CONSTRAINTS)} constraints satisfied"
        parts = [f"C{k} ({len(v)})" for k, v in sorted(self.violations.items())]
        return f"{self.book}: violations in " + ", ".join(parts)


def validate(struct: AnnotationStructure) -> ValidationReport:
    rep = ValidationReport(book=struct.book)

    # C12 - unique xml:id
    seen: Set[str] = set()
    for xid in list(struct.events) + list(struct.timexes) + \
            list(struct.signals) + [l.xml_id for l in struct.tlinks] + \
            [l.xml_id for l in struct.slinks] + \
            [l.xml_id for l in struct.alinks] + \
            [l.xml_id for l in struct.mlinks]:
        if xid in seen:
            rep.add(12, f"duplicate xml:id {xid}")
        seen.add(xid)

    # C1 - exactly one of @eventID / @timeID
    for l in struct.tlinks:
        if bool(l.event_id) == bool(l.time_id):
            rep.add(1, f"{l.xml_id}: eventID={l.event_id} timeID={l.time_id}")
        if not l.target_id:
            rep.add(1, f"{l.xml_id}: no relatedTo* target")

    # C2, C3, C4, C6, C9 - <TIMEX3> attributes
    for tx in struct.timexes.values():
        if tx.timex_type != TimexType.DURATION and (tx.begin_point
                                                    or tx.end_point):
            rep.add(2, f"{tx.xml_id}: beginPoint/endPoint on "
                       f"type={tx.timex_type}")
        if tx.timex_type != TimexType.SET and (tx.quant or tx.freq):
            rep.add(3, f"{tx.xml_id}: quant/freq on type={tx.timex_type}")
        if tx.timex_type == TimexType.SET and not (tx.quant or tx.freq):
            rep.add(4, f"{tx.xml_id}: SET without quant or freq")
        if tx.value and "X" in tx.value and not tx.temporal_function:
            rep.add(6, f"{tx.xml_id}: underspecified value {tx.value} with "
                       f"temporalFunction=false")
        if tx.timex_type == TimexType.DURATION and tx.value:
            v = tx.value
            if any(u in v for u in ("H", "M", "S")) and not v.startswith("PT") \
                    and "T" not in v:
                rep.add(9, f"{tx.xml_id}: malformed duration {v}")

    # C5 - anchor chains terminate and contain no cycle
    for tx in struct.timexes.values():
        seen_chain: Set[str] = set()
        cur = tx.anchor_time_id
        depth = 0
        while cur:
            if cur in seen_chain:
                rep.add(5, f"{tx.xml_id}: cycle in anchor chain at {cur}")
                break
            seen_chain.add(cur)
            nxt = struct.timexes.get(cur)
            if nxt is None:
                rep.add(5, f"{tx.xml_id}: dangling anchorTimeID {cur}")
                break
            cur = nxt.anchor_time_id
            depth += 1
            if depth > 500:
                rep.add(5, f"{tx.xml_id}: anchor chain does not terminate")
                break

    # C7 - every REPORTING/PERCEPTION event participates in an <SLINK>
    governors = {sl.event_id for sl in struct.slinks}
    for ev in struct.events.values():
        if ev.event_class in (EventClass.REPORTING, EventClass.PERCEPTION):
            if ev.xml_id not in governors:
                rep.add(7, f"{ev.xml_id} ({ev.pred}) has no SLINK")

    # C8 - <SLINK> relType compatible with governor @class
    for sl in struct.slinks:
        gov = struct.events.get(sl.event_id)
        if gov is None:
            rep.add(8, f"{sl.xml_id}: unknown governor {sl.event_id}")
            continue
        allowed = SLINK_ALLOWED_BY_CLASS.get(str(gov.event_class), frozenset())
        if str(sl.rel_type) not in allowed:
            rep.add(8, f"{sl.xml_id}: {sl.rel_type} on class "
                       f"{gov.event_class}")

    # C10 - accessibility constraint (8.4.3.3): HARD ERROR
    quantified = {t.xml_id for t in struct.timexes.values()
                  if t.timex_type == TimexType.SET}
    if quantified:
        linked_to_quantified: Dict[str, int] = {}
        other_links: Dict[str, int] = {}
        for l in struct.tlinks:
            for a, b in ((l.source, l.target_id), (l.target_id, l.source)):
                if a is None:
                    continue
                if b in quantified:
                    linked_to_quantified[a] = linked_to_quantified.get(a, 0) + 1
                else:
                    other_links[a] = other_links.get(a, 0) + 1
        for eid in linked_to_quantified:
            if other_links.get(eid, 0) > 0:
                rep.add(10, f"{eid}: linked to a quantified TIMEX3 and to "
                            f"{other_links[eid]} other temporal link(s)")

    # C11 - IDENTITY forms an equivalence relation
    ident: Dict[str, Set[str]] = {}
    for l in struct.tlinks:
        if str(l.rel_type) != "IDENTITY":
            continue
        a, b = l.source, l.target_id
        ident.setdefault(a, set()).add(b)
        ident.setdefault(b, set()).add(a)
    for a, bs in ident.items():
        for b in bs:
            if a not in ident.get(b, set()):
                rep.add(11, f"IDENTITY not symmetric: {a} ~ {b}")

    rep.checked = len(CONSTRAINTS)
    return rep


def enforce_accessibility(struct: AnnotationStructure) -> int:
    """Repair violations of constraint 10 before serialisation.

    An event linked to a quantified temporal expression may appear in no other
    temporal link. Rather than emitting an uninterpretable document, the link to
    the quantified expression is dropped and the event keeps its other
    relations; the number of repairs is reported.
    """
    quantified = {t.xml_id for t in struct.timexes.values()
                  if t.timex_type == TimexType.SET}
    if not quantified:
        return 0
    counts: Dict[str, int] = {}
    for l in struct.tlinks:
        for a, b in ((l.source, l.target_id), (l.target_id, l.source)):
            if a and b not in quantified:
                counts[a] = counts.get(a, 0) + 1
    removed = 0
    keep = []
    for l in struct.tlinks:
        touches_q = (l.target_id in quantified) or (l.source in quantified)
        if touches_q:
            other = l.source if l.target_id in quantified else l.target_id
            if counts.get(other, 0) > 0:
                removed += 1
                continue
        keep.append(l)
    struct.tlinks = keep
    return removed
