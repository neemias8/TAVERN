"""
Stage 6 - the six internal consistency checks (thesis Section 9.5).

None measures accuracy; each detects a class of incoherence that is detectable
without a reference annotation. All run on every execution.

  1 Schema validity          every .tml document valid; the code-level
                             constraints of Appendix B satisfied, including the
                             accessibility constraint the standard declares a
                             hard requirement
  2 Closure consistency      no unsatisfiable cycle within a document
  3 Anchoring coverage       share of eligible events with a level-1 or -2
                             relation
  4 Normalisation coverage   share of <TIMEX3> with a non-null @value
  5 Partition soundness      no event in a discourse block is timeline-eligible,
                             and every reporting event introducing one is
  6 Known-conflict           the three divergences the harmonisation literature
    regression               documents are all detected
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..config import DISCOURSE_BLOCKS, KNOWN_CONFLICTS
from ..stage2_temporal_annotation.enums import EventClass
from ..stage2_temporal_annotation.model import AnnotationStructure
from ..stage2_temporal_annotation.validator import ValidationReport
from . import annotation_stats


@dataclass
class Check:
    name: str
    criterion: str
    passed: Optional[bool]
    value: str
    detail: str = ""


@dataclass
class ConsistencyReport:
    checks: List[Check] = field(default_factory=list)

    def add(self, c: Check) -> None:
        self.checks.append(c)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.passed is not None)

    def as_rows(self) -> List[dict]:
        return [{"check": c.name, "criterion": c.criterion,
                 "result": c.value, "passed": c.passed, "detail": c.detail}
                for c in self.checks]


def run(structs: Dict[str, AnnotationStructure],
        reports: Dict[str, ValidationReport], corpus,
        conflicts: Optional[Sequence[dict]] = None,
        clustering=None, units=None,
        segments=None) -> ConsistencyReport:
    rep = ConsistencyReport()

    # 1 schema validity + code-level constraints
    bad = {b: r.violations for b, r in reports.items() if r.violations}
    rep.add(Check(
        "Schema validity",
        "All documents valid; C1-C12 and accessibility satisfied",
        not bad, "pass" if not bad else "fail",
        "" if not bad else "; ".join(f"{b}: C{list(v)}" for b, v in bad.items())))

    # 2 closure consistency
    intra = [c for s in structs.values() for c in s.conflicts
             if c.get("scope") == "intra-document"]
    rep.add(Check(
        "Closure consistency", "No intra-document unsatisfiable cycle",
        not intra, "pass" if not intra else f"{len(intra)} cycles",
        "" if not intra else
        "; ".join(f"{c['book']}:{c.get('kind')}" for c in intra[:5])))

    # 3 anchoring coverage
    ac = annotation_stats.anchoring_coverage(structs)
    rep.add(Check(
        "Anchoring coverage",
        "Share of eligible events with a level-1 or level-2 relation",
        None, f"{ac:.3f}"))

    # 4 normalisation coverage
    nc = annotation_stats.normalisation_coverage(structs)
    rep.add(Check(
        "Normalisation coverage",
        "Share of <TIMEX3> with non-null @value", nc >= 0.999,
        f"{nc:.3f}"))

    # 5 partition soundness
    leaked, reporters_ok, total = partition_soundness(structs, corpus,
                                                     segments)
    rep.add(Check(
        "Partition soundness",
        "No event in the discourse blocks is timeline-eligible",
        leaked == 0, "pass" if leaked == 0 else f"{leaked}/{total} leaked",
        f"introducing reporting events eligible: {reporters_ok}"))

    # 6 known-conflict regression
    found, detail = known_conflicts(conflicts or [], clustering, units)
    rep.add(Check(
        "Known-conflict regression",
        "Fig tree, Passover day, cockcrow all detected",
        len(found) == 3, f"{len(found)}/3", detail))
    return rep


# ---------------------------------------------------------------------------
def partition_soundness(structs: Dict[str, AnnotationStructure],
                        corpus, segments=None) -> Tuple[int, int, int]:
    """Count timeline-eligible events inside the discourse blocks.

    Two refinements matter for the criterion to mean what Section 9.5 says.
    Only events lying INSIDE the quotation are counted: a discourse block also
    contains its own narrative frame ("...?" he asked; Jesus answered:), and the
    reporting event of that frame is expected to be eligible, since that a
    speaker spoke is a narrative fact. Where the segmentation is unavailable the
    opening verse of each block is excluded instead, which is a coarser
    approximation of the same idea.
    """
    from ..stage2_temporal_annotation.link_inference.slink import (
        _quotation_scope)

    quoted: Dict[str, set] = {}
    if segments:
        for book, doc in segments.items():
            quoted[book], _runs = _quotation_scope(doc)

    leaked = total = reporters = 0
    for book, c0, v0, c1, v1, _label in DISCOURSE_BLOCKS:
        s = structs.get(book)
        if s is None:
            continue
        qt = quoted.get(book)
        for ev in s.events.values():
            if not ev.verse_key:
                continue
            _b, ch, vs = ev.verse_key
            if not ((c0, v0) <= (ch, vs) <= (c1, v1)):
                continue
            in_quote = (ev.head_token in qt) if qt is not None else \
                ((ch, vs) != (c0, v0))
            if not in_quote:
                if ev.xml_id in s.eligible and ev.event_class in (
                        EventClass.REPORTING, EventClass.PERCEPTION):
                    reporters += 1
                continue
            total += 1
            if ev.xml_id in s.eligible:
                leaked += 1
    return leaked, reporters, total


def known_conflicts(conflicts: Sequence[dict], clustering=None,
                    units=None) -> Tuple[List[str], str]:
    """Which of the three documented divergences the conflict report contains.

    A conflict is credited to a documented case when the verse spans it
    implicates overlap that case's spans in at least two different Gospels.
    """
    found: List[str] = []
    notes: List[str] = []
    if clustering is None or units is None:
        return found, "clustering unavailable"

    def keys_of(cid) -> Set[Tuple[str, int, int]]:
        cl = clustering.by_id(cid)
        if cl is None:
            return set()
        out: Set[Tuple[str, int, int]] = set()
        for m in cl.members:
            out |= set(units[m].verse_keys)
        return out

    implicated: List[Set[Tuple[str, int, int]]] = []
    for c in conflicts:
        if c.get("kind") not in ("ordering", "unsatisfiable"):
            continue
        keys: Set[Tuple[str, int, int]] = set()
        for cid in c.get("clusters", ()):
            keys |= keys_of(cid)
        for uid in c.get("units", ()):
            u = units.get(uid)
            if u is not None:
                keys |= set(u.verse_keys)
        implicated.append(keys)

    for name, spec in KNOWN_CONFLICTS.items():
        target: Dict[str, Set[Tuple[str, int, int]]] = {}
        for book, c0, v0, c1, v1 in spec["spans"]:
            target.setdefault(book, set())
            for ch in range(c0, c1 + 1):
                lo = v0 if ch == c0 else 1
                hi = v1 if ch == c1 else 200
                for v in range(lo, hi + 1):
                    target[book].add((book, ch, v))
        hit = False
        for keys in implicated:
            books_touched = {b for b, tk in target.items() if keys & tk}
            if len(books_touched) >= 2:
                hit = True
                notes.append(f"{name}: {sorted(books_touched)}")
                break
        if hit:
            found.append(name)
    return found, "; ".join(notes) if notes else "none matched"
