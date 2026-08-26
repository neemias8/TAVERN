"""
Stage 6 - oracle selection accuracy (thesis Section 9.3.2).

Over the contested events, the fraction on which the system selects the version
with the highest ROUGE-L F1 against that event's segment of the reference.

For TAVERN this requires one adjustment. Under a curated timeline the contested
events are given; under induction the system produces its own clusters, and a
cluster may not correspond to any canonical event. Accuracy is therefore
computed over the INTERSECTION: contested canonical events for which the induced
clustering produced a cluster containing at least two documents. The size of
that intersection is reported with the accuracy, since a system that clusters
conservatively could otherwise obtain a flattering accuracy on a small subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .chronology import CanonicalEvent, Chronology
from .content_metrics import rouge


@dataclass
class SelectionResult:
    accuracy: Optional[float]
    evaluated: int
    correct: int
    per_event: Dict[int, dict] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"accuracy": None if self.accuracy is None
                else round(self.accuracy, 4), "events": self.evaluated}


def reference_segments(chronology: Chronology, reference: str
                       ) -> Dict[int, str]:
    """Align the reference consolidation to canonical events.

    The reference numbers its events, opening each with the event's ordinal, so
    the segmentation is recoverable exactly rather than estimated. Where the
    numbering is absent the reference is split proportionally to the events'
    selected-version lengths, and those events are excluded from the accuracy.
    """
    import re

    events = chronology.events
    segments: Dict[int, str] = {}
    positions: List[Tuple[int, int]] = []
    cursor = 0
    for e in events:
        marker = re.compile(rf"(?:^|(?<=\s)){e.event_id}\s")
        m = marker.search(reference, cursor)
        if m is None:
            continue
        positions.append((m.start(), e.event_id))
        cursor = m.end()
    for (start, eid), nxt in zip(positions,
                                 positions[1:] + [(len(reference), None)]):
        body = reference[start:nxt[0]]
        body = re.sub(rf"^{eid}\s+", "", body.strip())
        segments[eid] = body.strip()
    return segments


def evaluate(chronology: Chronology, reference: str,
             selected: Dict[int, str],
             restrict_to: Optional[Sequence[int]] = None) -> SelectionResult:
    """`selected` maps canonical event id -> the book whose version was chosen."""
    segments = reference_segments(chronology, reference)
    allowed = set(restrict_to) if restrict_to is not None else None
    correct = evaluated = 0
    per_event: Dict[int, dict] = {}
    for e in chronology.events:
        if not e.contested():
            continue
        if allowed is not None and e.event_id not in allowed:
            continue
        seg = segments.get(e.event_id)
        if not seg:
            continue
        pick = selected.get(e.event_id)
        if pick is None:
            continue
        scores = {b: rouge(e.texts[b], seg)["rougeL"] for b in e.books}
        best = max(scores, key=scores.get)
        evaluated += 1
        hit = pick == best
        correct += int(hit)
        per_event[e.event_id] = {"picked": pick, "best": best,
                                 "scores": {k: round(v, 4)
                                            for k, v in scores.items()},
                                 "correct": hit}
    return SelectionResult(
        accuracy=(correct / evaluated) if evaluated else None,
        evaluated=evaluated, correct=correct, per_event=per_event)


def analytical_floor(chronology: Chronology) -> Tuple[float, int]:
    """Equation eq:floor: the expected accuracy of uniform random selection over
    the contested events."""
    num = 0.0
    n = 0
    for e in chronology.events:
        if e.contested():
            num += 1.0 / e.n_versions
            n += 1
    return (num / n if n else 0.0), n


def induced_selection(clustering, units, matching: Dict[int, str],
                      chosen_book: Dict[str, str]) -> Dict[int, str]:
    """Project the induced clusters' selections onto canonical event ids."""
    out: Dict[int, str] = {}
    for eid, cid in matching.items():
        book = chosen_book.get(cid)
        if book:
            out[eid] = book
    return out


def contested_intersection(chronology: Chronology, clustering,
                           matching: Dict[int, str]) -> List[int]:
    """Contested canonical events whose matched cluster has at least two
    documents."""
    out: List[int] = []
    for e in chronology.events:
        if not e.contested():
            continue
        cid = matching.get(e.event_id)
        if cid is None:
            continue
        cl = clustering.by_id(cid)
        if cl is not None and cl.size >= 2:
            out.append(e.event_id)
    return out
