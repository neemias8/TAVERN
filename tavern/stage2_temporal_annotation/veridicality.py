"""
Layer B - the veridicality partition (thesis Section 6.3.1).

For an event e, the modal context path w(e) is the sequence of <SLINK> relation
types on the path from the document root to e (Equation eq:worldpath). An event
is timeline-eligible iff its modal context path is empty or consists entirely
of FACTIVE relations (Equation eq:eligible).

Subordinated events are NOT discarded: they remain annotated, retain their
links and remain available to the generation stage. What they lose is
eligibility for the timeline. The reporting event itself is always eligible:
that a speaker spoke is a narrative fact even though what was said is not.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from .enums import ELIGIBILITY_PRESERVING
from .model import AnnotationStructure


def compute_modal_paths(struct: AnnotationStructure) -> Dict[str, List[str]]:
    """Shortest-depth modal context path per event.

    Where an event is subordinated by more than one governor, the path with the
    fewest non-eligibility-preserving relations is taken, so that an event
    reachable through a FACTIVE route is not penalised by an incidental
    EVIDENTIAL one.
    """
    children: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    subordinated: Set[str] = set()
    for sl in struct.slinks:
        children[sl.event_id].append((sl.subordinated_event, str(sl.rel_type)))
        subordinated.add(sl.subordinated_event)

    roots = [eid for eid in struct.events if eid not in subordinated]
    paths: Dict[str, List[str]] = {eid: [] for eid in roots}

    frontier = deque(roots)
    visited: Set[str] = set(roots)
    while frontier:
        cur = frontier.popleft()
        base = paths.get(cur, [])
        for child, rel in children.get(cur, []):
            cand = base + [rel]
            existing = paths.get(child)
            if existing is None or _cost(cand) < _cost(existing):
                paths[child] = cand
                if child not in visited or _cost(cand) < _cost(existing or []):
                    visited.add(child)
                    frontier.append(child)
        if len(visited) > 20 * (len(struct.events) + 1):
            break                      # cycle guard

    for eid in struct.events:
        paths.setdefault(eid, [])
    return paths


def _cost(path: List[str]) -> Tuple[int, int]:
    blocking = sum(1 for r in path if r not in ELIGIBILITY_PRESERVING)
    return (blocking, len(path))


def partition(struct: AnnotationStructure, enabled: bool = True) -> None:
    """Fill `struct.modal_paths` and `struct.eligible`.

    When `enabled` is False the partition is ablated: every annotated event is
    treated as timeline-eligible (thesis Section 9.6).
    """
    struct.modal_paths = compute_modal_paths(struct)
    if not enabled:
        struct.eligible = set(struct.events)
        return
    struct.eligible = {
        eid for eid, path in struct.modal_paths.items() if is_eligible(path)
    }


def is_eligible(path: List[str]) -> bool:
    """Equation eq:eligible."""
    return not path or all(r in ELIGIBILITY_PRESERVING for r in path)


def modal_context_types(path: List[str]) -> Set[str]:
    return set(path)
