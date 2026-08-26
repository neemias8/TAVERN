"""
Layer B - closure of the temporal network under Allen's interval algebra, with
conflict detection (thesis Section 6.3.2, Algorithm alg:ann-closure).

Allen's thirteen basic relations. ISO-TimeML has no relType for `overlaps` /
`overlapped-by`, so those two appear only as elements of a computed disjunction
and are never emitted as assertions (Appendix B).

When closure narrows an edge to the empty set the asserted relations on that
cycle are jointly unsatisfiable. Within a document that is almost certainly an
annotation error and is reported as such. Across documents it means the sources
disagree, so the cycle is recorded in the conflict set K, the lowest-confidence
assertion is relaxed so closure can complete, and K is passed to Stage 3.
"""
from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .enums import CONFIDENCE_OF_LEVEL, TLinkRel
from .model import AnnotationStructure, TLink

# ---------------------------------------------------------------------------
# Allen's algebra
# ---------------------------------------------------------------------------

#: The thirteen basic relations.
ALLEN = ("b", "bi", "m", "mi", "o", "oi", "s", "si", "d", "di", "f", "fi", "e")
FULL: FrozenSet[str] = frozenset(ALLEN)

INVERSE = {"b": "bi", "bi": "b", "m": "mi", "mi": "m", "o": "oi", "oi": "o",
           "s": "si", "si": "s", "d": "di", "di": "d", "f": "fi", "fi": "f",
           "e": "e"}

#: ISO-TimeML relType -> Allen disjunction.
ISO_TO_ALLEN: Dict[str, FrozenSet[str]] = {
    "BEFORE": frozenset({"b"}),
    "AFTER": frozenset({"bi"}),
    "IBEFORE": frozenset({"m"}),
    "IAFTER": frozenset({"mi"}),
    "INCLUDES": frozenset({"di"}),
    "IS_INCLUDED": frozenset({"d"}),
    "DURING": frozenset({"d"}),
    "DURING_INV": frozenset({"di"}),
    "SIMULTANEOUS": frozenset({"e"}),
    "IDENTITY": frozenset({"e"}),
    "BEGINS": frozenset({"s"}),
    "BEGUN_BY": frozenset({"si"}),
    "ENDS": frozenset({"f"}),
    "ENDED_BY": frozenset({"fi"}),
}

#: Allen singleton -> ISO relType, for emitting derived relations.
ALLEN_TO_ISO = {
    "b": "BEFORE", "bi": "AFTER", "m": "IBEFORE", "mi": "IAFTER",
    "di": "INCLUDES", "d": "IS_INCLUDED", "e": "SIMULTANEOUS",
    "s": "BEGINS", "si": "BEGUN_BY", "f": "ENDS", "fi": "ENDED_BY",
    # 'o' and 'oi' have no ISO relType and are never emitted
}

_PRECEDES = frozenset({"b", "m", "o", "s", "d", "f"})   # i starts before or within j
_STRICT_BEFORE = frozenset({"b", "m"})


def _build_composition() -> Dict[Tuple[str, str], FrozenSet[str]]:
    """Allen's composition table, generated from the endpoint algebra.

    An interval is a pair (s, e) with s < e. Each basic relation fixes the sign
    of the four endpoint comparisons; composition is the set of relations
    consistent with some point-consistent chaining. Generating the table
    removes the risk of a transcription error in 169 hand-typed cells.
    """
    # endpoint constraint signature per relation: comparisons of
    # (s1 vs s2, s1 vs e2, e1 vs s2, e1 vs e2), values in {-1, 0, 1}
    sig = {
        "b":  (-1, -1, -1, -1),
        "bi": (1, 1, 1, 1),
        "m":  (-1, -1, 0, -1),
        "mi": (1, 0, 1, 1),
        "o":  (-1, -1, 1, -1),
        "oi": (1, -1, 1, 1),
        "s":  (0, -1, 1, -1),
        "si": (0, -1, 1, 1),
        "d":  (1, -1, 1, -1),
        "di": (-1, -1, 1, 1),
        "f":  (1, -1, 1, 0),
        "fi": (-1, -1, 1, 0),
        "e":  (0, -1, 1, 0),
    }

    def rel_of(points) -> Optional[str]:
        for name, s in sig.items():
            if s == points:
                return name
        return None

    # Enumerate integer interval triples and record which compositions occur.
    table: Dict[Tuple[str, str], Set[str]] = {}
    coords = range(0, 6)
    intervals = [(a, b) for a in coords for b in coords if a < b]

    def signature(i, j):
        (s1, e1), (s2, e2) = i, j
        return (_cmp(s1, s2), _cmp(s1, e2), _cmp(e1, s2), _cmp(e1, e2))

    for i in intervals:
        for j in intervals:
            rij = rel_of(signature(i, j))
            if rij is None:
                continue
            for k in intervals:
                rjk = rel_of(signature(j, k))
                rik = rel_of(signature(i, k))
                if rjk is None or rik is None:
                    continue
                table.setdefault((rij, rjk), set()).add(rik)
    return {k: frozenset(v) for k, v in table.items()}


def _cmp(a: int, b: int) -> int:
    return -1 if a < b else (0 if a == b else 1)


COMPOSITION = _build_composition()


def compose(r1: FrozenSet[str], r2: FrozenSet[str]) -> FrozenSet[str]:
    out: Set[str] = set()
    for a in r1:
        for b in r2:
            out |= COMPOSITION.get((a, b), FULL)
    return frozenset(out)


def invert(r: Iterable[str]) -> FrozenSet[str]:
    return frozenset(INVERSE[x] for x in r)


def allen_of(rel_type) -> FrozenSet[str]:
    return ISO_TO_ALLEN.get(str(rel_type), FULL)


def implies_before(r: FrozenSet[str]) -> bool:
    """True when every relation in the disjunction places i before j."""
    return bool(r) and r <= _STRICT_BEFORE


def implies_precedence(r: FrozenSet[str]) -> bool:
    return bool(r) and r <= _PRECEDES and r != frozenset({"e"})


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------

class ClosureResult:
    def __init__(self):
        self.network: Dict[Tuple[str, str], FrozenSet[str]] = {}
        self.provenance: Dict[Tuple[str, str], dict] = {}
        self.conflicts: List[dict] = []
        self.identity: Dict[str, str] = {}
        self.nodes: List[str] = []


def close(struct: AnnotationStructure, enabled: bool = True) -> ClosureResult:
    """Algorithm alg:ann-closure, restricted to timeline-eligible events."""
    res = ClosureResult()
    V = [e for e in struct.events if e in struct.eligible]
    vset = set(V)

    # 1. IDENTITY assertions form an equivalence over interval variables
    parent: Dict[str, str] = {v: v for v in V}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for l in struct.tlinks:
        if str(l.rel_type) != "IDENTITY":
            continue
        a, b = l.source, l.target_id
        if a in parent and b in parent:
            union(a, b)
    res.identity = {v: find(v) for v in V}
    reps = sorted(set(res.identity.values()))
    res.nodes = reps

    # 2. initialise and intersect asserted relations
    net: Dict[Tuple[str, str], FrozenSet[str]] = {}
    prov: Dict[Tuple[str, str], dict] = {}

    def get(i, j) -> FrozenSet[str]:
        if i == j:
            return frozenset({"e"})
        return net.get((i, j), FULL)

    def put(i, j, r, meta=None):
        net[(i, j)] = r
        net[(j, i)] = invert(r)
        if meta:
            prov[(i, j)] = meta

    for l in struct.tlinks:
        if l.origin != "asserted":
            continue
        a, b = l.source, l.target_id
        if a not in vset or b not in vset:
            continue          # relations to timexes handled by the scaffold
        ra, rb = res.identity[a], res.identity[b]
        if ra == rb:
            continue
        r = allen_of(l.rel_type)
        cur = get(ra, rb)
        new = cur & r
        if not new:
            res.conflicts.append({
                "kind": "assertion",
                "nodes": [ra, rb],
                "relations": [str(l.rel_type)],
                "levels": [l.level],
                "confidence": l.confidence,
                "scope": "intra-document",
                "book": struct.book,
            })
            new = r                       # relax: the new assertion wins
        put(ra, rb, new, {
            "origin": "asserted", "level": l.level,
            "confidence": l.confidence, "signal": l.signal_id,
        })

    if not enabled:
        res.network = net
        res.provenance = prov
        return res

    # 3. path consistency
    changed = True
    guard = 0
    while changed and guard < 40:
        changed = False
        guard += 1
        for j in reps:
            for i in reps:
                if i == j:
                    continue
                rij = get(i, j)
                if rij == FULL:
                    continue
                for k in reps:
                    if k == i or k == j:
                        continue
                    rjk = get(j, k)
                    if rjk == FULL:
                        continue
                    cand = compose(rij, rjk)
                    cur = get(i, k)
                    new = cur & cand
                    if not new:
                        # inconsistency: relax the lowest-confidence assertion
                        cyc = [(i, j), (j, k), (i, k)]
                        confs = [(prov.get(e, {}).get("confidence", 1.0), e)
                                 for e in cyc]
                        confs.sort()
                        weakest = confs[0][1]
                        res.conflicts.append({
                            "kind": "cycle",
                            "nodes": [i, j, k],
                            "relaxed": list(weakest),
                            "levels": [prov.get(e, {}).get("level")
                                       for e in cyc],
                            "scope": "intra-document",
                            "book": struct.book,
                        })
                        net.pop(weakest, None)
                        net.pop((weakest[1], weakest[0]), None)
                        prov.pop(weakest, None)
                        changed = True
                        continue
                    if new != cur:
                        put(i, k, new)
                        if (i, k) not in prov:
                            prov[(i, k)] = {"origin": "closure", "level": 5,
                                            "confidence":
                                                CONFIDENCE_OF_LEVEL[5]}
                        changed = True

    res.network = net
    res.provenance = prov
    return res


def apply_to_struct(struct: AnnotationStructure, res: ClosureResult,
                    emit_derived: bool = True) -> None:
    """Record the closure on the annotation structure, emitting derived
    <TLINK> elements marked origin="closure"."""
    struct.closed_network = {k: set(v) for k, v in res.network.items()}
    struct.network_provenance = res.provenance
    struct.conflicts.extend(res.conflicts)
    struct.identity_classes = res.identity
    if not emit_derived:
        return
    seen = {(l.source, l.target_id) for l in struct.tlinks}
    for (i, j), r in res.network.items():
        if (i, j) in seen or (j, i) in seen:
            continue
        meta = res.provenance.get((i, j), {})
        if meta.get("origin") != "closure":
            continue
        if len(r) != 1:
            continue
        iso = ALLEN_TO_ISO.get(next(iter(r)))
        if iso is None:
            continue
        struct.add_tlink(TLink(
            xml_id=struct.next_id("l"),
            rel_type=TLinkRel(iso),
            event_id=i,
            related_to_event=j,
            origin="closure",
            level=5,
            confidence=CONFIDENCE_OF_LEVEL[5],
        ))
        seen.add((i, j))
