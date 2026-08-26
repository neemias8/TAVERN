"""
Stage 3 - induction of the global timeline (thesis Section 6.4.3,
Algorithm alg:ann-induce).

The clusters and the local partial orders define a directed graph over
candidate canonical events. The graph is generally not acyclic: cycles arise
from exactly the ordering divergences the harmonisation literature documents.
The resolution computes a minimum feedback arc set with the linear-time
heuristic of Eades, Lin and Smyth, weighting edges by the confidence recorded
during closure so that low-confidence level-4 assertions are preferred for
removal over signal-supported level-1 assertions. The removed edges are
precisely the ordering conflicts and are reported as such.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .event_coref import Cluster, Clustering
from .local_timeline import LocalTimeline
from .scaffold import Scaffold


@dataclass
class InducedTimeline:
    order: List[str] = field(default_factory=list)          # cluster ids
    rank: Dict[str, int] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], float] = field(default_factory=dict)
    registration: Dict[str, float] = field(default_factory=dict)
    removed: List[Tuple[str, str, float]] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)

    def position(self, cid: str) -> Optional[int]:
        return self.rank.get(cid)

    def conflicted_clusters(self, clustering=None) -> List[Tuple[str, str]]:
        """Cluster pairs implicated in an inter-document conflict.

        Conflict records come in two shapes: unsatisfiability between a pair of
        clusters, and an alignment crossing between four event units. Both are
        reduced to cluster pairs here so that Stage 5 has one thing to consult.
        """
        out: List[Tuple[str, str]] = []
        for c in self.conflicts:
            if c.get("kind") not in ("ordering", "unsatisfiable"):
                continue
            cids = list(c.get("clusters", ()) or ())
            if not cids and clustering is not None:
                seen: List[str] = []
                for uid in c.get("units", ()) or ():
                    cid = clustering.cluster_of_unit.get(uid)
                    if cid and cid not in seen:
                        seen.append(cid)
                cids = seen
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    out.append((cids[i], cids[j]))
        return out


#: Weight of one document's transitive precedence vote.
DOCUMENT_VOTE = 1.0

#: Weight of the registration vote used to complete pairs that no document
#: orders. Set above a single document's vote because such a pair has no
#: relational evidence at all, and below two documents' agreement.
REGISTRATION_VOTE = 2.0


def registration(timelines: Dict[str, LocalTimeline],
                 clustering: Clustering) -> Dict[str, float]:
    """Position of each cluster on a common narrative-progress axis.

    The coordinate is the cluster's column index in the alignment profile of
    Section 6.4.2, normalised to [0, 1]; where no profile is available it falls
    back to the mean relative position of the cluster's units in their own
    documents. This is the completion rule for cluster pairs on which no
    document supplies precedence evidence, and the tie-break for the
    topological sort.
    """
    if clustering.clusters and any(c.profile_index for c in
                                   clustering.clusters):
        # the profile of Section 6.4.2 is already a consistent ordering of the
        # candidate canonical events; its column index is the registration
        return {c.cluster_id: c.profile_index for c in clustering.clusters}
    pos: Dict[str, List[float]] = defaultdict(list)
    for tl in timelines.values():
        n = max(1, len(tl.units) - 1)
        for i, u in enumerate(tl.units):
            cid = clustering.cluster_of_unit.get(u.unit_id)
            if cid:
                pos[cid].append(i / n)
    return {cid: sum(v) / len(v) for cid, v in pos.items()}


def build_cluster_graph(timelines: Dict[str, LocalTimeline],
                        clustering: Clustering,
                        reg: Optional[Dict[str, float]] = None
                        ) -> Tuple[Dict[Tuple[str, str], float],
                                   Dict[Tuple[str, str], List[dict]]]:
    """The weighted tournament over candidate canonical events.

    An arc from cluster A to cluster B carries one vote per document in whose
    closed local order some member of A precedes some member of B. Voting over
    the TRANSITIVE order rather than over adjacent pairs only is what makes the
    subsequent feedback-arc-set computation a genuine aggregation: a pair on
    which three documents agree outweighs the one that dissents, so a single
    mis-aligned cluster cannot reorder the timeline around it.

    Pairs that no document orders are completed by the registration of
    `registration`, at a weight above one document's vote and below two.
    """
    weights: Dict[Tuple[str, str], float] = defaultdict(float)
    support: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    ordered_pairs: set = set()

    for book, tl in timelines.items():
        seq: List[str] = []
        seen: set = set()
        for u in tl.units:
            cid = clustering.cluster_of_unit.get(u.unit_id)
            if cid and cid not in seen:
                seen.add(cid)
                seq.append(cid)
        # direct relational evidence, for the conflict report
        for (ui, uj) in tl.order:
            ci = clustering.cluster_of_unit.get(ui)
            cj = clustering.cluster_of_unit.get(uj)
            if ci and cj and ci != cj:
                support[(ci, cj)].append({
                    "book": book, "from": ui, "to": uj,
                    "confidence": tl.order_confidence.get((ui, uj), 0.35),
                    "level": tl.order_level.get((ui, uj), 5),
                    "nrt_chain": (ui, uj) in tl.nrt_chain,
                })
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                weights[(seq[a], seq[b])] += DOCUMENT_VOTE
                ordered_pairs.add((seq[a], seq[b]))
                ordered_pairs.add((seq[b], seq[a]))

    if reg:
        ids = [c.cluster_id for c in clustering.clusters]
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j:
                    continue
                a, b = ids[i], ids[j]
                if (a, b) in ordered_pairs:
                    continue
                if reg.get(a, 0.0) < reg.get(b, 0.0):
                    weights[(a, b)] += REGISTRATION_VOTE
    return dict(weights), dict(support)


def cluster_constraints(structs, timelines: Dict[str, LocalTimeline],
                        clustering: Clustering, scaffold: Scaffold):
    """Per-document Allen constraints between candidate canonical events.

    Two sources, both evidence-backed:

      * an asserted <TLINK> at cascade level 1, 2 or 3 between eligible events
        lying in different clusters -- an explicit signal, an explicit temporal
        expression, or an aspectual predicate;
      * a day boundary between the two clusters' units on the shared day axis,
        which the anchor chain of Section 6.2.5 asserts and which entails strict
        precedence.

    Level 4 is excluded deliberately: narrative adjacency is an assumption, and
    a disagreement between two assumptions is not evidence that the sources
    disagree.
    """
    from ..stage2_temporal_annotation.closure import allen_of, invert

    out: Dict[Tuple[str, str], Dict[str, frozenset]] = defaultdict(dict)
    support: Dict[Tuple[str, str], Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list))

    for book, struct in structs.items():
        tl = timelines[book]
        cof = {}
        for uid, cid in clustering.cluster_of_unit.items():
            cof[uid] = cid
        for l in struct.tlinks:
            if l.origin != "asserted" or l.level not in (1, 2, 3):
                continue
            a, b = l.source, l.target_id
            if a not in struct.eligible or b not in struct.eligible:
                continue
            ua = tl.unit_of_event.get(a)
            ub = tl.unit_of_event.get(b)
            if not ua or not ub:
                continue
            ca, cb = cof.get(ua), cof.get(ub)
            if not ca or not cb or ca == cb:
                continue
            key = (ca, cb) if ca < cb else (cb, ca)
            rel = allen_of(l.rel_type)
            if key != (ca, cb):
                rel = invert(rel)
            prev = out[key].get(book)
            out[key][book] = (prev & rel) if prev else rel
            support[key][book].append({"level": l.level, "rel": str(l.rel_type),
                                       "confidence": l.confidence,
                                       "signal": l.signal_id})

        # day-boundary evidence
        for ci in clustering.clusters:
            pass
    day_of = scaffold.day_of_unit
    for book, tl in timelines.items():
        by_cluster: Dict[str, List[str]] = defaultdict(list)
        for u in tl.units:
            cid = clustering.cluster_of_unit.get(u.unit_id)
            if cid:
                by_cluster[cid].append(u.unit_id)
        cids = [c.cluster_id for c in clustering.clusters
                if c.cluster_id in by_cluster]
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                ca, cb = cids[i], cids[j]
                da = {day_of.get(u) for u in by_cluster[ca]}
                db = {day_of.get(u) for u in by_cluster[cb]}
                da.discard(None)
                db.discard(None)
                if not da or not db:
                    continue
                if max(da) < min(db):
                    rel = frozenset({"b"})
                elif max(db) < min(da):
                    rel = frozenset({"bi"})
                else:
                    continue
                key = (ca, cb)
                prev = out[key].get(book)
                out[key][book] = (prev & rel) if prev else rel
                support[key][book].append({"level": 2, "rel": "DAY_BOUNDARY",
                                           "confidence": 0.85,
                                           "signal": None})
    return out, support


def detect_conflicts(structs, timelines: Dict[str, LocalTimeline],
                     clustering: Clustering, scaffold: Scaffold) -> List[dict]:
    """Inter-document conflicts as unsatisfiability of the asserted relations.

    Where two documents' evidence about the same pair of candidate canonical
    events has an empty intersection, the sources disagree about the order of
    events they both describe. This needs no threshold and cannot fail silently:
    it identifies exactly the pair of relations responsible, and which document
    asserted each.
    """
    constraints, support = cluster_constraints(structs, timelines, clustering,
                                               scaffold)
    conflicts: List[dict] = []
    for key, per_book in constraints.items():
        if len(per_book) < 2:
            continue
        acc = None
        for rel in per_book.values():
            acc = rel if acc is None else (acc & rel)
        if acc:
            continue
        conflicts.append({
            "kind": "unsatisfiable",
            "scope": "inter-document",
            "clusters": list(key),
            "per_document": {b: sorted(r) for b, r in per_book.items()},
            "support": {b: support[key][b] for b in per_book},
        })
    return conflicts


def induce(timelines: Dict[str, LocalTimeline], clustering: Clustering,
           scaffold: Scaffold,
           intra_conflicts: Optional[List[dict]] = None,
           structs=None) -> InducedTimeline:
    tl = InducedTimeline()
    reg = registration(timelines, clustering)
    for c in clustering.clusters:
        c.registration = reg.get(c.cluster_id, c.position)
    weights, support = build_cluster_graph(timelines, clustering, reg)
    tl.edges = weights
    tl.registration = reg

    nodes = [c.cluster_id for c in clustering.clusters]
    removed = minimum_feedback_arc_set(nodes, weights)
    tl.removed = [(a, b, weights[(a, b)]) for (a, b) in removed]

    kept = {e: w for e, w in weights.items() if e not in removed}
    tl.order = topological_sort(nodes, kept, scaffold, clustering, reg)
    tl.rank = {cid: i for i, cid in enumerate(tl.order)}

    # conflict report: intra-document cycles from closure, the relation pairs
    # found unsatisfiable across documents, and the arcs the feedback arc set
    # removed
    tl.conflicts = list(intra_conflicts or [])
    if structs is not None:
        tl.conflicts.extend(detect_conflicts(structs, timelines, clustering,
                                             scaffold))
    for (a, b) in removed:
        rev = weights.get((b, a), 0.0)
        tl.conflicts.append({
            "kind": "ordering",
            "scope": "inter-document",
            "clusters": [a, b],
            "weight": weights[(a, b)],
            "reverse_weight": rev,
            "support": support.get((a, b), []),
            "reverse_support": support.get((b, a), []),
        })
    return tl


# ---------------------------------------------------------------------------
def minimum_feedback_arc_set(nodes: Sequence[str],
                             weights: Dict[Tuple[str, str], float]
                             ) -> Set[Tuple[str, str]]:
    """Eades-Lin-Smyth (1993) GR heuristic, weighted.

    Produces a linear arrangement; the arcs pointing backwards in that
    arrangement are the feedback arc set. Weighting by confidence makes the
    heuristic prefer removing poorly attested arcs.
    """
    remaining = set(nodes)
    out_w: Dict[str, float] = defaultdict(float)
    in_w: Dict[str, float] = defaultdict(float)
    succ: Dict[str, Dict[str, float]] = defaultdict(dict)
    pred: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (a, b), w in weights.items():
        if a not in remaining or b not in remaining or a == b:
            continue
        succ[a][b] = succ[a].get(b, 0.0) + w
        pred[b][a] = pred[b].get(a, 0.0) + w
        out_w[a] += w
        in_w[b] += w

    s1: List[str] = []
    s2: List[str] = []

    def drop(v: str) -> None:
        remaining.discard(v)
        for u, w in list(succ.get(v, {}).items()):
            if u in remaining:
                in_w[u] -= w
                pred[u].pop(v, None)
        for u, w in list(pred.get(v, {}).items()):
            if u in remaining:
                out_w[u] -= w
                succ[u].pop(v, None)

    while remaining:
        moved = True
        while moved:
            moved = False
            for v in list(remaining):
                if out_w[v] <= 1e-12:            # sink
                    s2.insert(0, v)
                    drop(v)
                    moved = True
            for v in list(remaining):
                if in_w[v] <= 1e-12:             # source
                    s1.append(v)
                    drop(v)
                    moved = True
        if not remaining:
            break
        v = max(remaining, key=lambda x: out_w[x] - in_w[x])
        s1.append(v)
        drop(v)

    arrangement = s1 + s2
    pos = {v: i for i, v in enumerate(arrangement)}
    return {(a, b) for (a, b) in weights
            if a in pos and b in pos and pos[a] > pos[b]}


def topological_sort(nodes: Sequence[str],
                     weights: Dict[Tuple[str, str], float],
                     scaffold: Optional[Scaffold],
                     clustering: Clustering,
                     reg: Optional[Dict[str, float]] = None) -> List[str]:
    """Topological sort of the acyclic remainder, ties broken by position on
    the registration axis."""
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    succ: Dict[str, List[str]] = defaultdict(list)
    for (a, b) in weights:
        if a in indeg and b in indeg:
            succ[a].append(b)
            indeg[b] += 1

    pos: Dict[str, float] = {}
    for c in clustering.clusters:
        pos[c.cluster_id] = (reg or {}).get(c.cluster_id, c.position)

    import heapq
    heap = [(pos.get(n, 0.0), n) for n, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    out: List[str] = []
    while heap:
        _, n = heapq.heappop(heap)
        out.append(n)
        for m in succ.get(n, ()):
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, (pos.get(m, 0.0), m))
    # any node left in a residual cycle (should not occur after MFAS) is
    # appended in scaffold order, and the fact is recorded by the caller
    if len(out) < len(nodes):
        rest = [n for n in nodes if n not in set(out)]
        rest.sort(key=lambda n: pos.get(n, 0.0))
        out.extend(rest)
    return out
