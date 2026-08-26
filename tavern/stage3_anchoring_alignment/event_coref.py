"""
Stage 3 - cross-document event coreference (thesis Section 6.4.2).

Candidate pairs are scored on four signals, each derived from the annotation
rather than from surface text alone:

  * anchor compatibility  -- two units in the same interval of the scaffold are
    candidates; two units separated by an anchor are not. Available only
    because the anchoring of Section 6.2.5 was performed.
  * predicate and argument similarity -- the @pred inventories under an
    inverse-document-frequency weighting, and the participant sets from the
    Stage 1 entity chains. A shared CRUCIFY says more than a shared SAY, and
    the weighting is what expresses that.
  * modal context compatibility -- a narrative-world event is not coreferent
    with an event inside a prophecy, even where the predicates match. This has
    practical bite in a corpus that repeatedly predicts events that later
    occur: the prediction and the fulfilment are distinct events.
  * event class and aspect agreement -- disagreement between a FUTURE and a
    PAST mention is evidence of the prediction/fulfilment relation rather than
    of identity.

Clusters are formed pairwise and then merged. The pairwise step is a
band-constrained monotone sequence alignment rather than greedy nearest-
neighbour matching, because the local partial orders of Section 6.3.3 are
themselves evidence: two parallel accounts of a week do not cross each other
wholesale, so an alignment that respects both documents' orders is preferred to
one that does not. The band comes from the anchor scaffold, which is what makes
the alignment tractable and is the role Section 6.4.1 assigns it. The merge
carries the transitivity constraint that a cluster holds at most one unit per
document.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .local_timeline import EventUnit, LocalTimeline
from .scaffold import Scaffold

WEIGHTS = {
    "predicate": 0.40,
    "participants": 0.25,
    "anchor": 0.15,
    "modal": 0.10,
    "class": 0.10,
}

#: Minimum score for a pair to be admitted as coreferent.
MATCH_THRESHOLD = 0.34

#: Band half-width on the shared day axis. Pairs further apart than this are
#: separated by an anchor and are not candidates.
ANCHOR_BAND = 1.25

#: Gap cost in the alignment. Set below the threshold so that a weak match is
#: never preferred to leaving both units unaligned.
GAP_COST = 0.0

#: Predicates too frequent to discriminate; retained but down-weighted by IDF.
_UBIQUITOUS_ENTITIES = {"JESUS", "DISCIPLES"}


@dataclass
class Cluster:
    cluster_id: str
    members: List[str] = field(default_factory=list)
    books: Set[str] = field(default_factory=set)
    position: float = 0.0
    anchor_interval: int = -1
    registration: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class Clustering:
    clusters: List[Cluster] = field(default_factory=list)
    cluster_of_unit: Dict[str, str] = field(default_factory=dict)
    scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pair_matches: int = 0

    def by_id(self, cid: str) -> Optional[Cluster]:
        return next((c for c in self.clusters if c.cluster_id == cid), None)

    def contested(self) -> List[Cluster]:
        return [c for c in self.clusters if c.size > 1]


# ---------------------------------------------------------------------------
class PredicateIDF:
    """Inverse document frequency over the @pred inventory, with units as the
    documents. Derived entirely from the annotation."""

    def __init__(self, units: Sequence[EventUnit]):
        df: Counter = Counter()
        for u in units:
            df.update(set(u.preds))
            df.update({f"T:{t}" for t in u.timex_preds})
        self.n = max(1, len(units))
        self.df = df

    def weight(self, term: str) -> float:
        return math.log((self.n + 1) / (1 + self.df.get(term, 0)))

    def vector(self, u: EventUnit) -> Dict[str, float]:
        terms = Counter(u.preds)
        for t in u.timex_preds:
            terms[f"T:{t}"] += 1
        vec = {t: (1.0 + math.log(c)) * self.weight(t)
               for t, c in terms.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


# ---------------------------------------------------------------------------
def cluster_units(timelines: Dict[str, LocalTimeline], scaffold: Scaffold,
                  embeddings: Optional[Dict[str, Sequence[float]]] = None
                  ) -> Clustering:
    units: Dict[str, EventUnit] = {}
    for tl in timelines.values():
        for u in tl.units:
            units[u.unit_id] = u

    idf = PredicateIDF(list(units.values()))
    vectors = {uid: idf.vector(u) for uid, u in units.items()}

    books = sorted(timelines)
    matches: List[Tuple[float, str, str]] = []
    for i in range(len(books)):
        for j in range(i + 1, len(books)):
            matches.extend(_align(timelines[books[i]], timelines[books[j]],
                                  scaffold, vectors, embeddings))

    matches.sort(reverse=True)

    clustering = Clustering(pair_matches=len(matches))
    parent: Dict[str, str] = {u: u for u in units}
    bookset: Dict[str, Set[str]] = {u: {units[u].book} for u in units}
    members: Dict[str, Set[str]] = {u: {u} for u in units}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, x, y in matches:
        clustering.scores[(x, y)] = s
        rx, ry = find(x), find(y)
        if rx == ry or (bookset[rx] & bookset[ry]):
            continue
        parent[ry] = rx
        bookset[rx] |= bookset[ry]
        members[rx] |= members[ry]

    groups: Dict[str, List[str]] = defaultdict(list)
    for u in units:
        groups[find(u)].append(u)

    def cluster_position(group: Sequence[str]) -> float:
        ps = [scaffold.position_of(u) for u in group]
        ps = [p for p in ps if p is not None]
        return sum(ps) / len(ps) if ps else 0.0

    ordered = sorted(groups.values(), key=cluster_position)
    for n, group in enumerate(ordered, start=1):
        group.sort(key=lambda u: (books.index(units[u].book), u))
        cid = f"c{n:03d}"
        ivs = [scaffold.interval_of(u) for u in group]
        ivs = [v for v in ivs if v is not None]
        cl = Cluster(cluster_id=cid, members=group,
                     books={units[u].book for u in group},
                     position=cluster_position(group),
                     anchor_interval=min(ivs) if ivs else -1)
        clustering.clusters.append(cl)
        for u in group:
            clustering.cluster_of_unit[u] = cid
    return clustering


# ---------------------------------------------------------------------------
def _align(ta: LocalTimeline, tb: LocalTimeline, scaffold: Scaffold,
           vectors, embeddings) -> List[Tuple[float, str, str]]:
    """Band-constrained monotone alignment of two documents' unit sequences.

    A Needleman-Wunsch recursion over the two sequences, with the substitution
    score given by `score` and cells outside the scaffold band forbidden. The
    traceback yields a set of pairs that respects both documents' orders.
    """
    A, B = ta.units, tb.units
    na, nb = len(A), len(B)
    if not na or not nb:
        return []

    pa = [scaffold.position_of(u.unit_id) for u in A]
    pb = [scaffold.position_of(u.unit_id) for u in B]

    NEG = -1e9
    prev = [0.0] * (nb + 1)
    ptr: List[List[int]] = [[0] * (nb + 1) for _ in range(na + 1)]
    sub_cache: Dict[Tuple[int, int], float] = {}

    for i in range(1, na + 1):
        cur = [0.0] * (nb + 1)
        ptr[i][0] = 1
        for j in range(1, nb + 1):
            if i == 1:
                ptr[0][j] = 2
            s = _banded_score(A[i - 1], B[j - 1], pa[i - 1], pb[j - 1],
                              scaffold, vectors, embeddings)
            diag = prev[j - 1] + (s if s >= MATCH_THRESHOLD else NEG)
            up = prev[j] - GAP_COST
            left = cur[j - 1] - GAP_COST
            best = max(diag, up, left)
            cur[j] = best
            if best == diag:
                ptr[i][j] = 0
                sub_cache[(i, j)] = s
            elif best == up:
                ptr[i][j] = 1
            else:
                ptr[i][j] = 2
        prev = cur

    out: List[Tuple[float, str, str]] = []
    i, j = na, nb
    while i > 0 and j > 0:
        d = ptr[i][j]
        if d == 0:
            s = sub_cache.get((i, j), 0.0)
            if s >= MATCH_THRESHOLD:
                out.append((s, A[i - 1].unit_id, B[j - 1].unit_id))
            i -= 1
            j -= 1
        elif d == 1:
            i -= 1
        else:
            j -= 1
    return out


def _banded_score(a: EventUnit, b: EventUnit, pa: Optional[float],
                  pb: Optional[float], scaffold: Scaffold, vectors,
                  embeddings) -> float:
    if pa is not None and pb is not None and abs(pa - pb) > ANCHOR_BAND:
        return -1.0
    return score(a, b, scaffold, vectors, embeddings)


# ---------------------------------------------------------------------------
def score(a: EventUnit, b: EventUnit, scaffold: Scaffold, vectors,
          embeddings=None) -> float:
    return (WEIGHTS["predicate"] * predicate_similarity(a, b, vectors,
                                                        embeddings)
            + WEIGHTS["participants"] * participant_similarity(a, b)
            + WEIGHTS["anchor"] * anchor_compatibility(a, b, scaffold)
            + WEIGHTS["modal"] * modal_compatibility(a, b)
            + WEIGHTS["class"] * class_agreement(a, b))


def anchor_compatibility(a: EventUnit, b: EventUnit,
                         scaffold: Scaffold) -> float:
    pa = scaffold.position_of(a.unit_id)
    pb = scaffold.position_of(b.unit_id)
    if pa is None or pb is None:
        return 0.4
    d = abs(pa - pb)
    if d > ANCHOR_BAND:
        return 0.0
    return max(0.0, 1.0 - d / ANCHOR_BAND)


def predicate_similarity(a: EventUnit, b: EventUnit, vectors,
                         embeddings=None) -> float:
    val = _cosine(vectors[a.unit_id], vectors[b.unit_id])
    if embeddings is not None:
        ea, eb = embeddings.get(a.unit_id), embeddings.get(b.unit_id)
        if ea is not None and eb is not None:
            num = sum(x * y for x, y in zip(ea, eb))
            da = math.sqrt(sum(x * x for x in ea))
            db = math.sqrt(sum(y * y for y in eb))
            if da and db:
                val = 0.6 * val + 0.4 * max(0.0, num / (da * db))
    return val


def participant_similarity(a: EventUnit, b: EventUnit) -> float:
    if not a.entities or not b.entities:
        return 0.0
    jac = len(a.entities & b.entities) / len(a.entities | b.entities)
    sa = a.entities - _UBIQUITOUS_ENTITIES
    sb = b.entities - _UBIQUITOUS_ENTITIES
    if sa and sb:
        sj = len(sa & sb) / len(sa | sb)
        return 0.35 * jac + 0.65 * sj
    return jac


def modal_compatibility(a: EventUnit, b: EventUnit) -> float:
    """A narrative-world unit is not coreferent with a unit inside a prophecy."""
    if (a.eligible_fraction > 0.5) != (b.eligible_fraction > 0.5):
        return 0.0
    ta, tb = a.modal_types, b.modal_types
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.3
    return len(ta & tb) / len(ta | tb)


def class_agreement(a: EventUnit, b: EventUnit) -> float:
    """Coreferent mentions should agree in event class; disagreement between a
    FUTURE and a PAST mention is evidence of prediction/fulfilment rather than
    identity."""
    fut_a = "FUTURE" in a.tenses and "PAST" not in a.tenses
    fut_b = "FUTURE" in b.tenses and "PAST" not in b.tenses
    if fut_a != fut_b:
        return 0.0
    if not (a.classes or b.classes):
        return 0.0
    return len(a.classes & b.classes) / len(a.classes | b.classes)
