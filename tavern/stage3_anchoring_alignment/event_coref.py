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

Clusters are formed by PROGRESSIVE PROFILE ALIGNMENT, not by independent
pairwise matching. The local partial orders of Section 6.3.3 are themselves
evidence: four parallel accounts of one week do not cross each other wholesale,
so an alignment respecting every document's order is preferred to one that does
not. Documents are added to a growing profile in decreasing order of length,
each by a band-constrained monotone alignment against the profile built so far;
the band comes from the anchor scaffold, which is the role Section 6.4.1
assigns it.

The reason for a profile rather than six pairwise alignments is not efficiency
but consistency. Independent pairwise alignments need not agree, and merging
them transitively produces clusters that violate the documents' own orders --
which then appear as cycles in the cluster graph and are resolved arbitrarily.
A profile is a single consistent ordered structure by construction, and its
column order is a strong registration signal for the induction of
Section 6.4.3. The transitivity constraint that a cluster holds at most one
unit per document is enforced by the profile's shape rather than added
afterwards.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from ..config import BOOK_ORDER
from .local_timeline import EventUnit, LocalTimeline
from .scaffold import Scaffold

WEIGHTS = {
    "predicate": 0.40,
    "participants": 0.25,
    "anchor": 0.15,
    "modal": 0.10,
    "class": 0.10,
}

#: H-A (structural, argument-driven -- see thesis discussion of cluster
#: purity). class_agreement and modal_compatibility as WEIGHTS-weighted
#: additive terms reward AGREEMENT as though it were positive evidence of
#: coreference, when the only real signal in either is DISAGREEMENT --
#: FUTURE-vs-PAST is a prediction/fulfilment pair, not the same event; a
#: narrative-world unit is not coreferent with a quoted one. When this is on,
#: score() drops the anchor term, renormalises predicate/participants to sum
#: to 1 (0.62/0.38, i.e. 0.40/0.65 and 0.25/0.65), and multiplies by two
#: {0,1} gates that fire only on that disagreement, never on agreement or on
#: absence of evidence.
GATED_SCORE = False

#: H-B (structural; reformulated -- see Addendum 4). Originally targeted
#: anchor_compatibility's 0.4 floor for a unit with no scaffold position, on
#: the premise that this rewards absent information. It does not: every unit
#: in a document gets an interpolated position unconditionally
#: (scaffold._solve_document), so position_of(uid) is not None for every real
#: unit and that floor is unreachable -- confirmed by instrumenting score()
#: over a full run (177,143 calls, 0 hit it). The real problem is one level
#: down: anchor_compatibility computes a distance between two positions
#: without asking whether either was OBSERVED (pinned by a unit's own anchor)
#: or INTERPOLATED (borrowed from its neighbours' pins), so a confident-
#: looking distance can be measured between two fabricated values. When this
#: is on, the anchor term contributes only when BOTH units are
#: scaffold.is_observed(); when either is interpolated, its 0.15 weight is
#: redistributed into predicate/participants for that pair, same as before.
#: A no-op under GATED_SCORE, which has already dropped the anchor term.
NO_ANCHOR_CREDIT = False

_NO_ANCHOR_WEIGHTS = (
    WEIGHTS["predicate"] + WEIGHTS["anchor"] * WEIGHTS["predicate"]
    / (WEIGHTS["predicate"] + WEIGHTS["participants"]),
    WEIGHTS["participants"] + WEIGHTS["anchor"] * WEIGHTS["participants"]
    / (WEIGHTS["predicate"] + WEIGHTS["participants"]),
)

#: Minimum score for a pair to be admitted as coreferent.
MATCH_THRESHOLD = 0.34

#: Band half-width on the shared day axis. Pairs further apart than this are
#: separated by an anchor and are not candidates. The scaffold's day resolution
#: is coarse relative to the density of the Passion day, so the band is set
#: generously: its function is to exclude gross misalignment, not to decide it.
ANCHOR_BAND = 4.0

#: Gap cost in the alignment. Small and positive: with free gaps the recursion
#: is indifferent among equal-scoring paths, and an unmatched block of one
#: document can land anywhere monotonicity permits. A small cost makes the
#: alignment advance both sequences in proportion, which registers documents of
#: unequal length against each other.
GAP_COST = 0.06

#: Predicates too frequent to discriminate; retained but down-weighted by IDF.
_UBIQUITOUS_ENTITIES = {"JESUS", "DISCIPLES"}

#: Largest span, in verses per document, that one candidate canonical event may
#: cover. Bounds the episode merge below. `None` disables this bound, leaving
#: the pericope boundary (below) as the only span limit.
MAX_EPISODE_VERSES: Optional[int] = 2

#: Override for the profile seed order (see `cluster_units`); None = default.
_SEED_ORDER = None


@dataclass
class Cluster:
    cluster_id: str
    members: List[str] = field(default_factory=list)
    books: Set[str] = field(default_factory=set)
    position: float = 0.0
    anchor_interval: int = -1
    registration: float = 0.0
    profile_index: float = 0.0

    @property
    def size(self) -> int:
        """Number of documents reporting this candidate canonical event."""
        return len(self.books)

    @property
    def n_units(self) -> int:
        return len(self.members)

    def spans(self, units) -> Dict[str, List[str]]:
        """The cluster's contribution per document: a contiguous span of units,
        in document order. This is the same shape as a canonical event, which
        cites a verse range per Gospel."""
        out: Dict[str, List[str]] = {}
        for uid in self.members:
            out.setdefault(units[uid].book, []).append(uid)
        return out


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

    # Documents enter the profile in canonical order. The choice has a
    # measurable but small effect, reported as a sensitivity analysis in
    # Section 10.4: over the five orders tested, tau ranges from 0.898 to 0.922
    # and downstream ROUGE-L from 0.524 to 0.595. Canonical order is adopted
    # because it is fixed independently of any measurement.
    books = (_SEED_ORDER(timelines) if _SEED_ORDER is not None
             else [b for b in BOOK_ORDER if b in timelines]
             or sorted(timelines))
    clustering = Clustering()

    profile: List[List[str]] = [[u.unit_id] for u in timelines[books[0]].units]
    for book in books[1:]:
        profile = _add_to_profile(profile, timelines[book].units, units,
                                  scaffold, vectors, embeddings, clustering)

    profile = _merge_episodes(profile, units, timelines)

    for n, column in enumerate(profile, start=1):
        cid = f"c{n:03d}"
        ivs = [scaffold.interval_of(u) for u in column]
        ivs = [v for v in ivs if v is not None]
        ps = [scaffold.position_of(u) for u in column]
        ps = [p for p in ps if p is not None]
        cl = Cluster(cluster_id=cid, members=list(column),
                     books={units[u].book for u in column},
                     position=(sum(ps) / len(ps)) if ps else 0.0,
                     anchor_interval=min(ivs) if ivs else -1,
                     profile_index=(n - 1) / max(1, len(profile) - 1))
        clustering.clusters.append(cl)
        for u in column:
            clustering.cluster_of_unit[u] = cid
    return clustering


def _merge_episodes(profile: List[List[str]], units: Dict[str, EventUnit],
                    timelines: Dict[str, LocalTimeline]) -> List[List[str]]:
    """Merge adjacent profile columns into candidate canonical events.

    Alignment operates on event units, which are finer than the episodes a
    harmony treats as single events: a canonical event covers a verse SPAN in
    each document that reports it, not a single unit. Columns are therefore
    merged while each document's contribution stays a contiguous run of its own
    units, and the merge stops at the evidence that a new event has begun -- an
    anchorable temporal expression, a pericope boundary, or the span bound of
    `MAX_EPISODE_VERSES`.

    A merged cluster holds at most one CONTIGUOUS SPAN per document, which is
    the same shape as a canonical event and is what makes version selection
    select an account rather than a fragment of one.
    """
    succ: Dict[str, Optional[str]] = {}
    for tl in timelines.values():
        for a, b in zip(tl.units, tl.units[1:]):
            succ[a.unit_id] = b.unit_id
        if tl.units:
            succ[tl.units[-1].unit_id] = None

    out: List[List[str]] = []
    for column in profile:
        if not out:
            out.append(list(column))
            continue
        cur = out[-1]
        if _mergeable(cur, column, units, succ):
            cur.extend(column)
        else:
            out.append(list(column))
    return out


def _mergeable(cur: Sequence[str], nxt: Sequence[str],
               units: Dict[str, EventUnit],
               succ: Dict[str, Optional[str]]) -> bool:
    last_of: Dict[str, str] = {}
    verses_of: Dict[str, int] = {}
    for uid in cur:
        b = units[uid].book
        last_of[b] = uid
        verses_of[b] = verses_of.get(b, 0) + len(units[uid].verse_keys)

    for uid in nxt:
        u = units[uid]
        # an anchorable temporal expression opens a new event
        if u.anchorable_timex:
            return False
        prev = last_of.get(u.book)
        if prev is not None:
            if succ.get(prev) != uid:
                return False                      # would break contiguity
            if units[prev].pericope_id != u.pericope_id:
                return False                      # pericope boundary
            if (MAX_EPISODE_VERSES is not None and
                    verses_of.get(u.book, 0) + len(u.verse_keys)
                    > MAX_EPISODE_VERSES):
                return False
    return True



def _add_to_profile(profile: List[List[str]], new_units: Sequence[EventUnit],
                    units: Dict[str, EventUnit], scaffold: Scaffold, vectors,
                    embeddings, clustering: Clustering) -> List[List[str]]:
    """Monotone alignment of one document's units against the profile.

    A Needleman-Wunsch recursion in which the substitution score of a unit
    against a profile column is the mean of its scores against that column's
    members. Gaps are free in both directions: a column no document but one
    describes is legitimate, and so is a unit no column matches.
    """
    m, k = len(profile), len(new_units)
    if not k:
        return profile
    if not m:
        return [[u.unit_id] for u in new_units]

    NEG = -1e9
    col_pos = []
    for column in profile:
        ps = [scaffold.position_of(u) for u in column]
        ps = [p for p in ps if p is not None]
        col_pos.append(sum(ps) / len(ps) if ps else None)
    new_pos = [scaffold.position_of(u.unit_id) for u in new_units]

    sub: Dict[Tuple[int, int], float] = {}

    def substitution(i: int, j: int) -> float:
        key = (i, j)
        if key in sub:
            return sub[key]
        pa, pb = col_pos[i], new_pos[j]
        if pa is not None and pb is not None and abs(pa - pb) > ANCHOR_BAND:
            sub[key] = -1.0
            return -1.0
        vals = [score(units[mem], new_units[j], scaffold, vectors, embeddings)
                for mem in profile[i]]
        s = sum(vals) / len(vals) if vals else 0.0
        sub[key] = s
        return s

    dp = [[0.0] * (k + 1) for _ in range(m + 1)]
    ptr = [[0] * (k + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ptr[i][0] = 1
    for j in range(1, k + 1):
        ptr[0][j] = 2
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            s = substitution(i - 1, j - 1)
            diag = dp[i - 1][j - 1] + (s if s >= MATCH_THRESHOLD else NEG)
            up = dp[i - 1][j] - GAP_COST
            left = dp[i][j - 1] - GAP_COST
            best = max(diag, up, left)
            dp[i][j] = best
            ptr[i][j] = 0 if best == diag else (1 if best == up else 2)

    merged: List[List[str]] = []
    i, j = m, k
    while i > 0 or j > 0:
        d = ptr[i][j] if (i > 0 and j > 0) else (1 if i > 0 else 2)
        if d == 0:
            col = list(profile[i - 1]) + [new_units[j - 1].unit_id]
            merged.append(col)
            s = sub.get((i - 1, j - 1), 0.0)
            for mem in profile[i - 1]:
                clustering.scores[(mem, new_units[j - 1].unit_id)] = s
            i -= 1
            j -= 1
        elif d == 1:
            merged.append(list(profile[i - 1]))
            i -= 1
        else:
            merged.append([new_units[j - 1].unit_id])
            j -= 1
    merged.reverse()
    return merged


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
    pred = predicate_similarity(a, b, vectors, embeddings)
    part = participant_similarity(a, b)

    if GATED_SCORE:                                                # H-A
        base = 0.62 * pred + 0.38 * part
        return base * _class_gate(a, b) * _modal_gate(a, b)

    if NO_ANCHOR_CREDIT:                                            # H-B
        if not (scaffold.is_observed(a.unit_id)
                and scaffold.is_observed(b.unit_id)):
            w_pred, w_part = _NO_ANCHOR_WEIGHTS
            return (w_pred * pred + w_part * part
                    + WEIGHTS["modal"] * modal_compatibility(a, b)
                    + WEIGHTS["class"] * class_agreement(a, b))

    return (WEIGHTS["predicate"] * pred
            + WEIGHTS["participants"] * part
            + WEIGHTS["anchor"] * anchor_compatibility(a, b, scaffold)
            + WEIGHTS["modal"] * modal_compatibility(a, b)
            + WEIGHTS["class"] * class_agreement(a, b))


def _class_gate(a: EventUnit, b: EventUnit) -> float:
    """H-A: class agreement as a veto, not a score.

    1.0 unless tense polarity flips (FUTURE vs PAST/present) -- real evidence
    of a prediction/fulfilment pair, not of identity. Agreement, or the
    absence of any class evidence on either side, passes.
    """
    fut_a = "FUTURE" in a.tenses and "PAST" not in a.tenses
    fut_b = "FUTURE" in b.tenses and "PAST" not in b.tenses
    return 0.0 if fut_a != fut_b else 1.0


def _modal_gate(a: EventUnit, b: EventUnit) -> float:
    """H-A: modal agreement as a veto.

    1.0 unless the two units sit in different narrative worlds (one timeline-
    eligible, one not). Agreement, or absence of modal evidence, passes.
    """
    return (0.0 if (a.eligible_fraction > 0.5) != (b.eligible_fraction > 0.5)
            else 1.0)


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


# ---------------------------------------------------------------------------
def detect_order_conflicts(timelines: Dict[str, LocalTimeline],
                           scaffold: Scaffold,
                           embeddings=None) -> List[dict]:
    """Ordering disagreements between sources, as alignment crossings.

    The profile of `cluster_units` is monotone, so it cannot represent a
    disagreement about order: where two documents cross, a monotone alignment
    silently drops one of the competing matches. The crossings are therefore
    recovered here, before that resolution is imposed.

    For each pair of documents an UNCONSTRAINED one-to-one matching is computed
    greedily by the same score and the same threshold the alignment uses. Any
    inversion in that matching -- two matched pairs whose order is opposite in
    the two documents -- is a pair of episodes the two sources place in
    different relative order. This is the mechanism Section 6.3.2 describes:
    disagreement is unsatisfiability, it needs no threshold of its own, it
    cannot fail silently, and it names the specific episodes responsible.

    The fig tree is the case the harmonisation literature discusses: Matthew has
    the cursing, the withering and the cleansing of the temple in one order and
    Mark in another, so no monotone alignment can accommodate all three.
    """
    units: Dict[str, EventUnit] = {}
    for tl in timelines.values():
        for u in tl.units:
            units[u.unit_id] = u
    idf = PredicateIDF(list(units.values()))
    vectors = {uid: idf.vector(u) for uid, u in units.items()}

    books = sorted(timelines)
    out: List[dict] = []
    for i in range(len(books)):
        for j in range(i + 1, len(books)):
            a, b = books[i], books[j]
            out.extend(_pair_conflicts(timelines[a], timelines[b], scaffold,
                                       vectors, embeddings))
    return out


def _pair_conflicts(ta: LocalTimeline, tb: LocalTimeline, scaffold: Scaffold,
                    vectors, embeddings) -> List[dict]:
    pos_a = {u.unit_id: i for i, u in enumerate(ta.units)}
    pos_b = {u.unit_id: i for i, u in enumerate(tb.units)}

    cands: List[Tuple[float, str, str]] = []
    for ua in ta.units:
        pa = scaffold.position_of(ua.unit_id)
        for ub in tb.units:
            pb = scaffold.position_of(ub.unit_id)
            if pa is not None and pb is not None and abs(pa - pb) > ANCHOR_BAND:
                continue
            s = score(ua, ub, scaffold, vectors, embeddings)
            if s >= MATCH_THRESHOLD:
                cands.append((s, ua.unit_id, ub.unit_id))
    # Mutually best pairs only. A match used as evidence of DISAGREEMENT must
    # be one the content itself insists on, so each unit's best partner in the
    # other document must be the unit whose best partner it is. This is
    # threshold-free beyond the alignment's own threshold and it removes the
    # incidental matches that a greedy pass admits.
    best_a: Dict[str, Tuple[float, str]] = {}
    best_b: Dict[str, Tuple[float, str]] = {}
    for s, x, y in cands:
        if x not in best_a or s > best_a[x][0]:
            best_a[x] = (s, y)
        if y not in best_b or s > best_b[y][0]:
            best_b[y] = (s, x)
    matched: List[Tuple[float, str, str]] = [
        (s, x, y) for x, (s, y) in best_a.items()
        if best_b.get(y, (0.0, None))[1] == x]

    matched.sort(key=lambda m: pos_a[m[1]])
    if len(matched) < 2:
        return []

    # The matches that a monotone alignment cannot accommodate are exactly
    # those outside a maximum increasing subsequence of the second document's
    # positions. Reporting the excluded matches rather than every inverted PAIR
    # is what makes the count interpretable: it is the minimum number of
    # episodes on which the two sources disagree, not the number of
    # disagreeing pairs that follow from them.
    seq = [pos_b[m[2]] for m in matched]
    keep = _longest_increasing(seq)
    kept = set(keep)
    out: List[dict] = []
    for i, m in enumerate(matched):
        if i in kept:
            continue
        # the kept match this one crosses, for the report
        partner = None
        for k in keep:
            if (k < i and seq[k] > seq[i]) or (k > i and seq[k] < seq[i]):
                partner = k
                break
        if partner is None:
            continue
        out.append({
            "kind": "ordering",
            "scope": "inter-document",
            "books": [ta.book, tb.book],
            "units": [m[1], m[2], matched[partner][1], matched[partner][2]],
            "scores": [round(m[0], 3), round(matched[partner][0], 3)],
        })
    return out


def _longest_increasing(seq: Sequence[int]) -> List[int]:
    """Indices of a longest strictly increasing subsequence."""
    import bisect
    tails: List[int] = []          # values
    tails_idx: List[int] = []      # index in seq of each tail
    back: List[int] = [-1] * len(seq)
    for i, v in enumerate(seq):
        j = bisect.bisect_left(tails, v)
        if j == len(tails):
            tails.append(v)
            tails_idx.append(i)
        else:
            tails[j] = v
            tails_idx[j] = i
        back[i] = tails_idx[j - 1] if j else -1
    out: List[int] = []
    k = tails_idx[-1] if tails_idx else -1
    while k >= 0:
        out.append(k)
        k = back[k]
    out.reverse()
    return out
