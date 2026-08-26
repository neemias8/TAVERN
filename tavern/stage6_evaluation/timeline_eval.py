"""
Stage 6 - Kendall's tau and coverage (thesis Section 9.3.3).

Three specifications govern the computation, each responding to a documented
failure of a previous approach.

  * MATCHING IS BY VERSE-SPAN OVERLAP, NOT TEXT SIMILARITY. The heuristic
    sentence-to-event matcher is not used. Its failure mode on this corpus is
    documented: matching by TF-IDF similarity aligned the first canonical
    event, "Mary anoints Jesus", with a post-crucifixion verse mentioning the
    anointing of a body. A single false alignment propagates.
  * COVERAGE IS REPORTED WITH EVERY TAU. Kendall's tau is insensitive to
    omission, so a system recovering five events in the correct order scores
    well while having failed the task.
  * NO REFERENCE-BASED VARIANT IS THE PRIMARY FIGURE, because the reference
    consolidation was itself constructed following the curated chronology.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from scipy.stats import kendalltau

from ..stage3_anchoring_alignment.event_coref import Clustering
from ..stage3_anchoring_alignment.global_timeline import InducedTimeline
from ..stage3_anchoring_alignment.local_timeline import EventUnit
from .chronology import CanonicalEvent, Chronology


@dataclass
class TimelineEvaluation:
    tau: Optional[float]
    p_value: Optional[float]
    pairwise_accuracy: Optional[float]
    coverage: float
    matched_events: int
    total_events: int
    concordant: int = 0
    discordant: int = 0
    matching: Dict[int, str] = field(default_factory=dict)
    unmatched: List[int] = field(default_factory=list)
    overlap_quality: float = 0.0

    def as_row(self) -> dict:
        return {
            "tau": None if self.tau is None else round(self.tau, 4),
            "pairwise_accuracy": None if self.pairwise_accuracy is None
            else round(self.pairwise_accuracy, 4),
            "coverage": round(self.coverage, 4),
            "matched": self.matched_events,
            "total": self.total_events,
        }


def match_clusters_to_events(chronology: Chronology, clustering: Clustering,
                             units: Dict[str, EventUnit],
                             min_overlap: float = 0.10
                             ) -> Tuple[Dict[int, str], Dict[int, float]]:
    """Match induced clusters to canonical events by verse-span overlap.

    Overlap is the Jaccard index over the sets of book:chapter:verse addresses,
    which is exact -- no text similarity is involved anywhere. The assignment is
    greedy by descending overlap and one-to-one, so a cluster cannot stand for
    two canonical events.
    """
    cluster_keys: Dict[str, Set[Tuple[str, int, int]]] = {}
    for cl in clustering.clusters:
        keys: Set[Tuple[str, int, int]] = set()
        for m in cl.members:
            keys |= set(units[m].verse_keys)
        cluster_keys[cl.cluster_id] = keys

    scored: List[Tuple[float, int, str]] = []
    for ev in chronology.events:
        ekeys = set(ev.all_keys)
        if not ekeys:
            continue
        for cid, ckeys in cluster_keys.items():
            inter = ekeys & ckeys
            if not inter:
                continue
            jac = len(inter) / len(ekeys | ckeys)
            recall = len(inter) / len(ekeys)
            scored.append((0.5 * jac + 0.5 * recall, ev.event_id, cid))
    scored.sort(reverse=True)

    used_c: Set[str] = set()
    used_e: Set[int] = set()
    matching: Dict[int, str] = {}
    quality: Dict[int, float] = {}
    for s, eid, cid in scored:
        if eid in used_e or cid in used_c or s < min_overlap:
            continue
        matching[eid] = cid
        quality[eid] = s
        used_e.add(eid)
        used_c.add(cid)
    return matching, quality


def evaluate(chronology: Chronology, clustering: Clustering,
             induced: InducedTimeline, units: Dict[str, EventUnit]
             ) -> TimelineEvaluation:
    matching, quality = match_clusters_to_events(chronology, clustering, units)
    ranks = chronology.rank()

    pairs = [(ranks[eid], induced.rank.get(cid))
             for eid, cid in matching.items()
             if induced.rank.get(cid) is not None]
    pairs.sort()
    total = len([e for e in chronology.events if e.all_keys])
    coverage = len(matching) / max(1, total)

    if len(pairs) < 3:
        return TimelineEvaluation(None, None, None, coverage, len(matching),
                                  total, matching=matching)

    ref = [p[0] for p in pairs]
    hyp = [p[1] for p in pairs]
    tau, p = kendalltau(ref, hyp)

    conc = disc = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a = hyp[i] - hyp[j]
            if a == 0:
                continue
            if a < 0:
                conc += 1
            else:
                disc += 1
    pw = conc / (conc + disc) if (conc + disc) else None

    return TimelineEvaluation(
        tau=float(tau), p_value=float(p), pairwise_accuracy=pw,
        coverage=coverage, matched_events=len(matching), total_events=total,
        concordant=conc, discordant=disc, matching=matching,
        unmatched=[e.event_id for e in chronology.events
                   if e.all_keys and e.event_id not in matching],
        overlap_quality=(sum(quality.values()) / len(quality)) if quality else 0.0,
    )
