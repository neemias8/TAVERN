"""
Stage 3 - construction of the cross-document event graph
(thesis Sections 7.2-7.3, Tables tab:graph-features and tab:graph-edges).

A node is one event version. Every node feature except the span embedding is
derived from the ISO-TimeML annotation; every edge weight is the confidence
recorded on the underlying assertion, never a lexical similarity. That is the
design implication of the negative result of Section 7.1 taken up.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..stage2_temporal_annotation.enums import (Aspect, EventClass, EventType,
                                                POS, SLinkRel, Tense)
from ..stage2_temporal_annotation.model import AnnotationStructure
from .event_coref import Clustering
from .global_timeline import InducedTimeline
from .local_timeline import EventUnit, LocalTimeline
from .scaffold import Scaffold

EDGE_TYPES = [
    "INTRA_BEFORE", "INTRA_AFTER", "INTRA_INCLUDES", "INTRA_IS_INCLUDED",
    "SIMULTANEOUS", "SAME_EVENT", "INTER_BEFORE", "INTER_AFTER",
    "SUBORDINATES", "ASPECTUAL", "CONFLICT",
]

_CLASSES = [c.value for c in EventClass]
_TYPES = [t.value for t in EventType]
_TENSES = [t.value for t in Tense]
_ASPECTS = [a.value for a in Aspect]
_POS = [p.value for p in POS]
_SLINKS = [s.value for s in SLinkRel]
_BOOKS = ["matthew", "mark", "luke", "john"]

#: Dimensions of the non-embedding feature block, in the order of
#: Table tab:graph-features.
STRUCTURAL_DIM = (len(_CLASSES) + len(_TYPES) + len(_TENSES) + len(_ASPECTS)
                  + len(_POS) + 1 + 1 + len(_SLINKS) + 1 + 1 + 1
                  + len(_BOOKS) + 1 + 5)


@dataclass
class EventGraph:
    g: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    node_features: Dict[str, List[float]] = field(default_factory=dict)
    node_units: Dict[str, EventUnit] = field(default_factory=dict)
    embedding_dim: int = 0

    @property
    def n_nodes(self) -> int:
        return self.g.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.g.number_of_edges()

    def edge_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for _u, _v, d in self.g.edges(data=True):
            out[d["type"]] += 1
        return dict(out)

    def cluster_members(self, cid: str) -> List[str]:
        return [n for n, d in self.g.nodes(data=True) if d["cluster"] == cid]


def build(structs: Dict[str, AnnotationStructure],
          timelines: Dict[str, LocalTimeline], clustering: Clustering,
          induced: InducedTimeline, scaffold: Scaffold,
          embeddings: Optional[Dict[str, Sequence[float]]] = None
          ) -> EventGraph:
    eg = EventGraph()
    units: Dict[str, EventUnit] = {}
    for tl in timelines.values():
        for u in tl.units:
            units[u.unit_id] = u
    eg.node_units = units

    n_clusters = max(1, len(clustering.clusters))
    emb_dim = 0
    if embeddings:
        emb_dim = len(next(iter(embeddings.values())))
    eg.embedding_dim = emb_dim

    for uid, u in units.items():
        cid = clustering.cluster_of_unit.get(uid)
        rank = induced.rank.get(cid, 0) if cid else 0
        eg.g.add_node(uid, cluster=cid, book=u.book, ref=u.ref,
                      rank=rank, text=u.text,
                      verses=[f"{b}:{c}:{v}" for b, c, v in u.verse_keys])
        eg.node_features[uid] = _features(u, scaffold, rank / n_clusters,
                                          embeddings)

    # --- intra-document temporal edges -----------------------------------
    for book, tl in timelines.items():
        for (a, b) in tl.order:
            conf = tl.order_confidence.get((a, b), 0.35)
            lvl = tl.order_level.get((a, b), 5)
            asserted = 1.0 if lvl <= 4 else 0.0
            eg.g.add_edge(a, b, type="INTRA_BEFORE", weight=conf,
                          asserted=asserted, level=lvl)
            eg.g.add_edge(b, a, type="INTRA_AFTER", weight=conf,
                          asserted=asserted, level=lvl)

    # --- containment and simultaneity within a document -------------------
    for book, struct in structs.items():
        tl = timelines[book]
        for (i, j), rel in struct.closed_network.items():
            r = set(rel)
            ui = tl.unit_of_event.get(i)
            uj = tl.unit_of_event.get(j)
            if not ui or not uj or ui == uj:
                continue
            meta = struct.network_provenance.get((i, j), {})
            conf = meta.get("confidence", 0.35)
            asserted = 1.0 if meta.get("origin") == "asserted" else 0.0
            if r == {"d"}:
                eg.g.add_edge(ui, uj, type="INTRA_IS_INCLUDED", weight=conf,
                              asserted=asserted, level=meta.get("level", 5))
            elif r == {"di"}:
                eg.g.add_edge(ui, uj, type="INTRA_INCLUDES", weight=conf,
                              asserted=asserted, level=meta.get("level", 5))
            elif r == {"e"}:
                eg.g.add_edge(ui, uj, type="SIMULTANEOUS", weight=conf,
                              asserted=asserted, level=meta.get("level", 5))
                eg.g.add_edge(uj, ui, type="SIMULTANEOUS", weight=conf,
                              asserted=asserted, level=meta.get("level", 5))

        # --- subordination ------------------------------------------------
        for sl in struct.slinks:
            ug = tl.unit_of_event.get(sl.event_id)
            us = tl.unit_of_event.get(sl.subordinated_event)
            if not ug or not us or ug == us:
                continue
            eg.g.add_edge(ug, us, type="SUBORDINATES", weight=0.8,
                          asserted=1.0, level=1,
                          rel=str(sl.rel_type))

        # --- aspectual ----------------------------------------------------
        for al in struct.alinks:
            ug = tl.unit_of_event.get(al.event_id)
            ua = tl.unit_of_event.get(al.related_to_event)
            if not ug or not ua or ug == ua:
                continue
            eg.g.add_edge(ug, ua, type="ASPECTUAL", weight=0.65,
                          asserted=1.0, level=3, rel=str(al.rel_type))

    # --- SAME_EVENT (cluster membership) ---------------------------------
    for cl in clustering.clusters:
        for i in range(len(cl.members)):
            for j in range(i + 1, len(cl.members)):
                a, b = cl.members[i], cl.members[j]
                w = clustering.scores.get((a, b),
                                          clustering.scores.get((b, a), 0.6))
                eg.g.add_edge(a, b, type="SAME_EVENT", weight=w,
                              asserted=1.0, level=1)
                eg.g.add_edge(b, a, type="SAME_EVENT", weight=w,
                              asserted=1.0, level=1)

    # --- INTER_BEFORE / INTER_AFTER (cluster precedence in T-hat) ---------
    order = induced.order
    for k in range(len(order) - 1):
        ca, cb = order[k], order[k + 1]
        wa = induced.edges.get((ca, cb), 0.3)
        ma = clustering.by_id(ca)
        mb = clustering.by_id(cb)
        if ma is None or mb is None:
            continue
        for a in ma.members:
            for b in mb.members:
                eg.g.add_edge(a, b, type="INTER_BEFORE", weight=wa,
                              asserted=0.0, level=5)
                eg.g.add_edge(b, a, type="INTER_AFTER", weight=wa,
                              asserted=0.0, level=5)

    # --- CONFLICT ---------------------------------------------------------
    for cf in induced.conflicts:
        if cf.get("kind") not in ("ordering", "unsatisfiable"):
            continue
        groups: List[List[str]] = []
        for cid in (cf.get("clusters", ()) or ()):
            cl = clustering.by_id(cid)
            if cl is not None:
                groups.append(list(cl.members))
        uids = [u for u in (cf.get("units", ()) or ()) if u in units]
        if uids:
            groups.append(uids)
        flat = [u for grp in groups for u in grp]
        for x in range(len(flat)):
            for y in range(x + 1, len(flat)):
                a, b = flat[x], flat[y]
                if a == b or not eg.g.has_node(a) or not eg.g.has_node(b):
                    continue
                eg.g.add_edge(a, b, type="CONFLICT", weight=1.0,
                              asserted=1.0, level=1)
                eg.g.add_edge(b, a, type="CONFLICT", weight=1.0,
                              asserted=1.0, level=1)
    return eg


# ---------------------------------------------------------------------------
def _one_hot(value: str, space: Sequence[str]) -> List[float]:
    return [1.0 if value == s else 0.0 for s in space]


def _multi_hot(values: Set[str], space: Sequence[str]) -> List[float]:
    return [1.0 if s in values else 0.0 for s in space]


def _features(u: EventUnit, scaffold: Scaffold, norm_rank: float,
              embeddings=None) -> List[float]:
    feat: List[float] = []
    feat += _multi_hot(u.classes, _CLASSES)
    feat += _multi_hot(u.types, _TYPES)
    feat += _multi_hot(u.tenses, _TENSES)
    feat += _multi_hot(u.aspects, _ASPECTS)
    feat += _multi_hot(u.pos_tags, _POS)
    feat += [1.0 - u.neg_fraction]                      # @polarity aggregate
    feat += [u.modal_depth]
    feat += _multi_hot(u.modal_types, _SLINKS)
    feat += [u.eligible_fraction]
    iv = scaffold.interval_of(u.unit_id)
    feat += [float(iv if iv is not None else -1)]
    feat += [norm_rank]
    feat += _one_hot(u.book, _BOOKS)
    feat += [float(len(u.text.split()))]
    feat += list(u.level_profile)
    if embeddings is not None:
        e = embeddings.get(u.unit_id)
        if e is not None:
            feat = list(e) + feat
    return feat
