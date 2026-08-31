"""
Stage 3 - anchoring, alignment and graph construction.

Output is the pair (G, T-hat): the interface at which the symbolic half of the
system hands over to the neural half (thesis Section 5.4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..config import TavernConfig
from ..stage1_preprocessing.coref import EntityChains
from ..stage1_preprocessing.corpus import Corpus
from ..stage1_preprocessing.pericopes import PericopeLayer
from ..stage2_temporal_annotation.model import AnnotationStructure
from . import graph as graph_mod
from . import scaffold as scaffold_mod
from .event_coref import (Clustering, cluster_units,
                          detect_order_conflicts)
from .global_timeline import InducedTimeline, induce
from .local_timeline import LocalTimeline, segment_corpus

__all__ = ["Stage3Result", "run"]


@dataclass
class Stage3Result:
    timelines: Dict[str, LocalTimeline]
    scaffold: "scaffold_mod.Scaffold"
    clustering: Clustering
    induced: InducedTimeline
    graph: "graph_mod.EventGraph"
    projection: Optional[Dict[str, int]] = None

    def stats(self) -> dict:
        units = sum(len(tl.units) for tl in self.timelines.values())
        sizes = [c.size for c in self.clustering.clusters]
        return {
            "units": units,
            "units_per_book": {b: len(tl.units)
                               for b, tl in self.timelines.items()},
            "clusters": len(self.clustering.clusters),
            "contested_clusters": sum(1 for s in sizes if s > 1),
            "cluster_size_hist": {k: sizes.count(k)
                                  for k in sorted(set(sizes))},
            "anchors": len(self.scaffold.anchors),
            "anchors_positioned": sum(1 for a in self.scaffold.anchors
                                      if a.position is not None),
            "graph_nodes": self.graph.n_nodes,
            "graph_edges": self.graph.n_edges,
            "edge_types": self.graph.edge_counts(),
            "removed_arcs": len(self.induced.removed),
            "conflicts": len(self.induced.conflicts),
            **(self.projection or {}),
        }


def run(structs: Dict[str, AnnotationStructure], corpus: Corpus,
        pericopes: PericopeLayer, chains: Dict[str, EntityChains],
        cfg: TavernConfig,
        embeddings: Optional[Dict[str, list]] = None,
        write: bool = True) -> Stage3Result:
    timelines = segment_corpus(structs, corpus, pericopes, chains)
    sc = scaffold_mod.build(structs, timelines,
                            enabled=cfg.use_anchor_scaffold)
    units_flat = {u.unit_id: u for tl in timelines.values() for u in tl.units}
    if cfg.disable_projection:
        # Addendum 11, R1: isolate the projection itself from Addendum 9's
        # other two changes by never running it -- EventUnit.projected_days/
        # parts and Timex3.projected_day/part stay at their empty defaults,
        # so PredicateIDF's D:/P: indexing (R2) has nothing to index.
        projection = {"timex_day_concrete": 0, "timex_day_subspecified": 0,
                     "timex_part_concrete": 0, "timex_part_subspecified": 0}
    else:
        projection = scaffold_mod.project_timexes(structs, sc, units_flat)
    clustering = cluster_units(timelines, sc, embeddings)
    intra = [c for s in structs.values() for c in s.conflicts]
    crossings = detect_order_conflicts(timelines, sc, embeddings)
    induced = induce(timelines, clustering, sc, intra + crossings,
                     structs=structs)
    eg = graph_mod.build(structs, timelines, clustering, induced, sc,
                         embeddings)
    res = Stage3Result(timelines, sc, clustering, induced, eg, projection)

    if write:
        out = cfg.run_dir() / "stage3"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "timeline.json", "w", encoding="utf-8") as fh:
            json.dump({
                "order": induced.order,
                "clusters": [
                    {"id": c.cluster_id, "members": c.members,
                     "books": sorted(c.books), "position": c.position,
                     "interval": c.anchor_interval,
                     "refs": [eg.node_units[m].ref for m in c.members],
                     "verses": [[f"{b}:{ch}:{v}" for b, ch, v in
                                 eg.node_units[m].verse_keys]
                                for m in c.members]}
                    for c in clustering.clusters],
                "removed_arcs": induced.removed,
                "conflicts": induced.conflicts,
                "stats": res.stats(),
            }, fh, indent=1)
    return res
