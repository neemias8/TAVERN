"""
Stage 5 - temporally-guided abstractive generation (thesis Chapter 8).

Micro-abstractive fusion: for each event of the induced timeline in order, the
cluster of aligned source spans is retrieved, a generator fuses that cluster
into a single paragraph, and the paragraphs are concatenated.

Chronology is a property of the loop. The generator is never asked to decide
ordering and cannot violate it: under a curated timeline this made tau = 1.000
an architectural guarantee; under the induced timeline it makes ordering errors
attributable entirely to Stage 3, which is the correct place for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..config import TavernConfig
from ..stage3_anchoring_alignment.event_coref import Clustering
from ..stage3_anchoring_alignment.global_timeline import InducedTimeline
from ..stage3_anchoring_alignment.graph import EventGraph
from ..stage3_anchoring_alignment.local_timeline import EventUnit

__all__ = ["Consolidation", "consolidate", "ExtractiveFuser",
           "SelectionStrategy"]


@dataclass
class Consolidation:
    text: str
    paragraphs: List[str] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)
    selected: Dict[str, str] = field(default_factory=dict)   # cluster -> book
    conflicted: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.text)

    def write(self, path: Path, with_markers: bool = False) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if with_markers:
            body = "\n\n".join(f"[{m}] {p}" for m, p in
                               zip(self.markers, self.paragraphs))
        else:
            body = self.text
        Path(path).write_text(body, encoding="utf-8")
        return Path(path)


# ---------------------------------------------------------------------------
class SelectionStrategy:
    """Version selection within a cluster.

    `graph_score` is the configuration of the thesis: the highest-scoring node
    representation from Stage 4. The others are the published baselines,
    retained so that the ladder of Table tab:setup-baselines can be reproduced
    on the induced clustering as well as on the curated one.
    """

    def __init__(self, name: str = "graph_score",
                 scores: Optional[Dict[str, float]] = None,
                 priority: Sequence[str] = ("john", "luke", "matthew", "mark"),
                 seed: int = 13):
        self.name = name
        self.scores = scores or {}
        self.priority = list(priority)
        self.seed = seed

    def pick(self, spans: Dict[str, List[str]],
             units: Dict[str, EventUnit]) -> str:
        """Choose the document whose account of this event is emitted.

        Selection is over DOCUMENTS rather than over units, because a candidate
        canonical event holds a contiguous span per document and a consolidation
        that switched source mid-episode would fragment the account.
        """
        books = list(spans)
        if len(books) == 1:
            return books[0]

        def text_len(b: str) -> int:
            return sum(len(units[u].text) for u in spans[b])

        if self.name == "longest":
            return max(books, key=text_len)
        if self.name == "priority":
            return min(books, key=lambda b: self.priority.index(b)
                       if b in self.priority else 99)
        if self.name == "random":
            import random
            rng = random.Random(f"{self.seed}:{sorted(spans)[0]}:"
                               f"{spans[sorted(spans)[0]][0]}")
            return rng.choice(sorted(books))
        if self.name == "centroid":
            return _centroid_book(spans, units)
        if self.scores:
            return max(books, key=lambda b: sum(self.scores.get(u, 0.0)
                                                for u in spans[b])
                       / max(1, len(spans[b])))
        return max(books, key=text_len)


def _centroid_book(spans: Dict[str, List[str]],
                   units: Dict[str, EventUnit]) -> str:
    """Intra-cluster lexical centroid: the published Timeline+Centroid rule,
    applied over documents."""
    import math
    from collections import Counter
    vecs = {}
    for b, span in spans.items():
        c = Counter(w.lower() for u in span for w in units[u].text.split())
        n = math.sqrt(sum(v * v for v in c.values())) or 1.0
        vecs[b] = {k: v / n for k, v in c.items()}
    best, best_s = sorted(spans)[0], -1.0
    for b in spans:
        s = 0.0
        for o in spans:
            if o == b:
                continue
            x, y = vecs[b], vecs[o]
            if len(x) > len(y):
                x, y = y, x
            s += sum(v * y.get(k, 0.0) for k, v in x.items())
        if s > best_s:
            best, best_s = b, s
    return best


# ---------------------------------------------------------------------------
class ExtractiveFuser:
    """The extractive configuration: emit the selected version verbatim.

    This is the configuration whose numbers are comparable with the extractive
    rows of the published degradation curve (thesis Section 10.6), which is why
    the decisive comparison of the thesis does not depend on a generator.
    """

    name = "extractive"
    instructable = False

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None) -> str:
        return texts[0] if texts else ""


# ---------------------------------------------------------------------------
def consolidate(induced: InducedTimeline, clustering: Clustering,
                units: Dict[str, EventUnit], fuser=None,
                strategy: Optional[SelectionStrategy] = None,
                conflicts: Optional[Sequence[Tuple[str, str]]] = None,
                cluster_context: Optional[Dict[str, object]] = None
                ) -> Consolidation:
    """Algorithm alg:gen-fusion."""
    fuser = fuser or ExtractiveFuser()
    strategy = strategy or SelectionStrategy("graph_score")
    conflicted_clusters = set()
    for a, b in (conflicts or ()):
        conflicted_clusters.add(a)
        conflicted_clusters.add(b)

    out = Consolidation(text="")
    for t, cid in enumerate(induced.order, start=1):
        cl = clustering.by_id(cid)
        if cl is None or not cl.members:
            continue
        spans = cl.spans(units)
        chosen = strategy.pick(spans, units)
        out.selected[cid] = chosen
        # texts ordered so that the account the graph judges most
        # representative appears first (thesis Section 8.5)
        ordered = [chosen] + [b for b in spans if b != chosen]
        texts = [" ".join(units[u].text for u in spans[b]).strip()
                 for b in ordered]
        texts = [x for x in texts if x]
        if not texts:
            continue
        is_conflicted = cid in conflicted_clusters
        if is_conflicted:
            out.conflicted.append(cid)
        if is_conflicted and not getattr(fuser, "instructable", False):
            # a faithful single account is preferable to a fused paragraph that
            # silently adjudicates a disagreement the system detected
            para = texts[0]
        else:
            para = fuser.fuse(texts, conflicted=is_conflicted,
                              context=(cluster_context or {}).get(cid))
        if not para.strip():
            continue
        out.paragraphs.append(para.strip())
        out.markers.append(f"E{t:03d}:{cid}")
    out.text = " ".join(out.paragraphs)
    return out
