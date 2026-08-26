"""
The published baseline ladder (thesis Table tab:setup-baselines).

Every system here operates on the CURATED chronology and therefore belongs to
Stage 6, not to the pipeline: these are the reference points against which
TAVERN is measured, and reproducing them on the canonical corpus is what makes
the comparison sound rather than a citation of numbers computed on other data.

Included:
  LexRank (750 sentences)   timeline-agnostic extractive multi-document
  Timeline+Random           the performance floor available from structure alone
  Timeline+Priority         a fixed source-priority editorial policy
  Timeline+Centroid         purely local selection by intra-cluster similarity
  Timeline+Longest          the bar, and the benchmark's open challenge
  TAEG (Algorithm 1)        centrality over the similarity-weighted event graph

and two analyses that the thesis needs:

  degradation_curve         the published robustness analysis, reproduced
  timeline_substitution     the same loop with the curated ORDER replaced by an
                            induced one, holding the curated segmentation fixed

The second is the comparison that matches the degradation curve's design. The
curve degrades the TIMELINE and holds the segmentation of the narrative into
events fixed; a configuration that also induces the segmentation is measured
against a reference built from the curated segmentation and is penalised for
disagreeing with it, which is a different quantity. Both are reported.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..stage6_evaluation.chronology import CanonicalEvent, Chronology

BOOK_PRIORITY = ["john", "luke", "matthew", "mark"]


# ---------------------------------------------------------------------------
def _tfidf(texts: Sequence[str]) -> List[Dict[str, float]]:
    docs = [Counter(w.lower().strip(".,;:!?\"'()") for w in t.split())
            for t in texts]
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs) or 1
    out = []
    for d in docs:
        vec = {w: (1 + math.log(c)) * math.log((n + 1) / (1 + df[w]))
               for w, c in d.items() if w}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        out.append({w: v / norm for w, v in vec.items()})
    return out


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


# ---------------------------------------------------------------------------
def timeline_selection(chronology: Chronology, rule: str = "longest",
                       seed: int = 13,
                       order: Optional[Sequence[int]] = None) -> str:
    """A timeline-aware extractive system: one version per canonical event."""
    events = [e for e in chronology.events if e.texts]
    if order is not None:
        rank = {eid: i for i, eid in enumerate(order)}
        events.sort(key=lambda e: rank.get(e.event_id, 10 ** 6))
    rng = random.Random(seed)
    paras: List[str] = []
    for e in events:
        paras.append(_pick(e, rule, rng))
    return " ".join(p for p in paras if p)


def _pick(e: CanonicalEvent, rule: str, rng: random.Random) -> str:
    if rule == "longest":
        return max(e.texts.values(), key=len)
    if rule == "shortest":
        return min(e.texts.values(), key=len)
    if rule == "priority":
        for b in BOOK_PRIORITY:
            if b in e.texts:
                return e.texts[b]
        return ""
    if rule == "random":
        return e.texts[rng.choice(sorted(e.texts))]
    if rule == "centroid":
        books = sorted(e.texts)
        if len(books) == 1:
            return e.texts[books[0]]
        vecs = _tfidf([e.texts[b] for b in books])
        best, best_s = books[0], -1.0
        for i, b in enumerate(books):
            s = sum(_cos(vecs[i], vecs[j]) for j in range(len(books))
                    if j != i)
            if s > best_s:
                best, best_s = b, s
        return e.texts[best]
    if rule == "all":
        return " ".join(e.texts[b] for b in e.books)
    raise ValueError(rule)


# ---------------------------------------------------------------------------
def taeg_algorithm1(chronology: Chronology,
                    order: Optional[Sequence[int]] = None
                    ) -> Tuple[str, Dict[int, str]]:
    """TAEG Algorithm 1: centrality over the similarity-weighted event graph.

    The graph of the benchmark study carries directed BEFORE edges between
    sequential events within a document and undirected SAME_EVENT edges between
    all versions of one canonical event, weighted by TF-IDF cosine similarity.
    The version of highest eigenvector centrality within each cluster is
    selected. Reproduced here because it is the row the published degradation
    curve is computed on.
    """
    import networkx as nx

    events = [e for e in chronology.events if e.texts]
    if order is not None:
        rank = {eid: i for i, eid in enumerate(order)}
        events.sort(key=lambda e: rank.get(e.event_id, 10 ** 6))

    nodes: List[Tuple[int, str]] = []
    texts: List[str] = []
    for e in events:
        for b in e.books:
            nodes.append((e.event_id, b))
            texts.append(e.texts[b])
    vecs = _tfidf(texts)
    index = {n: i for i, n in enumerate(nodes)}

    g = nx.Graph()
    g.add_nodes_from(nodes)
    # SAME_EVENT edges
    for e in events:
        bs = e.books
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = (e.event_id, bs[i]), (e.event_id, bs[j])
                g.add_edge(a, b, weight=max(1e-6,
                                            _cos(vecs[index[a]], vecs[index[b]])))
    # BEFORE edges: sequential events within a document
    by_book: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for e in events:
        for b in e.books:
            by_book[b].append((e.event_id, b))
    for b, seq in by_book.items():
        for x, y in zip(seq, seq[1:]):
            g.add_edge(x, y, weight=max(1e-6, _cos(vecs[index[x]],
                                                   vecs[index[y]])))

    try:
        cent = nx.eigenvector_centrality_numpy(g, weight="weight")
    except Exception:
        cent = nx.degree_centrality(g)

    chosen: Dict[int, str] = {}
    paras: List[str] = []
    for e in events:
        best = max(e.books, key=lambda b: cent.get((e.event_id, b), 0.0))
        chosen[e.event_id] = best
        paras.append(e.texts[best])
    return " ".join(paras), chosen


# ---------------------------------------------------------------------------
def lexrank(chronology: Chronology, corpus, n_sentences: int = 750) -> str:
    """LexRank over the concatenated corpus: the timeline-agnostic baseline."""
    import re

    import networkx as nx

    sents: List[str] = []
    for book in corpus.books:
        for v in corpus[book].verses:
            for s in re.split(r"(?<=[.!?])\s+", v.text):
                s = s.strip()
                if len(s.split()) >= 4:
                    sents.append(s)
    vecs = _tfidf(sents)
    g = nx.Graph()
    g.add_nodes_from(range(len(sents)))
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            w = _cos(vecs[i], vecs[j])
            if w > 0.12:
                g.add_edge(i, j, weight=w)
    try:
        pr = nx.pagerank(g, weight="weight")
    except Exception:
        pr = {i: 1.0 for i in range(len(sents))}
    top = sorted(range(len(sents)), key=lambda i: -pr.get(i, 0.0))[:n_sentences]
    top.sort()
    return " ".join(sents[i] for i in top)


# ---------------------------------------------------------------------------
@dataclass
class DegradationPoint:
    label: str
    fraction: float
    runs: int
    scores: dict


def degradation_curve(chronology: Chronology, reference: str,
                      fractions: Sequence[float] = (0.0, 0.10, 0.25, 0.50),
                      runs: int = 10, rule: str = "taeg",
                      seed: int = 13) -> List[DegradationPoint]:
    """Reproduce the published robustness analysis.

    Each level removes the stated percentage of canonical events at random and
    re-runs the same consolidation loop; values are the mean over `runs`.
    """
    from ..stage6_evaluation.content_metrics import rouge

    out: List[DegradationPoint] = []
    events = [e for e in chronology.events if e.texts]
    for frac in fractions:
        acc = defaultdict(float)
        n = runs if frac > 0 else 1
        for r in range(n):
            rng = random.Random(seed * 1000 + r)
            keep = list(events)
            if frac > 0:
                k = int(round(frac * len(events)))
                drop = set(rng.sample(range(len(events)), k))
                keep = [e for i, e in enumerate(events) if i not in drop]
            sub = Chronology(events=keep)
            if rule == "taeg":
                text, _ = taeg_algorithm1(sub)
            else:
                text = timeline_selection(sub, rule)
            sc = rouge(text, reference)
            for k2, v in sc.items():
                acc[k2] += v / n
            acc["length"] += len(text) / n
        out.append(DegradationPoint(
            label=f"Curated, -{int(frac * 100)}%" if frac else "Curated, complete",
            fraction=frac, runs=n, scores=dict(acc)))
    return out


def timeline_substitution(chronology: Chronology, reference: str,
                          induced_order: Sequence[int],
                          rule: str = "taeg") -> dict:
    """The same loop with the curated ORDER replaced by an induced one.

    Segmentation, selection and content are held fixed; only the ordering
    changes. This is the configuration directly comparable with the degradation
    curve, whose levels also vary only the timeline.
    """
    from ..stage6_evaluation.content_metrics import rouge

    if rule == "taeg":
        text, _ = taeg_algorithm1(chronology, order=induced_order)
    else:
        text = timeline_selection(chronology, rule, order=induced_order)
    out = rouge(text, reference)
    out["length"] = len(text)
    return out
