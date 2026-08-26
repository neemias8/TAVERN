"""
Stage 6 - annotation statistics (thesis Tables tab:res-annstats,
tab:res-cascade, tab:ann-density).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import BOOK_ORDER, DISCOURSE_BLOCKS
from ..stage2_temporal_annotation.enums import POS, CascadeLevel
from ..stage2_temporal_annotation.model import AnnotationStructure

ROWS = ["event_verbal", "event_nominal", "event_state", "timex", "timex_empty",
        "signal", "tlink_asserted", "tlink_derived", "slink", "alink", "mlink",
        "eligible", "subordinated"]


def element_counts(structs: Dict[str, AnnotationStructure]) -> dict:
    per_book = {b: structs[b].counts() for b in BOOK_ORDER if b in structs}
    total = {k: sum(v[k] for v in per_book.values()) for k in ROWS}
    return {"per_book": per_book, "total": total}


def cascade_distribution(structs: Dict[str, AnnotationStructure],
                         nrt_chain_count: int = 0) -> dict:
    """Distribution of asserted temporal relations over the cascade levels.

    The narrative-reference-time chain of Section 6.2.5 is counted separately:
    it orders PERICOPES through the anchor chain of Layer A and is not one of
    the five event-level evidence types of Table tab:ann-cascade.
    """
    per_level: Counter = Counter()
    for s in structs.values():
        for l in s.asserted_tlinks():
            per_level[l.level] += 1
    total = sum(per_level.values()) or 1
    rows = []
    labels = {1: "Explicit signal", 2: "Temporal expression",
              3: "Aspectual predicate", 4: "Narrative progression"}
    for lv in (1, 2, 3, 4):
        rows.append({"level": lv, "evidence": labels[lv],
                     "relations": per_level.get(lv, 0),
                     "share": per_level.get(lv, 0) / total})
    anchoring = (per_level.get(1, 0) + per_level.get(2, 0)) / total
    return {"rows": rows, "total": total, "anchoring_coverage": anchoring,
            "derived": sum(len(s.derived_tlinks()) for s in structs.values()),
            "nrt_chain": nrt_chain_count}


def anchoring_coverage(structs: Dict[str, AnnotationStructure]) -> float:
    """Share of timeline-eligible events carrying a level-1 or level-2
    relation (internal consistency check 3)."""
    supported = 0
    total = 0
    for s in structs.values():
        strong = set()
        for l in s.asserted_tlinks():
            if l.level in (1, 2):
                for x in (l.source, l.target_id):
                    if x:
                        strong.add(x)
        for eid in s.eligible:
            total += 1
            if eid in strong:
                supported += 1
    return supported / total if total else 0.0


def normalisation_coverage(structs: Dict[str, AnnotationStructure]) -> float:
    """Share of <TIMEX3> elements with a non-null @value (check 4)."""
    tot = val = 0
    for s in structs.values():
        for t in s.timexes.values():
            if t.is_empty:
                continue
            tot += 1
            if t.value:
                val += 1
    return val / tot if tot else 0.0


# ---------------------------------------------------------------------------
_CUE_RE = re.compile(
    r"\b(before|after|during|while|when|until|till|since|then|now|immediately|"
    r"meanwhile|afterwards?|later|earlier|already|soon|shortly|finally|next|"
    r"first|previously|beforehand|day|days|night|nights|hour|hours|morning|"
    r"evening|noon|midnight|dawn|daybreak|week|month|year|years|sabbath|"
    r"passover|feast|preparation|time|times|season|watch|cockcrow|rooster|"
    r"today|tomorrow|yesterday|daily|forty|third|sixth|ninth)\b", re.I)


def density(corpus, structs: Dict[str, AnnotationStructure]) -> dict:
    """Table tab:ann-density: verses with any temporal cue, and verses carrying
    an anchorable expression.

    "Any cue" is a lexical count over the corpus, independent of the tagger, so
    that the figure describes the resource rather than the implementation.
    "Anchorable" is the annotation's own judgement.
    """
    rows = []
    tot_v = tot_c = tot_a = 0
    for book in BOOK_ORDER:
        if book not in structs:
            continue
        verses = corpus[book].verses
        cues = sum(1 for v in verses if _CUE_RE.search(v.text))
        anchor_keys = {t.verse_key for t in structs[book].timexes.values()
                       if t.anchorable and t.verse_key}
        rows.append({"book": book, "verses": len(verses), "cues": cues,
                     "anchorable": len(anchor_keys)})
        tot_v += len(verses)
        tot_c += cues
        tot_a += len(anchor_keys)
    return {"rows": rows, "verses": tot_v, "cues": tot_c,
            "anchorable": tot_a,
            "cue_share": tot_c / tot_v if tot_v else 0.0,
            "anchorable_share": tot_a / tot_v if tot_v else 0.0}


def discourse_verse_count(corpus) -> dict:
    """Table tab:ann-discourse recomputed from the corpus."""
    rows = []
    total = 0
    for book, c0, v0, c1, v1, label in DISCOURSE_BLOCKS:
        n = sum(1 for v in corpus[book].verses
                if (c0, v0) <= (v.chapter, v.number) <= (c1, v1))
        rows.append({"book": book, "block": label, "verses": n,
                     "span": f"{c0}:{v0}-{c1}:{v1}"})
        total += n
    return {"rows": rows, "total": total,
            "share": total / corpus.total_verses()}
