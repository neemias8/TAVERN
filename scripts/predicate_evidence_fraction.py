#!/usr/bin/env python3
"""
Addendum 9/10 -- how much predicate evidence actually reaches the score.

Two wrong instruments were tried here before this one, and the reasoning
matters more than the numbers for anyone reopening this later.

WRONG #1 (Addendum 9's own 27%/43-159): intersected the predicate/TIMEX3
terms across ALL witnesses of a cluster at once. 96 of 159 multi-witness
clusters hold 3 or 4 Gospels; requiring one term common to every one of them
is far stricter than the decision the system actually makes, which is
PAIRWISE (predicate_similarity scores one pair of units at a time). Corrected
to pairwise -- does at least one cross-book PAIR share a term -- the
no-evidence count drops from 116/159 (73.0%) to 65/159 (40.9%).

WRONG #2 (this script's own first version): pairwise, but on raw set
INTERSECTION rather than the IDF-weighted cosine `predicate_similarity`
actually computes. Ordinary narrative verbs (SAY, GO, COME) are common
enough that almost any two witnesses share ONE, which makes raw intersection
an optimistic upper bound on evidence, not a measurement of it -- the IDF
weighting is precisely what suppresses a term like that towards zero. Kept
below as `share_any_term` for reference, labelled as what it is.

INSTRUMENT USED FOR THE THESIS: the pairwise IDF-weighted cosine itself
(`predicate_similarity`, `event_coref._cosine` over `PredicateIDF.vector()`),
pooled over every cross-book pair inside every multi-witness cluster, times
its WEIGHTS["predicate"]=0.40 contribution to score(). Reported as a
distribution (p10/p25/median/p75/p90) plus the fraction of pairs below
0.02/0.04/0.08 of contribution -- a single pair's cosine can be exactly 0.0,
so raw intersection and near-zero cosine coexist; the distribution is what
actually says how much of score()'s 0.34 threshold this term typically
supplies.

    python scripts/predicate_evidence_fraction.py outputs/canonical outputs/ancoragem
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline
from tavern.config import TavernConfig
from tavern.stage3_anchoring_alignment import event_coref as ec
from tavern.stage3_anchoring_alignment import scaffold as scaffold_mod
from tavern.stage3_anchoring_alignment.local_timeline import EventUnit, segment_corpus

PREDICATE_WEIGHT = ec.WEIGHTS["predicate"]


def build(cfg: TavernConfig):
    corpus, pericopes, segments, chains, structs, reports = pipeline.prepare(
        cfg, verify=True)
    timelines = segment_corpus(structs, corpus, pericopes, chains)
    units: Dict[str, EventUnit] = {}
    for tl in timelines.values():
        for u in tl.units:
            units[u.unit_id] = u
    scaffold = scaffold_mod.build(structs, timelines, enabled=True)
    scaffold_mod.project_timexes(structs, scaffold, units)
    idf = ec.PredicateIDF(list(units.values()))
    vectors = {uid: idf.vector(u) for uid, u in units.items()}
    terms = {uid: set(u.preds) | {f"T:{t}" for t in u.timex_preds}
            for uid, u in units.items()}
    book_of = {uid: u.book for uid, u in units.items()}
    return units, vectors, terms, book_of


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def analyse(path: Path, units, vectors, terms, book_of) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    cos_values: List[float] = []
    share_any_term_clusters = 0
    total = 0
    for cl in d["clusters"]:
        if len(set(cl["books"])) < 2:
            continue
        total += 1
        members = [m for m in cl["members"] if m in units]
        any_term = False
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ui, uj = members[i], members[j]
                if book_of[ui] == book_of[uj]:
                    continue
                cos_values.append(ec._cosine(vectors[ui], vectors[uj]))
                if terms[ui] & terms[uj]:
                    any_term = True
        if any_term:
            share_any_term_clusters += 1

    cos_values.sort()
    contrib = [PREDICATE_WEIGHT * c for c in cos_values]
    n = len(contrib)
    below = {t: (sum(1 for c in contrib if c < t) / n if n else 0.0)
            for t in (0.02, 0.04, 0.08)}
    return {
        "source": str(path),
        "multi_witness_clusters": total,
        "cross_book_pairs": n,
        "share_any_term_upper_bound":
            round(share_any_term_clusters / total, 4) if total else None,
        "share_any_term_upper_bound_note":
            "raw set intersection, pairwise -- an optimistic upper bound, "
            "not the score's own signal (see module docstring)",
        "predicate_contribution_0.40x_cosine": {
            "p10": round(percentile(contrib, 0.10), 4),
            "p25": round(percentile(contrib, 0.25), 4),
            "median": round(percentile(contrib, 0.50), 4),
            "p75": round(percentile(contrib, 0.75), 4),
            "p90": round(percentile(contrib, 0.90), 4),
        },
        "fraction_of_pairs_below_contribution": {
            str(t): round(v, 4) for t, v in below.items()
        },
    }


def main() -> int:
    tags = sys.argv[1:] or ["outputs/canonical", "outputs/ancoragem"]
    cfg = TavernConfig(tag="predicate_evidence_check", backbone="extractive")
    units, vectors, terms, book_of = build(cfg)
    results = []
    for tag in tags:
        p = Path(tag) / "stage3" / "timeline.json"
        if not p.exists():
            print(f"{tag}: no {p}, skipped")
            continue
        r = analyse(p, units, vectors, terms, book_of)
        results.append(r)
        print(f"=== {tag} ===")
        print(f"  multi-witness clusters: {r['multi_witness_clusters']}  "
             f"cross-book pairs: {r['cross_book_pairs']}")
        print(f"  share-any-term (upper bound, pairwise): "
             f"{r['share_any_term_upper_bound']:.4f}")
        c = r["predicate_contribution_0.40x_cosine"]
        print(f"  0.40*cosine contribution: p10={c['p10']} p25={c['p25']} "
             f"median={c['median']} p75={c['p75']} p90={c['p90']}")
        print(f"  fraction of PAIRS below contribution: "
             f"{r['fraction_of_pairs_below_contribution']}")
        print()

    out = Path("outputs/measurement_a/predicate_evidence_fraction.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
