#!/usr/bin/env python3
"""
Addendum 6, Measurement A -- recall@k of score() alone, the curated
segmentation given for free.

Isolates the ANNOTATION's discriminative signal (predicate, participants,
anchor, modal, class -- exactly score() in
tavern/stage3_anchoring_alignment/event_coref.py) from the ALIGNMENT
algorithm: no monotone profile, no MATCH_THRESHOLD, no ANCHOR_BAND, no
_merge_episodes. The question is narrower and more diagnostic than the
induced pipeline's own pureza: given the right segmentation, how well does
the scoring function alone pick the right cross-document match out of every
candidate?

For every curated event with >=2 Gospels and every ordered pair of books
(A, B) both reporting it: the query is A's real EventUnits for that event
(their verse-key union with the chronology's citation for A); the candidates
are EVERY one of B's own curated-event spans (all events B has a version
of, not just the ones A shares). Query-vs-candidate score is the mean of
score(a, b, scaffold, vectors, embeddings=None) over every (a, b) pair
between the two spans -- the same "mean of scores against column members"
rule _add_to_profile already uses for a single new unit against an existing
profile column, generalised symmetrically to span-vs-span since here both
sides can hold more than one unit. embeddings=None throughout, matching
pipeline.run's own call (SBERT is configured but never actually wired into
this scoring path).

    python scripts/measurement_a_recallk.py
    python scripts/measurement_a_recallk.py --ablate participants

This is Stage 6 diagnostic code -- it reads the chronology, so it must not be
importable from stages 1-5 (config.assert_no_chronology_import would catch
that). It is a standalone script for exactly that reason, same as
cluster_purity.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from collections import Counter

from tavern import pipeline
from tavern.config import OUTPUT_DIR, TavernConfig
from tavern.stage3_anchoring_alignment import scaffold as scaffold_mod
from tavern.stage3_anchoring_alignment.event_coref import (
    PredicateIDF, anchor_compatibility, class_agreement, modal_compatibility,
    participant_similarity, predicate_similarity)
from tavern.stage3_anchoring_alignment.local_timeline import segment_corpus
from tavern.stage6_evaluation import chronology as chrono_mod
from tavern.stage6_evaluation.redundancy import _STOP, _TOKEN, _stem

TERMS = ("predicate", "participants", "anchor", "modal", "class")
WEIGHTS = {"predicate": 0.40, "participants": 0.25, "anchor": 0.15,
          "modal": 0.10, "class": 0.10}


def raw_score(a, b, scaffold, vectors, zero_term=None):
    """score(), inlined so a single term's weight can be zeroed without
    touching event_coref.WEIGHTS (which the induced pipeline still reads)."""
    w = dict(WEIGHTS)
    if zero_term:
        w[zero_term] = 0.0
    return (w["predicate"] * predicate_similarity(a, b, vectors, None)
            + w["participants"] * participant_similarity(a, b)
            + w["anchor"] * anchor_compatibility(a, b, scaffold)
            + w["modal"] * modal_compatibility(a, b)
            + w["class"] * class_agreement(a, b))


# ---------------------------------------------------------------------------
# Addendum 7, Task 2: the lexical_baseline control. Zero annotation -- no
# predicate label, no entity chain, no anchor, no modal type, no class.
# Content-word term-frequency cosine over the raw verse text, using the same
# stopword list, tokenizer and suffix stemmer already validated in
# redundancy.py's content-coverage measurement, so this reuses one
# tokenisation, not a second ad hoc one.
_LEXICAL_CACHE: dict = {}


def _lexical_vector(unit_id, units):
    v = _LEXICAL_CACHE.get(unit_id)
    if v is not None:
        return v
    words = [_stem(m.group(0)) for m in _TOKEN.finditer(units[unit_id].text)
             if len(m.group(0)) > 2 and m.group(0).lower() not in _STOP]
    c = Counter(words)
    norm = math.sqrt(sum(x * x for x in c.values())) or 1.0
    v = {w: x / norm for w, x in c.items()}
    _LEXICAL_CACHE[unit_id] = v
    return v


def _cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(k, 0.0) for k, x in a.items())


def lexical_score(a, b, units):
    return _cosine(_lexical_vector(a, units), _lexical_vector(b, units))


def build_context(cfg: TavernConfig):
    corpus, pericopes, segments, chains, structs, reports = pipeline.prepare(
        cfg, verify=True)
    timelines = segment_corpus(structs, corpus, pericopes, chains)
    sc = scaffold_mod.build(structs, timelines, enabled=cfg.use_anchor_scaffold)
    units = {}
    for tl in timelines.values():
        for u in tl.units:
            units[u.unit_id] = u
    idf = PredicateIDF(list(units.values()))
    vectors = {uid: idf.vector(u) for uid, u in units.items()}
    ch = chrono_mod.load(corpus)
    return units, sc, vectors, ch


def event_spans(ch, units):
    """{(event_id, book): [unit_ids]} for every event x book the chronology
    cites, restricted to units this pipeline actually produced for those
    verses."""
    by_book = defaultdict(list)
    for u in units.values():
        by_book[u.book].append(u)
    spans = {}
    for e in ch.events:
        for b, keys in e.verse_keys.items():
            kset = set(keys)
            matched = [u.unit_id for u in by_book[b]
                      if set(u.verse_keys) & kset]
            if matched:
                spans[(e.event_id, b)] = matched
    return spans


def span_score(query_ids, cand_ids, units, scaffold, vectors, zero_term=None):
    vals = [raw_score(units[a], units[b], scaffold, vectors, zero_term)
           for a in query_ids for b in cand_ids]
    return sum(vals) / len(vals) if vals else 0.0


def lexical_span_score(query_ids, cand_ids, units):
    vals = [lexical_score(a, b, units) for a in query_ids for b in cand_ids]
    return sum(vals) / len(vals) if vals else 0.0


def run(cfg: TavernConfig, zero_term=None, lexical=False):
    units, sc, vectors, ch = build_context(cfg)
    spans = event_spans(ch, units)

    # candidates per book: every event that book has a (found) span for
    cands_by_book = defaultdict(list)
    for (eid, b), ids in spans.items():
        cands_by_book[b].append((eid, ids))

    ranks = []
    n_candidates = []
    skipped_no_query = skipped_no_true_candidate = 0
    for e in ch.events:
        if e.n_versions < 2:
            continue
        for a_book in e.books:
            query = spans.get((e.event_id, a_book))
            if not query:
                skipped_no_query += 1
                continue
            for b_book in e.books:
                if b_book == a_book:
                    continue
                cands = cands_by_book[b_book]
                if not any(eid == e.event_id for eid, _ in cands):
                    skipped_no_true_candidate += 1
                    continue
                if lexical:
                    scored = [(lexical_span_score(query, ids, units), eid)
                             for eid, ids in cands]
                else:
                    scored = [(span_score(query, ids, units, sc, vectors,
                                          zero_term), eid)
                             for eid, ids in cands]
                # stable, deterministic tie-break: score desc, event_id asc
                scored.sort(key=lambda t: (-t[0], t[1]))
                rank = next(i for i, (_, eid) in enumerate(scored, start=1)
                           if eid == e.event_id)
                ranks.append(rank)
                n_candidates.append(len(cands))

    n = len(ranks)
    r1 = sum(1 for r in ranks if r <= 1) / n if n else 0.0
    r5 = sum(1 for r in ranks if r <= 5) / n if n else 0.0
    r10 = sum(1 for r in ranks if r <= 10) / n if n else 0.0
    mrr = sum(1.0 / r for r in ranks) / n if n else 0.0
    avg_cand = sum(n_candidates) / n if n else 0.0
    return {
        "zero_term": "lexical_baseline" if lexical else zero_term, "queries": n,
        "skipped_no_query_span": skipped_no_query,
        "skipped_no_true_candidate": skipped_no_true_candidate,
        "avg_candidates": round(avg_cand, 2),
        "chance_at_1": round(1 / avg_cand, 4) if avg_cand else None,
        "recall@1": round(r1, 4), "recall@5": round(r5, 4),
        "recall@10": round(r10, 4), "mrr": round(mrr, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", choices=TERMS, default=None,
                    help="zero one term's weight and report recall@1 only "
                         "(the leave-one-out per-term contribution)")
    ap.add_argument("--all-terms", action="store_true",
                    help="run the full score plus all five single-term "
                         "ablations in one call")
    ap.add_argument("--lexical", action="store_true",
                    help="Addendum 7 Task 2: zero-annotation content-word "
                         "cosine baseline, same bench, same queries")
    args = ap.parse_args()

    cfg = TavernConfig(tag="measurement_a")
    out_dir = OUTPUT_DIR / "measurement_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.lexical:
        print("--- lexical_baseline (zero annotation) ---")
        row = run(cfg, lexical=True)
        print(row)
        rows.append({"config": "lexical_baseline", **row})
    elif args.all_terms:
        print("--- full score (all five terms) ---")
        full = run(cfg, zero_term=None)
        print(full)
        rows.append({"config": "full", **full})
        for t in TERMS:
            print(f"--- ablate: {t} ---")
            row = run(cfg, zero_term=t)
            print(row)
            rows.append({"config": f"-{t}", **row})
    else:
        row = run(cfg, zero_term=args.ablate)
        print(row)
        rows.append({"config": args.ablate or "full", **row})

    out_name = "recallk_lexical.json" if args.lexical else "recallk.json"
    (out_dir / out_name).write_text(json.dumps(rows, indent=1),
                                    encoding="utf-8")
    print(f"\nwrote {out_dir / out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
