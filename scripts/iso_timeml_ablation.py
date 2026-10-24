#!/usr/bin/env python3
"""
Full Stage 2 ISO-TimeML Ablation Study on the Unpatched TAVERN ('ancoragem') Pipeline.

    python scripts/iso_timeml_ablation.py
    python scripts/iso_timeml_ablation.py --json outputs/iso_timeml_ablation_report.json

Purpose
-------
Isolates the exact structural and downstream contribution of Stage 2 ISO 24617-1:2012
(ISO-TimeML) temporal and modal annotations within the primary TAVERN architecture
('ancoragem', WITHOUT the post-hoc Stage 3 refinement of Section 10.9).

Specifically, this script bridges the gap between the single-component ablations of
Table 10.2 (Without closure, Without veridicality, Without scaffold) and the
zero-annotation Positional Null Model (N1, tau = 0.8140) by evaluating what happens
when Stages 1 and 3 (pericope boundaries, entity coreference chains, progressive
monotone profile alignment, and tournament ordering) are kept intact while Stage 2
ISO-TimeML annotations are progressively or completely removed:

  1. Full TAVERN (Principal 'ancoragem'): All Stage 2 ISO-TimeML elements active
     (<EVENT>, <TIMEX3>, <SIGNAL>, <TLINK>, <SLINK>, <ALINK>, <MLINK>).
  2. Without Transitive Closure (use_closure=False): Only asserted <TLINK>s, no
     Allen-algebraic closure.
  3. Without Modal Veridicality (use_veridicality=False): No <SLINK> modal partition
     (prophecies, commands, and counterfactuals admitted as timeline-eligible).
  4. Without <TIMEX3> Scaffold & Projections (use_anchor_scaffold=False,
     disable_projection=True): Documents registered by relative position alone;
     no liturgical calendar pins or D:{day}/P:{part} terms.
  5. Without Temporal/Modal ISO-TimeML (Fixed 567 Units): Keeps the 567 unit
     boundaries fixed and retains bare <EVENT> verb/noun lemmas (@pred) + Stage 1
     entity chains, but strips all temporal/modal ISO-TimeML attributes (<TIMEX3>,
     <SIGNAL>, <TLINK>, <SLINK>, @tense, @aspect, @class).
  6. Without Temporal/Modal ISO-TimeML (Free Segmentation): Same as (5), but
     also removes <TIMEX3> and <SIGNAL> boundary triggers from Stage 3 unit
     segmentation, testing how much intra-pericope episode separation depends on
     temporal expressions and shift signals.
  7. Zero Stage 2 ISO-TimeML (Stage 1 + Stage 3 Only): Passes an empty
     AnnotationStructure to Stage 3 (zero <EVENT>, <TIMEX3>, <SIGNAL>, <TLINK>,
     <SLINK>), leaving only Stage 1 pericopes + entity chains and Stage 3
     progressive profile alignment over relative positions.
  8. Positional Null Model (N1): Zero annotation and zero alignment (uniform
     positional verse windows across the four Gospels).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_stage3_refinement as phr
import cluster_purity as cp
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR, TavernConfig
from tavern.stage1_preprocessing.coref import resolve_all
from tavern.stage1_preprocessing.corpus import Corpus
from tavern.stage1_preprocessing.pericopes import load_pericopes
from tavern.stage2_temporal_annotation import annotate_corpus
from tavern.stage2_temporal_annotation.model import AnnotationStructure
from tavern.stage3_anchoring_alignment import (
    event_coref,
    global_timeline,
    local_timeline,
    scaffold as scaffold_mod,
)
from tavern.stage5_generation import ExtractiveFuser, SelectionStrategy, consolidate
from tavern.stage6_evaluation import (
    chronology as chrono_mod,
    error_analysis,
    timeline_eval,
)


def get_tokens(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def rouge_n(hyp_toks: list[str], ref_toks: list[str], n: int = 1) -> float:
    h_ng = Counter(tuple(hyp_toks[i : i + n]) for i in range(len(hyp_toks) - n + 1))
    r_ng = Counter(tuple(ref_toks[i : i + n]) for i in range(len(ref_toks) - n + 1))
    overlap = sum((h_ng & r_ng).values())
    if overlap == 0:
        return 0.0
    p = overlap / max(1, sum(h_ng.values()))
    r = overlap / max(1, sum(r_ng.values()))
    return 2.0 * p * r / (p + r)


def ensure_unpatched_stage3() -> None:
    """Guarantee that Stage 3 runs in the exact unpatched 'ancoragem' mode."""
    event_coref.REFINE_GROUPING = False
    event_coref.MATCH_THRESHOLD = 0.34
    event_coref.ANCHOR_BAND = 4.0
    event_coref.GAP_COST = 0.06
    event_coref.MAX_EPISODE_VERSES = 2
    event_coref.GATED_SCORE = False
    event_coref.NO_ANCHOR_CREDIT = False
    event_coref.DISABLE_PROJECTION_INDEXING = False

    def unpatched_mergeable(cur, nxt, units, succ):
        last_of = {}
        verses_of = {}
        for uid in cur:
            b = units[uid].book
            last_of[b] = uid
            verses_of[b] = verses_of.get(b, 0) + len(units[uid].verse_keys)
        for uid in nxt:
            u = units[uid]
            if u.anchorable_timex:
                return False
            prev = last_of.get(u.book)
            if prev is not None:
                if succ.get(prev) != uid:
                    return False
                if units[prev].pericope_id != u.pericope_id:
                    return False
                if (
                    event_coref.MAX_EPISODE_VERSES is not None
                    and verses_of.get(u.book, 0) + len(u.verse_keys)
                    > event_coref.MAX_EPISODE_VERSES
                ):
                    return False
        return True

    event_coref._mergeable = unpatched_mergeable


def evaluate_condition(
    label: str,
    cur_structs: dict,
    timelines: dict,
    sc: scaffold_mod.Scaffold,
    cl: event_coref.Clustering,
    ind: global_timeline.InducedTimeline,
    ch,
    segments: dict,
    v2e: dict,
    e2day: dict,
    ref_toks: list[str],
) -> dict:
    units = {u.unit_id: u for tl in timelines.values() for u in tl.units}
    ev = timeline_eval.evaluate(ch, cl, ind, units)
    matching, _ = timeline_eval.match_clusters_to_events(ch, cl, units)
    errs = {
        c.name: c.count
        for c in error_analysis.analyse(
            ch, cl, units, ind, matching, cur_structs, segments
        )
    }
    cons = consolidate(
        ind,
        cl,
        units,
        fuser=ExtractiveFuser(),
        strategy=SelectionStrategy("longest"),
        conflicts=[],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"events": cons.records}, tf)
        tpath = tf.name
    pur = cp.analyse(Path(tpath), v2e, e2day)
    b3 = pur["bcubed"]
    hyp_toks = get_tokens(cons.text)
    r1 = rouge_n(hyp_toks, ref_toks, 1)
    r2 = rouge_n(hyp_toks, ref_toks, 2)
    over_m = errs.get("Over-merged cluster (two events joined)", 0)
    under_m = errs.get("Under-merged cluster (one event split)", 0)
    day_disp = errs.get("Misordered: displaced across a day boundary", 0)

    row = {
        "config": label,
        "units": len(units),
        "clusters": len(cl.clusters),
        "multi_witness_clusters": pur["multi_witness_clusters"],
        "purity": round(pur["purity"], 4),
        "day_mixing": round(pur["day_mixing"], 4),
        "bcubed": {
            "precision": round(b3["precision"], 4),
            "recall": round(b3["recall"], 4),
            "f1": round(b3["f1"], 4),
        },
        "tau_plan": round(ev.tau, 4) if ev.tau is not None else None,
        "pairwise_accuracy": round(ev.pairwise_accuracy, 4)
        if ev.pairwise_accuracy is not None
        else None,
        "matched_events": ev.matched_events,
        "total_events": ev.total_events,
        "coverage": round(ev.coverage, 4),
        "over_merged": over_m,
        "under_merged": under_m,
        "day_displaced": day_disp,
        "extractive_rouge1": round(r1, 4),
        "extractive_rouge2": round(r2, 4),
    }
    print(
        f"  {label:46s} | U={row['units']:3d} Cl={row['clusters']:3d} (MW={row['multi_witness_clusters']:3d}) | "
        f"Pur={row['purity']*100:5.2f}% DayMix={row['day_mixing']*100:5.2f}% | "
        f"B3={b3['precision']:.3f}/{b3['recall']:.3f}/{b3['f1']:.3f} | "
        f"tau={row['tau_plan']:.4f} cov={row['matched_events']}/{row['total_events']} ({row['coverage']*100:5.2f}%) | "
        f"Over={over_m:2d} Under={under_m:2d} | R1={r1:.4f} R2={r2:.4f}"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        default=str(OUTPUT_DIR / "iso_timeml_ablation_report.json"),
        help="Path to write the JSON ablation report.",
    )
    args = ap.parse_args()

    ensure_unpatched_stage3()
    corpus = Corpus()
    pericopes = load_pericopes(corpus)
    segments = phr.segment_corpus_pure(corpus)
    chains = resolve_all(segments)

    cfg_full = TavernConfig(tag="abl_full", backbone="extractive")
    structs_full, _ = annotate_corpus(segments, pericopes, cfg_full, write=False)

    ch = chrono_mod.load(corpus)
    v2e, e2day = cp.verse_to_event(DATA_DIR)
    ref_text = (DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace"
    )
    ref_toks = get_tokens(ref_text)

    print("=" * 135)
    print(
        "STAGE 2 ISO-TIMEML ABLATION STUDY ON UNPATCHED TAVERN ('ancoragem', NO POST-HOC REFINEMENT)"
    )
    print("=" * 135)

    rows = []

    # 1. Full TAVERN (unpatched, free segmentation)
    tl_1 = local_timeline.segment_corpus(structs_full, corpus, pericopes, chains)
    u_1 = {u.unit_id: u for tl in tl_1.values() for u in tl.units}
    sc_1 = scaffold_mod.build(structs_full, tl_1, enabled=True)
    scaffold_mod.project_timexes(structs_full, sc_1, u_1)
    cl_1 = event_coref.cluster_units(tl_1, sc_1, embeddings=None)
    ind_1 = global_timeline.induce(tl_1, cl_1, sc_1, structs=structs_full)
    rows.append(
        evaluate_condition(
            "1_full_tavern_unpatched",
            structs_full,
            tl_1,
            sc_1,
            cl_1,
            ind_1,
            ch,
            segments,
            v2e,
            e2day,
            ref_toks,
        )
    )

    # 2. Without <TIMEX3> Scaffold & Projections
    tl_2 = local_timeline.segment_corpus(structs_full, corpus, pericopes, chains)
    sc_2 = scaffold_mod.build(structs_full, tl_2, enabled=False)
    cl_2 = event_coref.cluster_units(tl_2, sc_2, embeddings=None)
    ind_2 = global_timeline.induce(tl_2, cl_2, sc_2, structs=structs_full)
    rows.append(
        evaluate_condition(
            "2_without_timex3_scaffold_and_projections",
            structs_full,
            tl_2,
            sc_2,
            cl_2,
            ind_2,
            ch,
            segments,
            v2e,
            e2day,
            ref_toks,
        )
    )

    # Build bare-predicate structures (only <EVENT> @pred lemma kept; no TIMEX3, SIGNAL, TLINK, SLINK, tense, aspect, class)
    bare_structs = {}
    for b, s in structs_full.items():
        bs = AnnotationStructure(book=b)
        for eid, ev in s.events.items():
            ev_copy = copy.copy(ev)
            ev_copy.tense = "NONE"
            ev_copy.aspect = "NONE"
            ev_copy.event_class = "NONE"
            bs.events[eid] = ev_copy
            bs.eligible.add(eid)
        bare_structs[b] = bs

    # 3. Without Temporal/Modal ISO-TimeML (Fixed segmentation, bare @pred + Entities)
    tl_3 = local_timeline.segment_corpus(structs_full, corpus, pericopes, chains)
    for tl in tl_3.values():
        tl.order.clear()
        tl.order_confidence.clear()
        tl.order_level.clear()
        tl.nrt_chain.clear()
        local_timeline._induce_order(tl, bare_structs[tl.book])
        for u in tl.units:
            u.timex_ids.clear()
            u.timex_preds.clear()
            u.anchorable_timex = False
            u.projected_days.clear()
            u.projected_parts.clear()
            u.signal_preds.clear()
            u.classes.clear()
            u.tenses.clear()
            u.modal_types.clear()
            u.eligible_fraction = 1.0
    sc_3 = scaffold_mod.build(bare_structs, tl_3, enabled=False)
    cl_3 = event_coref.cluster_units(tl_3, sc_3, embeddings=None)
    ind_3 = global_timeline.induce(tl_3, cl_3, sc_3, structs=bare_structs)
    rows.append(
        evaluate_condition(
            "3_without_temporal_modal_fixed_segmentation",
            bare_structs,
            tl_3,
            sc_3,
            cl_3,
            ind_3,
            ch,
            segments,
            v2e,
            e2day,
            ref_toks,
        )
    )

    # 4. Without Temporal/Modal ISO-TimeML (Free segmentation, bare @pred + Entities)
    tl_4 = local_timeline.segment_corpus(bare_structs, corpus, pericopes, chains)
    for tl in tl_4.values():
        for u in tl.units:
            u.classes.clear()
            u.tenses.clear()
            u.modal_types.clear()
    sc_4 = scaffold_mod.build(bare_structs, tl_4, enabled=False)
    cl_4 = event_coref.cluster_units(tl_4, sc_4, embeddings=None)
    ind_4 = global_timeline.induce(tl_4, cl_4, sc_4, structs=bare_structs)
    rows.append(
        evaluate_condition(
            "4_without_temporal_modal_free_segmentation",
            bare_structs,
            tl_4,
            sc_4,
            cl_4,
            ind_4,
            ch,
            segments,
            v2e,
            e2day,
            ref_toks,
        )
    )

    # 5. Zero Stage 2 ISO-TimeML (Empty AnnotationStructure: Stage 1 Pericopes+Entities + Stage 3 Profile only)
    empty_structs = {b: AnnotationStructure(book=b) for b in structs_full}
    tl_5 = local_timeline.segment_corpus(empty_structs, corpus, pericopes, chains)
    sc_5 = scaffold_mod.build(empty_structs, tl_5, enabled=False)
    cl_5 = event_coref.cluster_units(tl_5, sc_5, embeddings=None)
    ind_5 = global_timeline.induce(tl_5, cl_5, sc_5, structs=empty_structs)
    rows.append(
        evaluate_condition(
            "5_zero_stage2_iso_timeml_strict",
            empty_structs,
            tl_5,
            sc_5,
            cl_5,
            ind_5,
            ch,
            segments,
            v2e,
            e2day,
            ref_toks,
        )
    )

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved full JSON report to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())