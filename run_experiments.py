#!/usr/bin/env python3
"""
Reproduce every measured table of the thesis.

    python run_experiments.py --all
    python run_experiments.py --annotation --timeline
    python run_experiments.py --ablations

Results are written to outputs/<tag>/results.json and printed as LaTeX-ready
rows. Nothing here is read by stages 1 to 5.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from tavern import baselines, pipeline
from tavern.config import (DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR,
                           TavernConfig, verify_corpus)
from tavern.stage3_anchoring_alignment.global_timeline import InducedTimeline
from tavern.stage5_generation import (ExtractiveFuser, SelectionStrategy,
                                      consolidate)
from tavern.stage6_evaluation import (annotation_stats, chronology as chrono_mod,
                                      conflicts as conflicts_mod, consistency,
                                      content_metrics, error_analysis,
                                      selection_eval, text_quality,
                                      timeline_eval)

RESULTS: Dict[str, object] = {}


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
def corpus_tables(res, ch) -> None:
    banner("Corpus and resource (Tables tab:task-corpus, tab:task-md5, "
           "tab:ann-discourse, tab:ann-density)")
    c = res.corpus
    print(f"verses per book       {c.verse_count()}  total {c.total_verses()}")
    print(f"words                 {c.word_count()}")
    print(f"pericopes             {len(res.pericopes)}")
    for r in res.pericopes.repairs:
        print(f"  pericope repair: {r}")
    print(f"canonical events      {len(ch)}")
    print(f"versions per book     {ch.versions_per_book()}  "
          f"total {sum(ch.versions_per_book().values())}")
    print(f"version distribution  {dict(sorted(ch.version_distribution().items()))}")
    print(f"day distribution      {ch.day_distribution()}")
    floor, n_contested = selection_eval.analytical_floor(ch)
    print(f"analytical floor      {floor:.4f} over {n_contested} contested events")

    disc = annotation_stats.discourse_verse_count(c)
    print(f"discourse blocks      {disc['total']} verses "
          f"({disc['share']*100:.1f}% of corpus)")
    dens = annotation_stats.density(c, res.structs)
    print(f"temporal density      any cue {dens['cues']} "
          f"({dens['cue_share']*100:.1f}%), anchorable {dens['anchorable']} "
          f"({dens['anchorable_share']*100:.1f}%)")
    for row in dens["rows"]:
        print(f"    {row['book']:8s} verses {row['verses']:4d} "
              f"cues {row['cues']:4d} anchorable {row['anchorable']:3d}")
    RESULTS["corpus"] = {
        "verses": c.verse_count(), "total_verses": c.total_verses(),
        "words": c.word_count(), "pericopes": len(res.pericopes),
        "pericope_repairs": res.pericopes.repairs,
        "events": len(ch), "versions": ch.versions_per_book(),
        "version_distribution": ch.version_distribution(),
        "day_distribution": ch.day_distribution(),
        "analytical_floor": floor, "contested": n_contested,
        "discourse": disc, "density": dens,
        "md5": {k: v[0] for k, v in verify_corpus(strict=False).items()},
    }


def annotation_tables(res) -> None:
    banner("Annotation statistics (Tables tab:res-annstats, tab:res-cascade)")
    counts = annotation_stats.element_counts(res.structs)
    order = ["matthew", "mark", "luke", "john"]
    labels = [
        ("event_verbal", "<EVENT>, verbal"), ("event_nominal", "<EVENT>, nominal"),
        ("event_state", "<EVENT>, states"), ("timex", "<TIMEX3>, realised"),
        ("timex_empty", "<TIMEX3>, anchoring"), ("signal", "<SIGNAL>"),
        ("tlink_asserted", "<TLINK>, asserted"),
        ("tlink_derived", "<TLINK>, derived"), ("slink", "<SLINK>"),
        ("alink", "<ALINK>"), ("mlink", "<MLINK>"),
        ("eligible", "Timeline-eligible events"),
        ("subordinated", "Subordinated events"),
    ]
    print(f"{'Element':28s}" + "".join(f"{b.capitalize():>10s}" for b in order)
          + f"{'Total':>10s}")
    for key, label in labels:
        row = "".join(f"{counts['per_book'][b][key]:>10d}" for b in order)
        print(f"{label:28s}{row}{counts['total'][key]:>10d}")

    casc = annotation_stats.cascade_distribution(res.structs,
                                                res.nrt_chain_count())
    print()
    for r in casc["rows"]:
        print(f"  level {r['level']}  {r['evidence']:24s} "
              f"{r['relations']:5d}  {r['share']*100:5.1f}%")
    print(f"  anchoring coverage (levels 1-2)      {casc['anchoring_coverage']*100:5.1f}%")
    print(f"  closure-derived relations            {casc['derived']:5d}")
    print(f"  narrative reference chain (pericope) {casc['nrt_chain']:5d}")
    RESULTS["annotation"] = {"counts": counts, "cascade": casc}


def consistency_table(res, ch) -> None:
    banner("Internal consistency (Table tab:res-checks)")
    rep = consistency.run(res.structs, res.reports, res.corpus,
                          res.stage3.induced.conflicts,
                          res.stage3.clustering, res.units, res.segments)
    for row in rep.as_rows():
        mark = "pass" if row["passed"] else ("--" if row["passed"] is None
                                             else "FAIL")
        print(f"  {row['check']:26s} {row['result']:>14s}  [{mark}]"
              + (f"  {row['detail']}" if row["detail"] else ""))
    RESULTS["consistency"] = rep.as_rows()


def timeline_tables(res, ch) -> None:
    banner("Induced timeline (Tables tab:res-tau, tab:res-conflicts)")
    ev = timeline_eval.evaluate(ch, res.stage3.clustering, res.stage3.induced,
                               res.units)
    print(f"  TAVERN, induced timeline      tau={ev.tau:.4f}  "
          f"pairwise={ev.pairwise_accuracy:.4f}  coverage={ev.coverage:.4f}  "
          f"({ev.matched_events}/{ev.total_events} events)")
    st = res.stage3.stats()
    print(f"  event units {st['units']}   clusters {st['clusters']}   "
          f"contested {st['contested_clusters']}   "
          f"size histogram {st['cluster_size_hist']}")
    print(f"  graph {st['graph_nodes']} nodes, {st['graph_edges']} edges")
    for k, v in sorted(st["edge_types"].items()):
        print(f"      {k:22s} {v}")

    rep = conflicts_mod.report(res.structs, res.stage3.induced,
                              res.stage3.clustering, res.units)
    print()
    for row in rep.as_rows():
        print(f"  {row['class']:44s} {row['count']:4d}  {row['documented']}")
    print(f"  documented cases recovered: {rep.detail}")
    RESULTS["timeline"] = {"induced": ev.as_row(), "stage3": st,
                           "conflicts": rep.as_rows(),
                           "conflict_examples": rep.examples,
                           "documented": rep.documented}


def induced_event_order(res, ch) -> List[int]:
    """Order the canonical events by the induced timeline alone.

    Every canonical event is placed at the mean induced rank of the clusters
    whose verses overlap it, weighted by the size of the overlap. Unlike the
    one-to-one matching used for Kendall's tau, this places EVERY event,
    including those whose best-matching cluster was claimed by a neighbour, and
    it consults nothing but the induced timeline and the verse addresses. It is
    what makes the substitution experiment a measurement of the ordering rather
    than of the matching.
    """
    rank = res.stage3.induced.rank
    units = res.units
    cluster_keys = {}
    for cl in res.stage3.clustering.clusters:
        ks = set()
        for m in cl.members:
            ks |= set(units[m].verse_keys)
        cluster_keys[cl.cluster_id] = ks

    placed = []
    for e in ch.events:
        ek = set(e.all_keys)
        if not ek:
            continue
        num = den = 0.0
        for cid, ks in cluster_keys.items():
            w = len(ek & ks)
            if w and cid in rank:
                num += w * rank[cid]
                den += w
        placed.append(((num / den) if den else len(rank), e.event_id))
    placed.sort()
    return [eid for _p, eid in placed]


def degradation_table(res, ch) -> None:
    banner("Degradation curve and timeline substitution "
           "(Table tab:res-degradation)")
    ref = reference()
    curve = baselines.degradation_curve(ch, ref, rule="taeg", runs=10)
    for p in curve:
        s = p.scores
        print(f"  {p.label:22s} R-L={s['rougeL']:.4f}  R-1={s['rouge1']:.4f}  "
              f"len={int(s['length'])}   (mean of {p.runs})")

    seq = induced_event_order(res, ch)

    for rule in ("taeg", "longest"):
        sub = baselines.timeline_substitution(ch, ref, seq, rule=rule)
        print(f"  Induced order, curated segmentation, {rule:8s} "
              f"R-L={sub['rougeL']:.4f}  R-1={sub['rouge1']:.4f}  "
              f"len={sub['length']}")
        RESULTS.setdefault("substitution", {})[rule] = sub

    if res.consolidation is not None:
        sc = content_metrics.evaluate(res.consolidation.text, ref,
                                      with_meteor=True, with_bertscore=False)
        bk = res.consolidation.backbone
        print(f"  Induced end to end, fusion '{bk}'".ljust(48)
              + f"R-L={sc.rougeL:.4f}  R-1={sc.rouge1:.4f}  "
                f"len={sc.length}")
        RESULTS["induced_end_to_end"] = {"backbone": bk, **sc.as_row()}
    RESULTS["degradation"] = [
        {"label": p.label, "fraction": p.fraction, "runs": p.runs,
         "scores": p.scores} for p in curve]
    RESULTS["induced_event_order"] = seq


def downstream_table(res, ch) -> None:
    banner("Downstream consolidation (Tables tab:res-downstream, "
           "tab:res-selection)")
    ref = reference()
    units = res.units
    scores = res.gnn.node_scores if res.gnn else {}
    conf = res.ordering_conflicts

    rows = {}

    # the configuration the framework is for: per-event fusion, whichever
    # backbone was requested. Reported first, and separately from the
    # extractive rows, because the two are not the same kind of system.
    if res.consolidation is not None:
        bk = res.consolidation.backbone
        sc = content_metrics.evaluate(res.consolidation.text, ref,
                                      with_meteor=True, with_bertscore=False)
        label = f"TAVERN, induced, fusion '{bk}'"
        rows[label] = sc.as_row()
        print(f"  {label:32s} R-1={sc.rouge1:.4f} R-2={sc.rouge2:.4f} "
              f"R-L={sc.rougeL:.4f} "
              f"MET={'n/a' if sc.meteor is None else round(sc.meteor, 4)} "
              f"len={sc.length}")
        if bk not in ("extractive", "union"):
            gw = text_quality.scan(res.consolidation.paragraphs)
            n_events = len(res.consolidation.paragraphs)
            chars_per_event = sc.length / n_events if n_events else 0.0
            print(f"    glued-word events: {gw.corrupted}/{gw.total} "
                  f"({gw.fraction:.1%})   chars/event: {chars_per_event:.0f}")
            RESULTS["text_quality"] = {**gw.as_row(),
                                       "chars_per_event": chars_per_event}
        print()
    for name, strat in (("TAVERN, induced (graph score)", "graph_score"),
                        ("  - graph propagation", "graph_score_flat"),
                        ("  Timeline+Longest rule", "longest"),
                        ("  Timeline+Centroid rule", "centroid"),
                        ("  Timeline+Priority rule", "priority"),
                        ("  Timeline+Random rule", "random")):
        use = scores
        if strat == "graph_score_flat":
            from tavern.stage4_gnn import aggregate_without_propagation
            use = aggregate_without_propagation(res.stage3.graph).node_scores
            strat = "graph_score"
        cons = consolidate(res.stage3.induced, res.stage3.clustering, units,
                           ExtractiveFuser(),
                           SelectionStrategy(strat, scores=use),
                           conflicts=conf)
        sc = content_metrics.evaluate(cons.text, ref, with_meteor=True,
                                      with_bertscore=False)
        rows[name] = sc.as_row()
        print(f"  {name:32s} R-1={sc.rouge1:.4f} R-2={sc.rouge2:.4f} "
              f"R-L={sc.rougeL:.4f} MET={sc.meteor if sc.meteor is None else round(sc.meteor,4)} "
              f"len={sc.length}")

    # oracle timeline configuration: curated ordering and segmentation
    print()
    for rule in ("longest", "centroid", "priority", "random"):
        text = baselines.timeline_selection(ch, rule)
        sc = content_metrics.evaluate(text, ref, with_meteor=True,
                                      with_bertscore=False)
        rows[f"Timeline+{rule.capitalize()} (curated)"] = sc.as_row()
        print(f"  Timeline+{rule.capitalize():9s} (curated)      "
              f"R-1={sc.rouge1:.4f} R-2={sc.rouge2:.4f} R-L={sc.rougeL:.4f} "
              f"len={sc.length}")
    text, chosen = baselines.taeg_algorithm1(ch)
    sc = content_metrics.evaluate(text, ref, with_meteor=True,
                                  with_bertscore=False)
    rows["TAEG (Algorithm 1, curated)"] = sc.as_row()
    print(f"  TAEG (Algorithm 1, curated)      R-1={sc.rouge1:.4f} "
          f"R-2={sc.rouge2:.4f} R-L={sc.rougeL:.4f} len={sc.length}")

    # selection-level accuracy
    print()
    matching, _q = timeline_eval.match_clusters_to_events(
        ch, res.stage3.clustering, res.units)
    subset = selection_eval.contested_intersection(ch, res.stage3.clustering,
                                                   matching)
    sel_rows = {}
    for name, strat in (("TAVERN, induced (graph score)", "graph_score"),
                        ("  - graph propagation", "flat"),
                        ("  longest rule", "longest"),
                        ("  centroid rule", "centroid")):
        use = scores
        s = strat
        if strat == "flat":
            from tavern.stage4_gnn import aggregate_without_propagation
            use = aggregate_without_propagation(res.stage3.graph).node_scores
            s = "graph_score"
        cons = consolidate(res.stage3.induced, res.stage3.clustering, units,
                           ExtractiveFuser(), SelectionStrategy(s, scores=use),
                           conflicts=conf)
        picked = selection_eval.induced_selection(
            res.stage3.clustering, units, matching, cons.selected)
        r = selection_eval.evaluate(ch, ref, picked, restrict_to=subset)
        sel_rows[name] = r.as_row()
        acc = "n/a" if r.accuracy is None else f"{r.accuracy:.4f}"
        print(f"  selection accuracy {name:32s} {acc}  "
              f"over {r.evaluated} events")

    for rule in ("longest", "centroid", "priority", "random"):
        import random as _r
        rng = _r.Random(13)
        picked = {}
        for e in ch.events:
            if not e.texts:
                continue
            from tavern.baselines import _pick
            t = _pick(e, rule, rng)
            picked[e.event_id] = next(b for b in e.books if e.texts[b] == t)
        r = selection_eval.evaluate(ch, ref, picked)
        sel_rows[f"Timeline+{rule.capitalize()} (curated)"] = r.as_row()
        print(f"  selection accuracy Timeline+{rule.capitalize():9s} (curated) "
              f"{r.accuracy:.4f} over {r.evaluated} events")
    _t, chosen = baselines.taeg_algorithm1(ch)
    r = selection_eval.evaluate(ch, ref, chosen)
    sel_rows["TAEG (Algorithm 1, curated)"] = r.as_row()
    print(f"  selection accuracy TAEG (Algorithm 1)          {r.accuracy:.4f} "
          f"over {r.evaluated} events")

    RESULTS["downstream"] = rows
    RESULTS["selection"] = sel_rows


def error_table(res, ch) -> None:
    banner("Error taxonomy (Table tab:res-qualerrors)")
    matching, _q = timeline_eval.match_clusters_to_events(
        ch, res.stage3.clustering, res.units)
    classes = error_analysis.analyse(ch, res.stage3.clustering, res.units,
                                     res.stage3.induced, matching, res.structs,
                                     res.segments)
    for c in classes:
        print(f"  {c.name:44s} {c.count:4d}   {c.example}")
    RESULTS["errors"] = [{"class": c.name, "count": c.count,
                          "example": c.example} for c in classes]


def ablation_table(base_cfg: TavernConfig, ch) -> None:
    banner("Ablations (Table tab:res-ablations)")
    ref = reference()
    rows = []

    configs = [
        ("TAVERN, full (induced)", {}),
        ("- veridicality partition", {"use_veridicality": False}),
        ("- closure", {"use_closure": False}),
        ("- anchor scaffold", {"use_anchor_scaffold": False}),
        ("- graph propagation", {"use_graph_propagation": False}),
    ]
    for label, over in configs:
        cfg = TavernConfig(**{**asdict(base_cfg), **over,
                              "tag": f"abl_{label.strip('- ').replace(' ', '_')}"})
        res = pipeline.run(cfg, with_gnn=True, write=False, verify=False)
        ev = timeline_eval.evaluate(ch, res.stage3.clustering,
                                    res.stage3.induced, res.units)
        cons = res.consolidation
        sc = content_metrics.rouge(cons.text, ref)
        matching, _q = timeline_eval.match_clusters_to_events(
            ch, res.stage3.clustering, res.units)
        subset = selection_eval.contested_intersection(
            ch, res.stage3.clustering, matching)
        picked = selection_eval.induced_selection(
            res.stage3.clustering, res.units, matching, cons.selected)
        sel = selection_eval.evaluate(ch, ref, picked, restrict_to=subset)
        row = {"config": label, "tau": ev.tau, "coverage": ev.coverage,
               "rougeL": sc["rougeL"], "rouge1": sc["rouge1"],
               "selection": sel.accuracy, "selection_n": sel.evaluated,
               "clusters": len(res.stage3.clustering.clusters)}
        rows.append(row)
        tau = "n/a" if ev.tau is None else f"{ev.tau:.4f}"
        acc = "n/a" if sel.accuracy is None else f"{sel.accuracy:.4f}"
        print(f"  {label:28s} tau={tau} cov={ev.coverage:.4f} "
              f"R-L={sc['rougeL']:.4f} sel={acc} ({sel.evaluated}) "
              f"clusters={row['clusters']}")

    # cascade ablation, level by level
    print()
    for levels in ((1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)):
        cfg = TavernConfig(**{**asdict(base_cfg), "cascade_levels": levels,
                              "tag": f"casc_{''.join(map(str, levels))}"})
        res = pipeline.run(cfg, with_gnn=False, write=False, verify=False)
        ev = timeline_eval.evaluate(ch, res.stage3.clustering,
                                    res.stage3.induced, res.units)
        sc = content_metrics.rouge(res.consolidation.text, ref)
        n_asserted = sum(len(s.asserted_tlinks()) for s in res.structs.values())
        tau = "n/a" if ev.tau is None else f"{ev.tau:.4f}"
        print(f"  cascade levels {str(levels):16s} asserted={n_asserted:5d} "
              f"tau={tau} cov={ev.coverage:.4f} R-L={sc['rougeL']:.4f}")
        rows.append({"config": f"cascade {levels}", "tau": ev.tau,
                     "coverage": ev.coverage, "rougeL": sc["rougeL"],
                     "asserted": n_asserted})
    RESULTS["ablations"] = rows


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--annotation", action="store_true")
    ap.add_argument("--consistency", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--degradation", action="store_true")
    ap.add_argument("--downstream", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--backbone", default="extractive",
                    choices=("extractive", "union", "bart", "pegasus",
                             "primera", "instruct", "ollama"),
                    help="Stage 5 backbone for the TAVERN rows. 'extractive' "
                         "keeps them comparable with the degradation curve, "
                         "whose rows are extractive; an abstractive backbone "
                         "measures the configuration the framework is actually "
                         "for, and both are reported when given.")
    ap.add_argument("--backbone-model", default="",
                    help="override the checkpoint or Ollama model name")
    ap.add_argument("--ollama-repeat-penalty", type=float, default=None,
                    help="A/B override for OllamaFuser's repeat_penalty "
                         "(see tavern/stage5_generation/backbones.py, "
                         "OLLAMA_REPEAT_PENALTY). Not a general tuning knob: "
                         "for measuring the effect of the backend-specific "
                         "decoding fix, both values reported side by side.")
    args = ap.parse_args()

    want = {k: getattr(args, k) for k in
            ("corpus", "annotation", "consistency", "timeline", "degradation",
             "downstream", "errors", "ablations")}
    if args.all or not any(want.values()):
        want = {k: True for k in want}

    extra = {}
    if args.ollama_repeat_penalty is not None:
        extra["ollama_repeat_penalty"] = args.ollama_repeat_penalty
    cfg = TavernConfig(tag=args.tag, backbone=args.backbone,
                       backbone_model=args.backbone_model, extra=extra)
    print(f"Running stages 1-5 (backbone '{args.backbone}') ...")
    res = pipeline.run(cfg, with_gnn=True, write=True)
    if res.backbone_note:
        print(f"  NOTE: {res.backbone_note}")
    RESULTS["backbone"] = {"requested": args.backbone,
                           "used": res.consolidation.backbone,
                           "model": args.backbone_model or None,
                           "note": res.backbone_note,
                           "ollama_repeat_penalty": args.ollama_repeat_penalty}
    ch = chrono_mod.load(res.corpus)

    if want["corpus"]:
        corpus_tables(res, ch)
    if want["annotation"]:
        annotation_tables(res)
    if want["consistency"]:
        consistency_table(res, ch)
    if want["timeline"]:
        timeline_tables(res, ch)
    if want["degradation"]:
        degradation_table(res, ch)
    if want["downstream"]:
        downstream_table(res, ch)
    if want["errors"]:
        error_table(res, ch)
    if want["ablations"]:
        ablation_table(cfg, ch)

    out = cfg.run_dir() / "results.json"
    out.write_text(json.dumps(RESULTS, indent=1, default=str),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
