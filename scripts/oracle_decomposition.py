#!/usr/bin/env python3
"""
Addendum 7, Task 3 -- decompose Stage 3's contribution into grouping
(cluster_units) and ordering (induce), by crossing induced/oracle on each
axis independently.

  A (canonical)  induced clustering, induced ordering  -- the real system
  B              oracle clustering,  induced ordering  -- perfect linking,
                 real tournament: isolates what ordering alone costs
  C              induced clustering, oracle ordering   -- real linking,
                 perfect order: isolates what grouping alone costs
  D (oracle)     oracle clustering,  oracle ordering    -- ceiling.

This is the "oracle timeline" configuration the thesis describes -- and, as
of Addendum 7 Task 1, the ONLY place it exists. TavernConfig used to carry a
`use_oracle_timeline` flag; it was declared and never read anywhere, a dead
switch. It has been removed rather than wired up, because wiring it up
inside pipeline.run would require pipeline.py (Stage 1-5) to read the
chronology, which config.assert_no_chronology_import() exists specifically
to refuse -- correctly, in this case. The oracle configuration is a Stage 6
BENCH: it takes real Stage 1-3 annotation and units (via pipeline.prepare /
segment_corpus, chronology-free) and substitutes the chronology-built
oracle clustering/ordering for the induced ones only here, downstream of the
guard, the same reasoning as cluster_purity.py and oracle_roundtrip.py. If a
pipeline-level oracle configuration is wanted later, it has to be built this
way -- a Stage 6 driver that calls into Stage 1-3 for annotation and units,
never the reverse.

B reuses global_timeline.induce() directly -- the real registration/
build_cluster_graph/minimum_feedback_arc_set/topological_sort pipeline --
on the oracle clustering, so "induced ordering" in B is the actual Stage 3
algorithm, not a stand-in.

C reorders the REAL induced clustering by the curated event each cluster
best matches (timeline_eval.match_clusters_to_events, the same greedy
Jaccard+recall matcher everything else in Stage 6 uses); clusters with no
match keep their own induced rank as a tiebreak, appended after every
matched cluster, and the unmatched count is reported, not hidden.

    python scripts/oracle_decomposition.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_purity
from oracle_roundtrip import build_oracle
from tavern import pipeline
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR, TavernConfig
from tavern.stage3_anchoring_alignment.global_timeline import InducedTimeline, induce
from tavern.stage5_generation import ExtractiveFuser, SelectionStrategy, consolidate
from tavern.stage6_evaluation import (chronology as chrono_mod, content_metrics,
                                      redundancy, selection_eval, timeline_eval)


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def measure(label, units, clustering, induced_tl, ch, matching_override=None):
    ref = reference()
    ev = timeline_eval.evaluate(ch, clustering, induced_tl, units)

    if matching_override is not None:
        matching = matching_override
    else:
        matching, _q = timeline_eval.match_clusters_to_events(ch, clustering, units)

    cons = consolidate(induced_tl, clustering, units, fuser=ExtractiveFuser(),
                       strategy=SelectionStrategy("longest"), conflicts=[])
    sc = content_metrics.evaluate(cons.text, ref, with_meteor=False,
                                  with_bertscore=False)

    subset = selection_eval.contested_intersection(ch, clustering, matching)
    picked = selection_eval.induced_selection(clustering, units, matching,
                                              cons.selected)
    sel = selection_eval.evaluate(ch, ref, picked, restrict_to=subset)

    out_dir = OUTPUT_DIR / "oracle_decomposition" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    curation_path = out_dir / "curation.json"
    curation_path.write_text(json.dumps({"events": cons.records}, indent=1),
                             encoding="utf-8")
    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    purity = cluster_purity.analyse(curation_path, v2e, e2day)
    pairs = [(rec.get("consolidated", ""),
             [s.get("text", "") for s in (rec.get("sources") or [])])
            for rec in cons.records]
    cov = redundancy.coverage_over_events(pairs)

    row = {
        "config": label, "clusters": len(clustering.clusters),
        **ev.as_row(), "rouge1": sc.rouge1, "rougeL": sc.rougeL,
        "length": sc.length,
        "purity": purity["purity"] if purity else None,
        "bcubed": purity["bcubed"] if purity else None,
        "coverage_over_events": cov,
        "selection_accuracy": sel.accuracy, "selection_n": sel.evaluated,
    }
    b3 = row["bcubed"] or {}
    print(f"{label:12s} clusters={row['clusters']:4d} tau={row['tau']}  "
         f"cov={row['coverage']}  R-1={row['rouge1']:.4f}  "
         f"R-L={row['rougeL']:.4f}  purity={row['purity']}  "
         f"B3-F1={b3.get('f1')}  content_cov={cov.get('coverage')}  "
         f"sel={row['selection_accuracy']} ({row['selection_n']})")
    return row


def main() -> int:
    cfg = TavernConfig(tag="decomp_induced", backbone="extractive")

    # ---- A: induced clustering, induced ordering (the real system) -------
    res = pipeline.run(cfg, with_gnn=False, write=False, verify=True)
    units_a = res.units
    clustering_a = res.stage3.clustering
    induced_a = res.stage3.induced
    ch = chrono_mod.load(res.corpus)
    row_a = measure("A_induced_induced", units_a, clustering_a, induced_a, ch)

    # ---- oracle building blocks (shared by B and D) -----------------------
    units_o, sc_o, clustering_o, induced_o_trivial, ch2, corpus_o = build_oracle(
        TavernConfig(tag="decomp_oracle", backbone="extractive"))

    # ---- D: oracle clustering, oracle ordering (the ceiling) --------------
    matching_d = {}
    for cl in clustering_o.clusters:
        eid = int(cl.cluster_id.split("_")[1])
        matching_d[eid] = cl.cluster_id
    row_d = measure("D_oracle_oracle", units_o, clustering_o,
                    induced_o_trivial, ch2, matching_override=matching_d)

    # ---- B: oracle clustering, INDUCED ordering ---------------------------
    # real Stage 3 timelines/structs, needed by induce()'s registration and
    # conflict detection -- rebuilt once, cheaply (extractive, no GNN)
    from tavern.stage3_anchoring_alignment.local_timeline import segment_corpus
    corpus_b, pericopes_b, segments_b, chains_b, structs_b, reports_b = \
        pipeline.prepare(TavernConfig(tag="decomp_b", backbone="extractive"),
                         verify=True)
    timelines_b = segment_corpus(structs_b, corpus_b, pericopes_b, chains_b)
    induced_b = induce(timelines_b, clustering_o, sc_o, intra_conflicts=[],
                       structs=structs_b)
    row_b = measure("B_oracle_induced", units_o, clustering_o, induced_b, ch2,
                    matching_override=matching_d)

    # ---- C: INDUCED clustering, oracle ordering ----------------------------
    matching_a, _q = timeline_eval.match_clusters_to_events(ch, clustering_a,
                                                            units_a)
    ch_rank = ch.rank()
    real_rank = induced_a.rank
    matched_cids = {cid for cid in matching_a.values()}
    order_key = {}
    for cl in clustering_a.clusters:
        cid = cl.cluster_id
        eid = next((e for e, c in matching_a.items() if c == cid), None)
        if eid is not None:
            order_key[cid] = (0, ch_rank[eid])
        else:
            order_key[cid] = (1, real_rank.get(cid, 10 ** 6))
    order_c = sorted((cl.cluster_id for cl in clustering_a.clusters),
                     key=lambda cid: order_key[cid])
    n_unmatched = sum(1 for cl in clustering_a.clusters
                      if cl.cluster_id not in matched_cids)
    print(f"C: {len(matched_cids)}/{len(clustering_a.clusters)} clusters "
         f"matched to a curated event ({n_unmatched} kept their own induced "
         f"rank as a tiebreak)")
    induced_c = InducedTimeline(order=order_c,
                                rank={cid: i for i, cid in enumerate(order_c)})
    row_c = measure("C_induced_oracle", units_a, clustering_a, induced_c, ch,
                    matching_override=matching_a)

    rows = [row_a, row_b, row_c, row_d]
    out_dir = OUTPUT_DIR / "oracle_decomposition"
    (out_dir / "decomposition_summary.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")

    print("\n--- decomposition ---")
    for label, key in (("tau", "tau"), ("coverage", "coverage"),
                       ("rougeL", "rougeL"), ("selection_accuracy",
                                              "selection_accuracy")):
        vals = {r["config"]: r.get(key) for r in rows}
        print(f"  {label}: {vals}")

    print(f"\nwrote {out_dir / 'decomposition_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
