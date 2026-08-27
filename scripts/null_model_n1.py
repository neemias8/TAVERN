#!/usr/bin/env python3
"""
N1 -- the positional-interleaving null model (Addendum 5, task 1).

No ISO-TimeML annotation, no cross-document scoring, no graph. Each
document's verse i gets position i/(n-1) on its own [0,1] axis; verses from
any document are grouped into the same canonical-event-sized cluster when
their positions fall in the same one of K windows over that shared axis;
clusters are ordered by window index; one account per cluster is emitted by
the same longest-account rule TAVERN's own extractive/longest baseline uses
(SelectionStrategy("longest") + ExtractiveFuser, unchanged, so N1 and TAVERN
are scored by identical downstream code).

K defaults to 169, the number of curated canonical events -- an externally
given target granularity, known independent of any measurement, NOT tuned
against N1's own tau/purity/R-L. This is meant to be the most competent
reasonable implementation of "no annotation", not a strawman: a null model
weakened by construction would prove nothing.

    python scripts/null_model_n1.py
    python scripts/null_model_n1.py --windows 169 --tag n1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_purity
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR
from tavern.stage1_preprocessing.corpus import Corpus
from tavern.stage3_anchoring_alignment.event_coref import Cluster, Clustering
from tavern.stage3_anchoring_alignment.global_timeline import InducedTimeline
from tavern.stage3_anchoring_alignment.local_timeline import EventUnit
from tavern.stage5_generation import ExtractiveFuser, SelectionStrategy, consolidate
from tavern.stage6_evaluation import chronology as chrono_mod
from tavern.stage6_evaluation import content_metrics, timeline_eval


def build(windows: int):
    corpus = Corpus()
    units: dict = {}
    positioned = []
    for book in corpus.books:
        verses = corpus.documents[book].verses
        n = len(verses)
        for i, v in enumerate(verses):
            uid = f"{book}:{v.chapter}:{v.number}"
            pos = (i / (n - 1)) if n > 1 else 0.0
            units[uid] = EventUnit(unit_id=uid, book=book, pericope_id=None,
                                   verse_keys=[v.key], text=v.text)
            positioned.append((pos, uid))

    bins: dict = {}
    for pos, uid in positioned:
        b = min(windows - 1, int(pos * windows))
        bins.setdefault(b, []).append(uid)

    clustering = Clustering()
    order = []
    for n, b in enumerate(sorted(bins), start=1):
        cid = f"n1_{n:04d}"
        members = bins[b]
        clustering.clusters.append(Cluster(
            cluster_id=cid, members=members,
            books={units[u].book for u in members},
            position=b / max(1, windows - 1)))
        for u in members:
            clustering.cluster_of_unit[u] = cid
        order.append(cid)

    induced = InducedTimeline(order=order,
                              rank={cid: i for i, cid in enumerate(order)})
    return corpus, units, clustering, induced


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=169,
                    help="bin count over the shared [0,1] position axis")
    ap.add_argument("--tag", default="n1")
    args = ap.parse_args()

    corpus, units, clustering, induced = build(args.windows)
    cons = consolidate(induced, clustering, units, fuser=ExtractiveFuser(),
                       strategy=SelectionStrategy("longest"), conflicts=[])

    out_dir = OUTPUT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    curation_path = out_dir / "curation.json"
    curation_path.write_text(json.dumps({"events": cons.records}, indent=1),
                             encoding="utf-8")
    cons.write(out_dir / "consolidated.txt")

    ch = chrono_mod.load(corpus)
    ref = reference()
    ev = timeline_eval.evaluate(ch, clustering, induced, units)
    sc = content_metrics.evaluate(cons.text, ref, with_meteor=False,
                                  with_bertscore=False)
    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    purity = cluster_purity.analyse(curation_path, v2e, e2day)

    row = {
        "config": "N1 (positional interleaving, no annotation)",
        "windows": args.windows,
        "units": len(units),
        "clusters": len(clustering.clusters),
        "tau": ev.tau, "pairwise_accuracy": ev.pairwise_accuracy,
        "coverage": ev.coverage, "matched_events": ev.matched_events,
        "total_events": ev.total_events,
        "purity": purity["purity"] if purity else None,
        "day_mixing": purity["day_mixing"] if purity else None,
        "multi_witness_clusters": purity["multi_witness_clusters"]
        if purity else None,
        "bcubed": purity["bcubed"] if purity else None,
        "rouge1": sc.rouge1, "rougeL": sc.rougeL, "length": sc.length,
    }
    (out_dir / "n1_summary.json").write_text(json.dumps(row, indent=1),
                                             encoding="utf-8")
    tau = "n/a" if row["tau"] is None else f"{row['tau']:.4f}"
    pw = "n/a" if row["pairwise_accuracy"] is None \
        else f"{row['pairwise_accuracy']:.4f}"
    b = row["bcubed"] or {}
    print(f"N1  windows={args.windows}  units={row['units']}  "
          f"clusters={row['clusters']}")
    print(f"    tau={tau}  pairwise={pw}  coverage={row['coverage']:.4f}  "
          f"matched={row['matched_events']}/{row['total_events']}")
    print(f"    purity={row['purity']}  day_mix={row['day_mixing']}  "
          f"multi_witness={row['multi_witness_clusters']}")
    if b:
        print(f"    B-cubed P={b['precision']}  R={b['recall']}  "
              f"F1={b['f1']}")
    print(f"    R-1={row['rouge1']:.4f}  R-L={row['rougeL']:.4f}  "
          f"len={row['length']}")
    print(f"\nwrote {out_dir / 'n1_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
