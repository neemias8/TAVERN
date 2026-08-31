#!/usr/bin/env python3
"""
Addendum 11, Task 2 -- profile seed-order sensitivity, verified against an
artefact instead of cited from memory.

Section 10.12 reports tau in [0.898, 0.922] and ROUGE-L in [0.524, 0.594]
across five profile entry orders, from a run that predates every zip this
repository has shipped and could not be reconciled when Chapter 10 was last
checked against artefacts -- flagged there as unverifiable, and it stayed in
the text anyway. This reproduces it against the current (ancoragem) code:
extractive backbone, "longest"-rule selection, matching
`oracle_decomposition.py`'s own "A" cell methodology (its magnitude -- R-L
around 0.59-0.66 -- is what the cited range's upper end matches).

`event_coref._SEED_ORDER` overrides `cluster_units`'s book-entry order for
the profile; `_apply_granularity` never touches it, so setting it directly
before `pipeline.run()` is the same pattern the ablation flags use.
`pipeline.prepare()`'s cache is keyed on stage 1-2 config only, so the four
extra runs here reuse the same annotation instead of re-annotating five times.

    python scripts/seed_order_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_purity
from tavern import pipeline
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR, TavernConfig
from tavern.stage3_anchoring_alignment import event_coref as ec
from tavern.stage5_generation import ExtractiveFuser, SelectionStrategy, consolidate
from tavern.stage6_evaluation import chronology as chrono_mod, content_metrics, timeline_eval

ORDERS = [
    ("MtMkLkJn (canonical)", ["matthew", "mark", "luke", "john"]),
    ("MkMtLkJn", ["mark", "matthew", "luke", "john"]),
    ("LkMtMkJn", ["luke", "matthew", "mark", "john"]),
    ("JnMtMkLk", ["john", "matthew", "mark", "luke"]),
    ("MtLkMkJn", ["matthew", "luke", "mark", "john"]),
]


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def main() -> int:
    ref = reference()
    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    rows = []
    for label, order in ORDERS:
        tag = "seed_" + label.split()[0]
        ec._SEED_ORDER = (lambda timelines, _o=order:
                          [b for b in _o if b in timelines])
        cfg = TavernConfig(tag=tag, backbone="extractive")
        res = pipeline.run(cfg, with_gnn=False, write=False, verify=True)
        ch = chrono_mod.load(res.corpus)
        ev = timeline_eval.evaluate(ch, res.stage3.clustering,
                                    res.stage3.induced, res.units)
        cons = consolidate(res.stage3.induced, res.stage3.clustering,
                           res.units, fuser=ExtractiveFuser(),
                           strategy=SelectionStrategy("longest"), conflicts=[])
        sc = content_metrics.evaluate(cons.text, ref, with_meteor=False,
                                      with_bertscore=False)
        curation_path = OUTPUT_DIR / tag / "curation.json"
        curation_path.parent.mkdir(parents=True, exist_ok=True)
        curation_path.write_text(
            json.dumps({"events": cons.records}, indent=1), encoding="utf-8")
        purity = cluster_purity.analyse(curation_path, v2e, e2day)
        row = {
            "order": label, "tau": ev.tau, "pairwise": ev.pairwise_accuracy,
            "coverage": ev.coverage,
            "clusters": len(res.stage3.clustering.clusters),
            "rougeL": sc.rougeL,
            "purity": purity["purity"] if purity else None,
        }
        rows.append(row)
        print(f"{label:22s} tau={row['tau']:.4f} pairwise={row['pairwise']:.4f} "
             f"cov={row['coverage']:.4f} clusters={row['clusters']:4d} "
             f"R-L={row['rougeL']:.4f} purity={row['purity']}")

    ec._SEED_ORDER = None  # restore the default before this process exits
    out_dir = OUTPUT_DIR / "seed_order_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1),
                                          encoding="utf-8")
    print(f"\nwrote {out_dir / 'summary.json'}")

    taus = [r["tau"] for r in rows]
    rls = [r["rougeL"] for r in rows]
    print(f"tau range: [{min(taus):.4f}, {max(taus):.4f}]  "
         f"(thesis cites [0.898, 0.922])")
    print(f"R-L range: [{min(rls):.4f}, {max(rls):.4f}]  "
         f"(thesis cites [0.524, 0.594])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
