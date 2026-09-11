#!/usr/bin/env python3
"""
Addendum 17 -- dispersion of selection accuracy (and R-1/R-2/R-L/METEOR)
across repeated, independently-seeded GNN training runs, holding Stage 1-3
fixed per configuration.

Stages 1-3 are deterministic (the Addendum 16 control confirmed: tau,
coverage, clusters, purity, B-cubed, conflicts and the error taxonomy are
bit-identical across runs). Only Stage 4 (GNN training, `mean_over_seeds`)
and Stage 5's graph-score selection are re-run per sample, N times per
configuration, reusing the SAME Stage 3 Clustering/InducedTimeline/
EventGraph each time -- so this measures exactly the quantity the Addendum
16 control sampled twice (0.3600, 0.3733), now with enough samples for a
mean and a standard deviation instead of two points.

R-1/R-2/R-L/METEOR are computed the same way the control's varying
"graph score" downstream row was: ExtractiveFuser + SelectionStrategy(
"graph_score", scores=...) -- not the Ollama abstractive fusion, which the
control found stable (fusion-cache hit) and which would cost an LLM call
per sample for no diagnostic reason.

    python scripts/seed_sweep_selection.py            # N=10
    python scripts/seed_sweep_selection.py --n 20

Configurations match Table tab:res-repair-ablation:
  before          pre-repair (R1+R2+R3 all reverted together)
  full            adopted (ancoragem)
  - projection    R1  (disable_projection)
  - indexing      R2  (disable_projection_indexing)
  - participants  R3  (legacy_participants)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR, TavernConfig
from tavern.stage3_anchoring_alignment import run as run_stage3
from tavern.stage4_gnn import mean_over_seeds
from tavern.stage5_generation import ExtractiveFuser, SelectionStrategy, consolidate
from tavern.stage6_evaluation import (chronology as chrono_mod, content_metrics,
                                      selection_eval, timeline_eval)

CONFIGS = {
    "before": dict(disable_projection=True, disable_projection_indexing=True,
                   legacy_participants=True),
    "full": dict(),
    "- projection": dict(disable_projection=True),
    "- indexing": dict(disable_projection_indexing=True),
    "- participants": dict(legacy_participants=True),
}


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def stats(values: List[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def run_config(label: str, overrides: dict, n: int, ref: str) -> dict:
    cfg = TavernConfig(tag=f"seedsweep_{label.strip('- ').replace(' ', '_')}",
                       backbone="extractive", **overrides)
    pipeline._apply_granularity(cfg)
    corpus, pericopes, segments, chains, structs, reports = pipeline.prepare(
        cfg, verify=True)
    stage3 = run_stage3(structs, corpus, pericopes, chains, cfg, write=False)
    units = stage3.graph.node_units
    ch = chrono_mod.load(corpus)
    matching, _q = timeline_eval.match_clusters_to_events(
        ch, stage3.clustering, units)
    subset = selection_eval.contested_intersection(ch, stage3.clustering, matching)
    n_versions = {e.event_id: e.n_versions for e in ch.events}
    floor = statistics.fmean(1.0 / n_versions[eid] for eid in subset)

    sel_acc: List[float] = []
    r1s: List[float] = []
    r2s: List[float] = []
    rls: List[float] = []
    mets: List[float] = []
    for i in range(n):
        gnn = mean_over_seeds(stage3.graph, cfg)
        cons = consolidate(stage3.induced, stage3.clustering, units,
                           fuser=ExtractiveFuser(),
                           strategy=SelectionStrategy("graph_score",
                                                       scores=gnn.node_scores),
                           conflicts=[])
        sc = content_metrics.evaluate(cons.text, ref, with_meteor=True,
                                      with_bertscore=False)
        picked = selection_eval.induced_selection(stage3.clustering, units,
                                                   matching, cons.selected)
        sel = selection_eval.evaluate(ch, ref, picked, restrict_to=subset)
        sel_acc.append(sel.accuracy)
        r1s.append(sc.rouge1)
        r2s.append(sc.rouge2)
        rls.append(sc.rougeL)
        mets.append(sc.meteor)
        print(f"  {label:16s} run {i + 1:2d}/{n}: "
             f"selection={sel.accuracy:.4f}  R-L={sc.rougeL:.4f}", flush=True)

    return {
        "config": label, "clusters": len(stage3.clustering.clusters),
        "selection_subset": len(subset), "selection_floor": round(floor, 4),
        "selection_accuracy": stats(sel_acc),
        "rouge1": stats(r1s), "rouge2": stats(r2s), "rougeL": stats(rls),
        "meteor": stats(mets),
        "raw": {"selection_accuracy": sel_acc, "rouge1": r1s, "rouge2": r2s,
               "rougeL": rls, "meteor": mets},
    }


def latex_table(rows: List[dict]) -> str:
    lines = [
        "% Addendum 17 -- selection-accuracy dispersion across N seeded GNN runs.",
        "% Generated by scripts/seed_sweep_selection.py; do not hand-edit.",
        "\\begin{table}[h]\n\\centering",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Configuration & floor & mean & stdev & min & max \\\\",
        "\\midrule",
    ]
    for r in rows:
        s = r["selection_accuracy"]
        lines.append(
            f"{r['config']} & {r['selection_floor']:.4f} & {s['mean']:.4f} & "
            f"{s['stdev']:.4f} & {s['min']:.4f} & {s['max']:.4f} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        f"\\caption{{Selection accuracy, mean/stdev/min/max over "
        f"{rows[0]['selection_accuracy']['n']} independently seeded GNN "
        "training runs per configuration, Stage 1-3 held fixed.}}",
        "\\label{tab:seed-sweep-selection}", "\\end{table}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    ref = reference()
    rows = []
    for label, overrides in CONFIGS.items():
        print(f"=== {label} ===", flush=True)
        rows.append(run_config(label, overrides, args.n, ref))

    out_dir = OUTPUT_DIR / "seedsweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps(rows, indent=1),
                                            encoding="utf-8")
    (out_dir / "selection_table.tex").write_text(latex_table(rows),
                                                  encoding="utf-8")

    print("\n=== SUMMARY ===")
    for r in rows:
        s = r["selection_accuracy"]
        print(f"{r['config']:16s} floor={r['selection_floor']:.4f}  "
             f"mean={s['mean']:.4f}  stdev={s['stdev']:.4f}  "
             f"min={s['min']:.4f}  max={s['max']:.4f}  "
             f"(n={s['n']}, subset={r['selection_subset']})")
    print(f"\nwrote {out_dir / 'selection.json'}")
    print(f"wrote {out_dir / 'selection_table.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
