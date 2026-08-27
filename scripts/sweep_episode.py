#!/usr/bin/env python3
"""
Sweep of `max_episode_verses`, the per-document span cap in `_mergeable`
(tavern/stage3_anchoring_alignment/event_coref.py), and the pericope-boundary
variant that replaces it (`max_episode_verses=None`).

    python scripts/sweep_episode.py sweep
    python scripts/sweep_episode.py compare --best 8

Nothing here reads the chronology outside Stage 6 evaluation calls, which is
where the rest of the framework reads it too. No abstractive backbone is used:
the questions this script answers (cluster count, error taxonomy, induced
tau/coverage, extractive R-L) don't need one, and Ollama would cost hours for
no change in any of those figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR, TavernConfig
from tavern.stage6_evaluation import (chronology as chrono_mod,
                                      conflicts as conflicts_mod, consistency,
                                      content_metrics, error_analysis,
                                      timeline_eval)

SWEEP_VALUES = [2, 3, 4, 6, 8, 12, None]


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def run_one(max_episode_verses, tag: str, full_checks: bool = False) -> dict:
    """Run stages 1-5 at one `max_episode_verses` setting and measure it.

    `full_checks=True` additionally runs the six consistency checks and the
    inter-document conflict report, for the Task 3 side-by-side comparison.
    """
    cfg = TavernConfig(tag=tag, backbone="extractive",
                       max_episode_verses=max_episode_verses)
    res = pipeline.run(cfg, with_gnn=True, write=False, verify=True)
    ch = chrono_mod.load(res.corpus)
    ref = reference()

    ev = timeline_eval.evaluate(ch, res.stage3.clustering, res.stage3.induced,
                                res.units)
    st = res.stage3.stats()
    matching, _q = timeline_eval.match_clusters_to_events(
        ch, res.stage3.clustering, res.units)
    classes = error_analysis.analyse(ch, res.stage3.clustering, res.units,
                                     res.stage3.induced, matching, res.structs,
                                     res.segments)
    errors = {c.name: c.count for c in classes}
    sc = content_metrics.evaluate(res.consolidation.text, ref,
                                  with_meteor=False, with_bertscore=False)

    row = {
        "max_episode_verses": "None" if max_episode_verses is None
                              else max_episode_verses,
        "clusters": st["clusters"],
        "contested_clusters": st["contested_clusters"],
        "cluster_size_hist": st["cluster_size_hist"],
        "tau": ev.tau,
        "pairwise_accuracy": ev.pairwise_accuracy,
        "coverage": ev.coverage,
        "matched_events": ev.matched_events,
        "total_events": ev.total_events,
        "under_merged": errors["Under-merged cluster (one event split)"],
        "over_merged": errors["Over-merged cluster (two events joined)"],
        "not_aligned": errors["Detected but not aligned across documents"],
        "not_detected": errors["Event not detected"],
        "misordered_adjacent": errors["Misordered: adjacent transposition"],
        "misordered_displaced":
            errors["Misordered: displaced across a day boundary"],
        "subordinated_leak":
            errors["Subordinated event admitted to the timeline"],
        "rouge1": sc.rouge1,
        "rougeL": sc.rougeL,
        "length": sc.length,
    }

    if full_checks:
        rep = consistency.run(res.structs, res.reports, res.corpus,
                              res.stage3.induced.conflicts,
                              res.stage3.clustering, res.units, res.segments)
        checks = rep.as_rows()
        n_pass = sum(1 for c in checks if c["passed"] is True)
        n_checked = sum(1 for c in checks if c["passed"] is not None)
        crep = conflicts_mod.report(res.structs, res.stage3.induced,
                                    res.stage3.clustering, res.units)
        inter = next((r["count"] for r in crep.as_rows()
                     if r["class"].startswith("Inter-document")), None)
        row["consistency_checks"] = checks
        row["consistency_pass"] = f"{n_pass}/{n_checked}"
        row["inter_document_conflicts"] = inter
        row["documented_recovered"] = crep.detail

    return row


def cmd_sweep(args) -> None:
    out_dir = OUTPUT_DIR / "sweep_episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in SWEEP_VALUES:
        tag = f"sweep_episode_{v if v is not None else 'none'}"
        print(f"--- max_episode_verses={v} (tag={tag})")
        row = run_one(v, tag)
        rows.append(row)
        print(f"    clusters={row['clusters']}  tau={row['tau']:.4f}  "
              f"cov={row['coverage']:.4f}  under={row['under_merged']}  "
              f"over={row['over_merged']}  R-1={row['rouge1']:.4f}  "
              f"R-L={row['rougeL']:.4f}")

    (out_dir / "summary.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    fields = ["max_episode_verses", "clusters", "contested_clusters", "tau",
              "pairwise_accuracy", "coverage", "matched_events",
              "total_events", "under_merged", "over_merged", "not_aligned",
              "not_detected", "misordered_adjacent", "misordered_displaced",
              "subordinated_leak", "rouge1", "rougeL", "length"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir / 'summary.json'} and summary.csv")


def cmd_compare(args) -> None:
    out_dir = OUTPUT_DIR / "sweep_episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [("base", 2), ("pericope", None), ("best_sweep", args.best)]
    rows = []
    for label, v in configs:
        print(f"--- {label}: max_episode_verses={v}")
        row = run_one(v, f"cmp_{label}", full_checks=True)
        row["label"] = label
        rows.append(row)
        print(f"    clusters={row['clusters']}  tau={row['tau']:.4f}  "
              f"cov={row['coverage']:.4f}  under={row['under_merged']}  "
              f"over={row['over_merged']}  R-1={row['rouge1']:.4f}  "
              f"R-L={row['rougeL']:.4f}  "
              f"consistency={row['consistency_pass']}  "
              f"inter_doc_conflicts={row['inter_document_conflicts']}")
    (out_dir / "compare.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir / 'compare.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sweep")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--best", type=int, default=None,
                       help="max_episode_verses for the 'best_sweep' row "
                            "(omit for None / pericope-only)")
    args = ap.parse_args()
    if args.cmd == "sweep":
        cmd_sweep(args)
    else:
        cmd_compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
