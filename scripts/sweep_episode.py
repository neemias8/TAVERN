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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_purity
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


def run_one(max_episode_verses, tag: str, full_checks: bool = False,
           no_anchor_credit: bool = False, v2e=None, e2day=None) -> dict:
    """Run stages 1-5 at one `max_episode_verses` setting and measure it.

    `full_checks=True` additionally runs the six consistency checks and the
    inter-document conflict report, for the Task 3 side-by-side comparison.
    `no_anchor_credit=True` runs H-B (Addendum 4/5) instead of the published
    default. `v2e`/`e2day` (from cluster_purity.verse_to_event) are accepted
    so a multi-row caller parses the chronology XML once, not per row; purity
    needs curation.json on disk, hence write=True here.
    """
    cfg = TavernConfig(tag=tag, backbone="extractive",
                       max_episode_verses=max_episode_verses,
                       no_anchor_credit=no_anchor_credit)
    res = pipeline.run(cfg, with_gnn=True, write=True, verify=True)
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
    if v2e is None or e2day is None:
        v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    purity = cluster_purity.analyse(cfg.run_dir() / "curation.json", v2e, e2day)

    row = {
        "max_episode_verses": "None" if max_episode_verses is None
                              else max_episode_verses,
        "no_anchor_credit": no_anchor_credit,
        "purity": purity["purity"] if purity else None,
        "day_mixing": purity["day_mixing"] if purity else None,
        "multi_witness_clusters": purity["multi_witness_clusters"]
        if purity else None,
        "bcubed": purity["bcubed"] if purity else None,
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


def _print_row(row: dict) -> None:
    b = row.get("bcubed") or {}
    b3 = (f"P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}"
          if b else "n/a")
    print(f"    clusters={row['clusters']}  tau={row['tau']:.4f}  "
          f"cov={row['coverage']:.4f}  purity={row['purity']}  "
          f"day_mix={row['day_mixing']}  B3[{b3}]  under={row['under_merged']}  "
          f"over={row['over_merged']}  R-1={row['rouge1']:.4f}  "
          f"R-L={row['rougeL']:.4f}")


def cmd_sweep(args) -> None:
    out_dir = OUTPUT_DIR / "sweep_episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    suffix = "_hb" if args.no_anchor_credit else ""
    rows = []
    for v in SWEEP_VALUES:
        tag = f"sweep_episode_{v if v is not None else 'none'}{suffix}"
        print(f"--- max_episode_verses={v} (tag={tag})")
        row = run_one(v, tag, no_anchor_credit=args.no_anchor_credit,
                     v2e=v2e, e2day=e2day)
        rows.append(row)
        _print_row(row)

    out = out_dir / f"summary{suffix}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    fields = ["max_episode_verses", "no_anchor_credit", "clusters",
              "contested_clusters", "purity", "day_mixing",
              "multi_witness_clusters", "tau", "pairwise_accuracy",
              "coverage", "matched_events", "total_events", "under_merged",
              "over_merged", "not_aligned", "not_detected",
              "misordered_adjacent", "misordered_displaced",
              "subordinated_leak", "rouge1", "rougeL", "length"]
    with open(out_dir / f"summary{suffix}.csv", "w", newline="",
             encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} and summary{suffix}.csv")


def cmd_compare(args) -> None:
    out_dir = OUTPUT_DIR / "sweep_episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    configs = [("base", 2, False), ("H-B", 2, True),
              ("pericope", None, False), ("best_sweep", args.best, False)]
    rows = []
    for label, v, hb in configs:
        print(f"--- {label}: max_episode_verses={v} no_anchor_credit={hb}")
        row = run_one(v, f"cmp_{label}", full_checks=True,
                     no_anchor_credit=hb, v2e=v2e, e2day=e2day)
        row["label"] = label
        rows.append(row)
        _print_row(row)
        print(f"    consistency={row['consistency_pass']}  "
              f"inter_doc_conflicts={row['inter_document_conflicts']}")
    (out_dir / "compare.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir / 'compare.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--no-anchor-credit", action="store_true",
                         help="run the sweep with H-B (Addendum 4/5) active "
                              "instead of the published default")
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
