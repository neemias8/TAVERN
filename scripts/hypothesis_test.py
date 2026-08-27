#!/usr/bin/env python3
"""
Tests H-A, H-B, H-C against cluster purity, one at a time, then a combination
of whichever moved purity. See Addendum 3 to Briefing 2 (2026-08-26).

    python scripts/hypothesis_test.py single
    python scripts/hypothesis_test.py combine --with H-A H-C

Extractive backbone throughout: purity, cluster count, tau, coverage and the
error taxonomy are Stage 3 properties, independent of Stage 5 generation, so
Ollama would cost hours here for no change in any of them.

Each config gets its own overrides on top of the published defaults; nothing
here searches over MATCH_THRESHOLD, the WEIGHTS or ANCHOR_BAND for a value
that maximises purity or R-L. Each hypothesis is one fixed, argued setting,
run once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_purity
from tavern import pipeline
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR, TavernConfig
from tavern.stage6_evaluation import (chronology as chrono_mod, content_metrics,
                                      error_analysis, timeline_eval)

# H-A: class/modal as multiplicative {0,1} vetoes on disagreement, not
#      additive scores for agreement; anchor term dropped, predicate/
#      participants renormalised to 0.62/0.38 (event_coref.GATED_SCORE).
# H-B: a pair with no scaffold position on either side no longer gets the
#      automatic 0.4 anchor credit; its 0.15 weight moves into predicate/
#      participants for that pair (event_coref.NO_ANCHOR_CREDIT).
# H-C: the scaffold's claim is day-level placement, so the band should be the
#      scaffold's own resolution, +-1 day, not +-4 (TavernConfig.anchor_band,
#      already wired -- no code change).
SINGLE_CONFIGS = {
    "base": {},
    "H-A": {"gated_score": True},
    "H-B": {"no_anchor_credit": True},
    "H-C": {"anchor_band": 1.0},
}


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def run_one(tag: str, overrides: dict, v2e, e2day) -> dict:
    cfg = TavernConfig(tag=tag, backbone="extractive", **overrides)
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
    purity = cluster_purity.analyse(cfg.run_dir() / "curation.json", v2e, e2day)

    row = {
        "config": tag,
        "overrides": overrides,
        "purity": purity["purity"] if purity else None,
        "day_mixing": purity["day_mixing"] if purity else None,
        "multi_witness_clusters": purity["multi_witness_clusters"]
        if purity else None,
        "clusters": st["clusters"],
        "under_merged": errors["Under-merged cluster (one event split)"],
        "over_merged": errors["Over-merged cluster (two events joined)"],
        "tau": ev.tau,
        "coverage": ev.coverage,
        "rouge1": sc.rouge1,
        "rougeL": sc.rougeL,
    }
    print(f"  {tag:8s} purity={row['purity']}  day_mix={row['day_mixing']}  "
          f"clusters={row['clusters']}  under={row['under_merged']}  "
          f"over={row['over_merged']}  tau={row['tau']:.4f}  "
          f"cov={row['coverage']:.4f}  R-1={row['rouge1']:.4f}  "
          f"R-L={row['rougeL']:.4f}")
    return row


def cmd_single(args) -> None:
    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    out_dir = OUTPUT_DIR / "hypothesis_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one(tag, ov, v2e, e2day)
            for tag, ov in SINGLE_CONFIGS.items()]
    (out_dir / "single.json").write_text(json.dumps(rows, indent=1),
                                         encoding="utf-8")
    print(f"\nwrote {out_dir / 'single.json'}")


def cmd_combine(args) -> None:
    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    out_dir = OUTPUT_DIR / "hypothesis_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {}
    for label in args.with_:
        overrides.update(SINGLE_CONFIGS[label])
    tag = "combine_" + "_".join(args.with_)
    row = run_one(tag, overrides, v2e, e2day)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(row, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("single")
    p_c = sub.add_parser("combine")
    p_c.add_argument("--with", dest="with_", nargs="+",
                     choices=list(SINGLE_CONFIGS), required=True)
    args = ap.parse_args()
    if args.cmd == "single":
        cmd_single(args)
    else:
        cmd_combine(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
