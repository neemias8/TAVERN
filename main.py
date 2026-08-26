#!/usr/bin/env python3
"""
Run stages 1 to 5 and write the consolidated narrative.

This entry point never touches the chronology or the reference consolidation.
Use `run_experiments.py` for anything that needs them -- that is where Stage 6
lives, and `config.assert_no_chronology_import` enforces the separation.

    python main.py                       # the configuration of the thesis
    python main.py --projection absolute --year 33
    python main.py --no-veridicality --no-scaffold   # ablations
"""
from __future__ import annotations

import argparse
import json
import sys

from tavern import pipeline
from tavern.config import TavernConfig, verify_corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run",
                    help="output directory under outputs/")
    ap.add_argument("--projection", choices=("relative", "absolute"),
                    default="relative",
                    help="domain-profile projection mode (Appendix A)")
    ap.add_argument("--year", type=int, default=30,
                    help="Julian year for absolute mode (30 or 33)")
    ap.add_argument("--backbone", default="union",
                    choices=("union", "extractive", "bart", "pegasus",
                             "primera", "instruct", "ollama"),
                    help="Stage 5 fusion backbone. 'union' is deterministic "
                         "and detail-preserving but not abstractive; "
                         "'extractive' emits one account verbatim and exists "
                         "for comparison with the degradation curve; the rest "
                         "are abstractive and need a model.")
    ap.add_argument("--backbone-model", default="",
                    help="override the checkpoint or Ollama model name")
    ap.add_argument("--no-gnn", action="store_true",
                    help="skip Stage 4; select by account length")
    ap.add_argument("--no-veridicality", action="store_true")
    ap.add_argument("--no-closure", action="store_true")
    ap.add_argument("--no-scaffold", action="store_true")
    ap.add_argument("--no-propagation", action="store_true")
    ap.add_argument("--cascade", default="1,2,3,4",
                    help="cascade levels to apply, comma-separated")
    args = ap.parse_args()

    cfg = TavernConfig(
        tag=args.tag,
        projection_mode=args.projection,
        absolute_year=args.year,
        use_veridicality=not args.no_veridicality,
        use_closure=not args.no_closure,
        use_anchor_scaffold=not args.no_scaffold,
        use_graph_propagation=not args.no_propagation,
        cascade_levels=tuple(int(x) for x in args.cascade.split(",") if x),
        backbone=args.backbone,
        backbone_model=args.backbone_model,
    )

    print("Verifying corpus digests ...")
    for name, (status, _exp, _act) in verify_corpus(strict=False).items():
        print(f"  {status:8s} {name}")
    verify_corpus()

    print("\nRunning stages 1-5 ...")
    res = pipeline.run(cfg, with_gnn=not args.no_gnn, write=True)

    out = cfg.run_dir()
    counts = {b: s.counts() for b, s in res.structs.items()}
    total = {k: sum(v[k] for v in counts.values()) for k in
             next(iter(counts.values()))}
    st = res.stage3.stats()

    print(f"\nAnnotation      {total['event_total']} events, "
          f"{total['timex']} realised TIMEX3 (+{total['timex_empty']} anchoring), "
          f"{total['signal']} signals")
    print(f"                {total['tlink_asserted']} asserted TLINK, "
          f"{total['tlink_derived']} derived, {total['slink']} SLINK")
    print(f"                {total['eligible']} timeline-eligible, "
          f"{total['subordinated']} subordinated")
    for book, rep in res.reports.items():
        print(f"  conformance   {rep.summary()}")
    print(f"\nInduction       {st['units']} event units -> "
          f"{st['clusters']} candidate canonical events "
          f"({st['contested_clusters']} contested)")
    print(f"                graph {st['graph_nodes']} nodes, "
          f"{st['graph_edges']} edges")
    print(f"                {st['conflicts']} conflicts reported")
    cons = res.consolidation
    kind = ("abstractive" if getattr(cons, "backbone", "") in
            ("bart", "pegasus", "primera", "instruct", "ollama")
            else "not abstractive")
    print(f"\nConsolidation   backbone '{cons.backbone}' ({kind})")
    if res.backbone_note:
        print(f"  NOTE          {res.backbone_note}")
    print(f"                {len(cons.paragraphs)} paragraphs, "
          f"{cons.length} characters")
    print(f"\nWrote {out}/")
    print(f"  consolidated.txt, consolidated_with_markers.txt")
    print(f"  curation.json          per event: every source account + the fusion")
    print(f"  annotation/<book>.tml, .tokens.xml, .json")
    print(f"  stage3/timeline.json")
    print("\nNo chronology and no reference consolidation were read. "
          "Run run_experiments.py to evaluate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
