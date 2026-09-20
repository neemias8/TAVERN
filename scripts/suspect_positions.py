#!/usr/bin/env python3
"""
Emit the induced positions involved in an ordering inversion, as the
`--suspect` input of `scripts/make_human_eval.py`.

    python scripts/suspect_positions.py --out human_eval/suspect.json

Section 9.3.5's Part B pre-registers 30 adjacent pairs drawn half from the
monotone subsequence and half from suspected transpositions, precisely so
that the raw proportion correct can be corrected into a stratum-weighted
estimate -- the only figure comparable with the pairwise agreement against
the chronology. `make_human_eval.py build` takes that suspect list through
`--suspect` and warns when it is missing, but nothing in the repository
produced it: `error_analysis.analyse` counts the transpositions and keeps one
example string, discarding which clusters they were. Without this script the
booklet's Part B comes out `{"monotone": 30}` and the weighted estimate
cannot be computed at all, so the protocol silently degrades to its own
pessimistic bound.

This reads the chronology, so it is a Stage 6 script and must never be
imported from stages 1-5 -- the same reason `make_human_eval.py` is one.

The definition is `error_analysis`'s own, kept deliberately in step with it:
take the curated events that matched a cluster, order them by the
chronology's rank, and flag every consecutive pair the induced order puts
the other way round. Both members of the inverted pair are suspect -- the
instrument cannot say which of the two moved, and the booklet only needs to
know that the region is contested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline                                       # noqa: E402
from tavern.config import TavernConfig                            # noqa: E402
from tavern.stage6_evaluation import chronology as chrono_mod     # noqa: E402
from tavern.stage6_evaluation import timeline_eval                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ancoragem")
    ap.add_argument("--backbone", default="extractive",
                    help="the fusion backbone is irrelevant here -- only the "
                         "induced order is read -- so the default avoids "
                         "paying for a generation pass")
    ap.add_argument("--out", type=Path, default=Path("human_eval/suspect.json"))
    args = ap.parse_args()

    res = pipeline.run(TavernConfig(tag=args.tag, backbone=args.backbone),
                       with_gnn=True, write=False, verify=False)
    ch = chrono_mod.load(res.corpus)
    induced = res.stage3.induced
    matching, _q = timeline_eval.match_clusters_to_events(
        ch, res.stage3.clustering, res.units)

    rank = ch.rank()
    pairs = sorted((rank[eid], induced.rank.get(cid), eid, cid)
                   for eid, cid in matching.items()
                   if induced.rank.get(cid) is not None)

    day_of = {e.event_id: e.day for e in ch.events}
    suspect, detail = set(), []
    for (ri, hi, ei, ci), (rj, hj, ej, cj) in zip(pairs, pairs[1:]):
        if hi <= hj:
            continue
        # `consolidate` numbers events from 1 over `induced.order`; `rank` is
        # the 0-based index into the same list, so the booklet's `position`
        # is rank + 1. Off by one here and the strata are silently wrong.
        pi, pj = hi + 1, hj + 1
        suspect.update((pi, pj))
        detail.append({
            "curated": [ei, ej], "clusters": [ci, cj], "positions": [pi, pj],
            "same_day": day_of.get(ei) == day_of.get(ej),
            "kind": ("adjacent transposition"
                     if day_of.get(ei) == day_of.get(ej)
                     else "displaced across a day boundary"),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sorted(suspect)), encoding="utf-8")
    side = args.out.with_name(args.out.stem + "_detail.json")
    side.write_text(json.dumps(detail, indent=1), encoding="utf-8")

    adj = sum(1 for d in detail if d["same_day"])
    print(f"{len(detail)} inversions ({adj} same-day transpositions, "
          f"{len(detail) - adj} across a day boundary) over "
          f"{len(pairs)} matched events")
    print(f"{len(suspect)} distinct induced positions -> {args.out}")
    print(f"per-inversion detail -> {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
