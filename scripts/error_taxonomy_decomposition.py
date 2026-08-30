#!/usr/bin/env python3
"""
Addendum 7 (user's follow-up on Task 1) -- decompose the error taxonomy's
under-merged/over-merged counts by whether they involve a curated event
touched by verse-key collision (two curated events citing the same whole
verse via a half-verse split Aschmann makes but the corpus's verse-key
granularity cannot -- see cluster_purity.py's verse_to_event docstring and
local_timeline.segment, which opens boundaries only between whole verses).

Separates forced structural error (the reference itself cites an
unreachable distinction) from genuine alignment error, the same role
MAX_EPISODE_VERSES's sweep played for under-merging in Addendum 2.

    python scripts/error_taxonomy_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline
from tavern.config import OUTPUT_DIR, TavernConfig
from tavern.stage6_evaluation import chronology as chrono_mod
from tavern.stage6_evaluation import timeline_eval


def collision_events(ch):
    key_to_events = {}
    for e in ch.events:
        for book, keys in e.verse_keys.items():
            for k in keys:
                key_to_events.setdefault(k, set()).add(e.event_id)
    touched = set()
    for k, eids in key_to_events.items():
        if len(eids) > 1:
            touched |= eids
    return touched, key_to_events


def main() -> int:
    cfg = TavernConfig(tag="error_decomposition", backbone="extractive")
    res = pipeline.run(cfg, with_gnn=True, write=False, verify=True)
    ch = chrono_mod.load(res.corpus)
    units = res.units
    clustering = res.stage3.clustering

    touched, key_to_events = collision_events(ch)
    print(f"events touched by verse-key collision: {len(touched)} / {len(ch.events)}")

    # -- under-merged: one curated event's verses spread over >=3 clusters --
    under = []
    for e in ch.events:
        if not e.all_keys:
            continue
        cids = set()
        for cl in clustering.clusters:
            for m in cl.members:
                if set(units[m].verse_keys) & set(e.all_keys):
                    cids.add(cl.cluster_id)
        if len(cids) >= 3:
            under.append(e.event_id)

    under_touched = [eid for eid in under if eid in touched]
    print(f"\nunder-merged events: {len(under)}")
    print(f"  of which touched by collision: {len(under_touched)} "
         f"({len(under_touched)/len(under):.1%})" if under else "  (none)")
    print(f"  touched ids: {sorted(under_touched)}")

    # -- over-merged: one cluster covers >=2 curated events at >=50% overlap --
    over = []
    for cl in clustering.clusters:
        keys = set()
        for m in cl.members:
            keys |= set(units[m].verse_keys)
        hits = [e.event_id for e in ch.events
               if e.all_keys and len(set(e.all_keys) & keys)
               >= 0.5 * len(set(e.all_keys))]
        if len(hits) >= 2:
            over.append((cl.cluster_id, hits))

    over_touched = [(cid, hits) for cid, hits in over
                   if any(h in touched for h in hits)]
    print(f"\nover-merged clusters: {len(over)}")
    print(f"  of which involve a touched event: {len(over_touched)} "
         f"({len(over_touched)/len(over):.1%})" if over else "  (none)")
    for cid, hits in over_touched:
        print(f"    {cid}: events {hits}, touched: "
             f"{[h for h in hits if h in touched]}")

    out = {
        "events_touched_by_collision": sorted(touched),
        "under_merged": {"total": len(under), "touched": sorted(under_touched)},
        "over_merged": {"total": len(over),
                        "touched": [{"cluster": cid, "events": hits}
                                   for cid, hits in over_touched]},
    }
    out_dir = OUTPUT_DIR / "error_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decomposition.json").write_text(json.dumps(out, indent=1),
                                                encoding="utf-8")
    print(f"\nwrote {out_dir / 'decomposition.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
