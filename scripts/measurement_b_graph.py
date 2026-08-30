#!/usr/bin/env python3
"""
Addendum 6, Measurement B -- the induced graph against a reference graph
built from the curated chronology, on the same node set. No R-L, no
gemma3:4b, no reference-that-is-a-selection: this validates the substituted
artefact (the cross-document event graph) as an artefact, not through
downstream generation.

  SAME_EVENT   vs curated coreference: two units are truly coreferent if
               their verses map to the same curated event AND they come from
               different Gospels. Precision/recall/F1 over all cross-book
               unit pairs with a resolvable curated event -- this is the
               quality of the LINK, measured directly, independent of pureza
               (which is cluster-level; this is edge-level).
  INTER_BEFORE vs curated order: a before b if the curated event_id of a is
               smaller. TAVERN's INTER_BEFORE edges connect only temporally
               ADJACENT clusters in the induced order, not a full pairwise
               closure -- so recall against the full cross-book "true before"
               universe is structurally bounded well below 1 by design, not
               by defect. Reported as specified anyway; the mechanism is
               explained, not hidden.
  INTRA_BEFORE is narrative order by construction (same book, same document
               sequence) -- a CONTROL, not a test. Low precision here would
               mean the verse-to-event mapping is wrong, not that TAVERN is.

TAEG (tavern/baselines/taeg_algorithm1) builds its own graph fresh from the
chronology on every call rather than persisting one to disk -- there is no
separate on-disk TAEG artefact in this repository to check against; its
edge semantics (SAME_EVENT across books sharing a curated event, BEFORE
between consecutive curated events within a book) are exactly what the
reference graph below already encodes.

This is Stage 6 diagnostic code (reads the chronology) and must not be
importable from stages 1-5 -- standalone script, same reasoning as
cluster_purity.py and measurement_a_recallk.py.

    python scripts/measurement_b_graph.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_purity
from tavern import pipeline
from tavern.config import DATA_DIR, OUTPUT_DIR, TavernConfig


def unit_event_id(unit, v2e):
    """The single curated event a unit's verses resolve to, or None if zero
    or more than one (ambiguous spans are excluded, not guessed)."""
    ids = {v2e[f"{b}:{c}:{v}"] for b, c, v in unit.verse_keys
          if f"{b}:{c}:{v}" in v2e}
    return next(iter(ids)) if len(ids) == 1 else None


def prf(tp, pred_n, true_n):
    p = tp / pred_n if pred_n else None
    r = tp / true_n if true_n else None
    f1 = (2 * p * r / (p + r)) if (p and r and (p + r)) else None
    return {"tp": tp, "predicted": pred_n, "true": true_n,
           "precision": round(p, 4) if p is not None else None,
           "recall": round(r, 4) if r is not None else None,
           "f1": round(f1, 4) if f1 is not None else None}


def main() -> int:
    cfg = TavernConfig(tag="measurement_b", backbone="extractive")
    res = pipeline.run(cfg, with_gnn=False, write=False, verify=True)
    eg = res.stage3.graph
    units = eg.node_units

    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    eid_of = {uid: unit_event_id(u, v2e) for uid, u in units.items()}
    resolved = {uid: eid for uid, eid in eid_of.items() if eid is not None}
    print(f"nodes: {len(units)}   resolved to a single curated event: "
          f"{len(resolved)}   unresolved/ambiguous: "
          f"{len(units) - len(resolved)}")

    ids_sorted = sorted({e for e in resolved.values()})
    monotone = ids_sorted == sorted(set(ids_sorted))
    print(f"curated event_id used directly as order key "
          f"({len(ids_sorted)} distinct ids, min={min(ids_sorted)}, "
          f"max={max(ids_sorted)})")

    nodes = list(resolved)
    book_of = {uid: units[uid].book for uid in nodes}

    # ---- SAME_EVENT ------------------------------------------------------
    same_event_pred = set()
    for a, b, d in eg.g.edges(data=True):
        if d["type"] != "SAME_EVENT":
            continue
        if a in resolved and b in resolved and book_of[a] != book_of[b]:
            same_event_pred.add(frozenset((a, b)))

    same_event_true = set()
    by_event = {}
    for uid in nodes:
        by_event.setdefault(resolved[uid], []).append(uid)
    for eid, us in by_event.items():
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                if book_of[us[i]] != book_of[us[j]]:
                    same_event_true.add(frozenset((us[i], us[j])))

    tp = len(same_event_pred & same_event_true)
    same_event_result = prf(tp, len(same_event_pred), len(same_event_true))
    print("SAME_EVENT vs curated coreference:", same_event_result)

    # A curated event is coarser than an EventUnit: two units that both map
    # to the SAME curated event have no before/after claim in the reference
    # at all (they're SAME_EVENT, not ordered) -- excluded from both the
    # predicted and the true side of INTER_BEFORE/INTRA_BEFORE, not counted
    # as a miss. Confirmed necessary empirically: 283 of 598 INTRA_BEFORE
    # edges were same-event pairs; scoring them as wrong dragged the control
    # down to 0.505 precision, and excluding them recovers 0.9587.

    # ---- INTER_BEFORE ------------------------------------------------------
    inter_pred = set()
    for a, b, d in eg.g.edges(data=True):
        if d["type"] != "INTER_BEFORE":
            continue
        if (a in resolved and b in resolved and book_of[a] != book_of[b]
                and resolved[a] != resolved[b]):
            inter_pred.add((a, b))

    inter_true = set()
    for a in nodes:
        for b in nodes:
            if (a != b and book_of[a] != book_of[b]
                    and resolved[a] != resolved[b] and resolved[a] < resolved[b]):
                inter_true.add((a, b))

    tp_i = len(inter_pred & inter_true)
    inter_result = prf(tp_i, len(inter_pred), len(inter_true))
    print("INTER_BEFORE vs curated order (full cross-book pair universe, "
          "same-event pairs excluded, TAVERN's edges are adjacent-cluster-"
          "only by design):", inter_result)

    # ---- INTRA_BEFORE (control) -------------------------------------------
    intra_pred = set()
    for a, b, d in eg.g.edges(data=True):
        if d["type"] != "INTRA_BEFORE":
            continue
        if (a in resolved and b in resolved and book_of[a] == book_of[b]
                and resolved[a] != resolved[b]):
            intra_pred.add((a, b))

    intra_true = set()
    for book in set(book_of.values()):
        book_nodes = [u for u in nodes if book_of[u] == book]
        for a in book_nodes:
            for b in book_nodes:
                if (a != b and resolved[a] != resolved[b]
                        and resolved[a] < resolved[b]):
                    intra_true.add((a, b))

    tp_ib = len(intra_pred & intra_true)
    intra_result = prf(tp_ib, len(intra_pred), len(intra_true))
    print("INTRA_BEFORE vs curated order (CONTROL -- should be near-perfect "
          "precision):", intra_result)

    out = {
        "nodes": len(units), "resolved": len(resolved),
        "same_event": same_event_result,
        "inter_before": inter_result,
        "intra_before_control": intra_result,
        "graph_edge_counts": eg.edge_counts(),
    }
    out_dir = OUTPUT_DIR / "measurement_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_vs_reference.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir / 'graph_vs_reference.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
