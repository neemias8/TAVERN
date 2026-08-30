#!/usr/bin/env python3
"""
Addendum 7, Task 1 -- calibrate the measurement instrument before trusting
any induced reading.

`TavernConfig` used to carry a `use_oracle_timeline` flag; it was declared
and never read anywhere in the codebase -- dead, not implemented. Running
the real pipeline with it set to True changed nothing, and it has since
been removed rather than wired up (Addendum 8, Task 3): doing that inside
pipeline.run would need Stage 1-5 code to read the chronology, which
config.assert_no_chronology_import() exists to refuse.

What this script does instead: build the ORACLE clustering/ordering
directly from the curated chronology -- one cluster per curated event,
containing every real Stage 1-3 EventUnit whose verses the chronology cites
for that event, ordered exactly as the chronology lists events -- and run
it through the SAME measurement functions (timeline_eval.evaluate,
cluster_purity, the SAME_EVENT graph comparison of measurement_b_graph.py)
that score the induced pipeline. This tests whether those instruments
return the perfect score they should when handed a perfect input, which is
the calibration question Addendum 7 asks, independent of whether
use_oracle_timeline was ever wired into pipeline.run.

Maximum achievable coverage is 168/168, not 169/169: event 53
("(Jesus rests?)") has empty refs in every book -- a purely inferential
entry in Aschmann's harmonisation with no verse citation at all, already
excluded from timeline_eval.evaluate's own denominator
(`total = len([e for e in chronology.events if e.all_keys])`). No system,
oracle included, can be evaluated against an event with no textual anchor.

    python scripts/oracle_roundtrip.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_purity
from tavern import pipeline
from tavern.config import DATA_DIR, GOLDEN_SAMPLE_FILE, OUTPUT_DIR, TavernConfig
from tavern.stage3_anchoring_alignment.event_coref import Cluster, Clustering
from tavern.stage3_anchoring_alignment.global_timeline import InducedTimeline
from tavern.stage5_generation import ExtractiveFuser, SelectionStrategy, consolidate
from tavern.stage6_evaluation import chronology as chrono_mod
from tavern.stage6_evaluation import content_metrics, redundancy, timeline_eval


def reference() -> str:
    return Path(DATA_DIR / GOLDEN_SAMPLE_FILE).read_text(
        encoding="utf-8", errors="replace")


def build_oracle(cfg: TavernConfig):
    """Run stages 1-3 for real (real annotation, real units), then replace
    the INDUCED clustering/ordering with one built straight from the
    chronology -- the oracle. No chronology import happens inside stages
    1-5: this function lives in a Stage 6 script, same guard as
    cluster_purity.py and measurement_b_graph.py."""
    corpus, pericopes, segments, chains, structs, reports = pipeline.prepare(
        cfg, verify=True)
    from tavern.stage3_anchoring_alignment.local_timeline import segment_corpus
    from tavern.stage3_anchoring_alignment import scaffold as scaffold_mod
    timelines = segment_corpus(structs, corpus, pericopes, chains)
    sc = scaffold_mod.build(structs, timelines, enabled=cfg.use_anchor_scaffold)

    units = {}
    for tl in timelines.values():
        for u in tl.units:
            units[u.unit_id] = u
    by_book = defaultdict(list)
    for u in units.values():
        by_book[u.book].append(u)

    ch = chrono_mod.load(corpus)

    clustering = Clustering()
    order = []
    for e in ch.events:
        if not e.all_keys:
            continue
        members = []
        for b, keys in e.verse_keys.items():
            kset = set(keys)
            members += [u.unit_id for u in by_book[b]
                       if set(u.verse_keys) & kset]
        if not members:
            continue
        cid = f"oracle_{e.event_id:03d}"
        clustering.clusters.append(Cluster(
            cluster_id=cid, members=members,
            books={units[m].book for m in members}))
        for m in members:
            clustering.cluster_of_unit[m] = cid
        order.append(cid)

    induced = InducedTimeline(order=order,
                              rank={cid: i for i, cid in enumerate(order)})
    return units, sc, clustering, induced, ch, corpus


def assert_calibrated(ev, tol: float = 1e-9) -> None:
    """Regression guard: the oracle round-trip's tau and pairwise accuracy
    must be exactly 1.0. This calibration was expensive to discover (Addendum
    7, Task 1 found cluster_purity's own verse parser silently mismapping 21
    of 169 curated events); it must not regress silently. Purity/B-cubed are
    NOT asserted here -- they have a known structural ceiling below 1.0 (see
    cluster_purity.py's module docstring), not a calibration target."""
    bad = {}
    if ev.tau is None or abs(ev.tau - 1.0) > tol:
        bad["tau"] = ev.tau
    if ev.pairwise_accuracy is None or abs(ev.pairwise_accuracy - 1.0) > tol:
        bad["pairwise_accuracy"] = ev.pairwise_accuracy
    if bad:
        raise AssertionError(
            f"oracle round-trip miscalibrated: {bad} (expected 1.0 for "
            f"both) -- timeline_eval.evaluate is no longer trustworthy for "
            f"any induced reading until this is fixed")


def main() -> int:
    cfg = TavernConfig(tag="oracle_roundtrip", backbone="extractive")
    units, sc, clustering, induced, ch, corpus = build_oracle(cfg)

    print(f"oracle clusters: {len(clustering.clusters)} "
          f"(curated events with >=1 resolvable unit span, "
          f"of {len(ch.events)} total)")

    ev = timeline_eval.evaluate(ch, clustering, induced, units)
    print("timeline_eval.evaluate:", ev.as_row())
    try:
        assert_calibrated(ev)
        print("CALIBRATION: PASS (tau == pairwise_accuracy == 1.0)")
    except AssertionError as exc:
        print(f"CALIBRATION: FAIL -- {exc}")
        return 1

    ref = reference()
    cons = consolidate(induced, clustering, units, fuser=ExtractiveFuser(),
                       strategy=SelectionStrategy("longest"), conflicts=[])
    sc_content = content_metrics.evaluate(cons.text, ref, with_meteor=False,
                                          with_bertscore=False)
    print("R-1/R-L (extractive, longest rule):", sc_content.as_row())

    out_dir = OUTPUT_DIR / "oracle_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)
    curation_path = out_dir / "curation.json"
    curation_path.write_text(json.dumps({"events": cons.records}, indent=1),
                             encoding="utf-8")

    v2e, e2day = cluster_purity.verse_to_event(DATA_DIR)
    purity = cluster_purity.analyse(curation_path, v2e, e2day)
    print("purity:", purity["purity"], "bcubed:", purity["bcubed"])

    pairs = [(rec.get("consolidated", ""),
             [s.get("text", "") for s in (rec.get("sources") or [])])
            for rec in cons.records]
    cov = redundancy.coverage_over_events(pairs)
    print("coverage_over_events:", cov)

    # SAME_EVENT vs reference, same construction as measurement_b_graph.py
    def unit_event_id(unit):
        ids = {v2e[f"{b}:{c}:{v}"] for b, c, v in unit.verse_keys
              if f"{b}:{c}:{v}" in v2e}
        return next(iter(ids)) if len(ids) == 1 else None

    eid_of = {uid: unit_event_id(u) for uid, u in units.items()}
    resolved = {uid: eid for uid, eid in eid_of.items() if eid is not None}
    book_of = {uid: units[uid].book for uid in units}

    same_event_pred = set()
    for cl in clustering.clusters:
        m = cl.members
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                if book_of[m[i]] != book_of[m[j]]:
                    same_event_pred.add(frozenset((m[i], m[j])))

    same_event_true = set()
    by_event = defaultdict(list)
    for uid, eid in resolved.items():
        by_event[eid].append(uid)
    for eid, us in by_event.items():
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                if book_of[us[i]] != book_of[us[j]]:
                    same_event_true.add(frozenset((us[i], us[j])))

    tp = len(same_event_pred & same_event_true)
    p = tp / len(same_event_pred) if same_event_pred else None
    r = tp / len(same_event_true) if same_event_true else None
    f1 = (2 * p * r / (p + r)) if (p and r) else None
    same_event_result = {"tp": tp, "predicted": len(same_event_pred),
                         "true": len(same_event_true),
                         "precision": round(p, 4) if p is not None else None,
                         "recall": round(r, 4) if r is not None else None,
                         "f1": round(f1, 4) if f1 is not None else None}
    print("SAME_EVENT vs reference (oracle clustering as predictor):",
          same_event_result)

    out = {
        "oracle_clusters": len(clustering.clusters),
        "timeline_eval": ev.as_row(),
        "content_metrics": sc_content.as_row(),
        "purity": purity["purity"], "bcubed": purity["bcubed"],
        "coverage_over_events": cov,
        "same_event_vs_reference": same_event_result,
    }
    (out_dir / "oracle_roundtrip_summary.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir / 'oracle_roundtrip_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
