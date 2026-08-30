#!/usr/bin/env python3
"""
Stage 6 - alignment precision AND recall: cluster purity with B-cubed.

    python scripts/cluster_purity.py outputs/ollama/curation.json
    python scripts/cluster_purity.py outputs/*/curation.json --json purity.json

Why this exists. The error taxonomy counts a cluster as over-merged when it
joins two curated events, and it reported 25 such clusters out of 159
multi-witness clusters. That count turned out to understate the problem badly,
because it asks a question about whole clusters rather than about each witness
in them. The question that matters for the downstream text is finer: does every
witness a cluster holds actually belong to the same curated event? A cluster
whose Mark span belongs to event 8 and whose Luke span belongs to event 6 emits
one document's account of a boundary that falls in the wrong place, and the
consolidated text becomes a shingled, offset version of the reference -- every
unigram present, the subsequence broken at every seam. That is the measured
signature: R-1 0.93 against R-L 0.59.

Three figures are reported and they answer different questions.

  purity      the fraction of multi-witness clusters in which every witness maps
              to the same curated event. This is alignment precision.
  amplitude   max(event id) - min(event id) within a cluster. Separates local
              misregistration, where a cluster straddles adjacent event
              boundaries, from a spurious merge across the week.
  day mixing  the fraction of clusters whose witnesses fall on different days of
              the Passion Week. A cluster that mixes days cannot be a single
              event under any reading.

Read purity ALONGSIDE the cluster count, never instead of it. Loosening the
merge bound moves the cluster count towards the 169 curated events while
merging more material into each cluster; if purity falls at the same time, the
change is making the system worse and the count better. That pairing is the
whole point of the measurement.

STRUCTURAL CEILING: purity, B-cubed and SAME_EVENT precision cap out at
roughly 89.5%, not 100%, and this is not a bug -- verified against a perfect
(oracle) clustering, which still lands there (Addendum 7, Task 1). The cause
is a genuine granularity mismatch between the system and the reference:
`local_timeline.segment`'s `for v in verses` opens a unit boundary only
between whole verses -- TAVERN's atom is the verse -- while Aschmann's
harmonisation occasionally individuates at the HALF-verse level ("14:66-68a"
for one event, "14:68b" for the next). 26 verse keys are claimed by two
curated events this way, touching 40 of 169 events; 10 of those 40 have no
verse exclusive to them at all (events 47, 93, 99, 119, 132, 133, 155, 156,
157, 163) and so cannot be individuated as distinct objects under ANY
clustering, however correct -- 11 counting event 53, which has no verse
citation in any book at all. Do not read a purity/B-cubed number against
100%; read it against ~89.5%, and report both.

A finer distinction, confirmed by an independent recomputation and worth
keeping separate from the 10 above: of those 10, THREE (47, 93, 157) never
win `v2e.setdefault` for even one of their own cited verses -- every verse
they cite is also cited by an earlier-processed competing event, so they
never appear as a value in `verse_to_event`'s map at all. Add event 53 (no
citation anywhere) and **4 of 168 citable events are invisible to any
verse-keyed instrument, not just under-individuated by one** -- they can
never be a `gold_clusters` entry in `bcubed`, nor a `modal` winner here.
168 - 4 = 164 is not printed anywhere as its own figure, but it is why
`bcubed`'s `gold_clusters` reads 165 (168 total minus the 3 that have
citations but never win one), not 168 or 169. Not fixed here or upstream:
doing so would mean a half-verse key threaded through
Corpus/ReferenceParser/Chronology, used everywhere, for a gain in reporting
accuracy rather than in the system itself -- see DATA_PROVENANCE.md.

This script reads the chronology and therefore belongs to Stage 6. It is a
standalone SCRIPT (outside the `tavern` package) so that no import path
exists from stages 1-5 into it -- but it still reuses the package's own
Stage 1 corpus parser and Stage 6 chronology loader rather than
re-implementing verse-reference parsing a second time (see the fix note on
`verse_to_event` below). Importing `tavern.stage1_preprocessing` and
`tavern.stage6_evaluation` here is a forward reference, the same direction
`run_experiments.py` already takes; `assert_no_chronology_import` inspects
the call stack for stage1-5 frames, and this script has none.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern.stage1_preprocessing.corpus import Corpus
from tavern.stage6_evaluation import chronology as chrono_mod

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ("matthew", "mark", "luke", "john")


def verse_to_event(data: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map `book:chapter:verse` to the curated event that cites it.

    Built from `Chronology.load`, the same parser the rest of the framework
    uses (`stage1_preprocessing.corpus.ReferenceParser` /`VerseSplitter`) --
    not a second, independent regex. The earlier version had its own
    `(\\d+):(\\d+)(-(\\d+)?:(\\d+))?` pattern, which does not understand
    half-verse citations ("14:68a", "14:68b"). Two adjacent curated events
    routinely split a verse that way (event 92 "Peter denies Jesus the first
    time" cites mark 14:66-68a; event 93 "A rooster crows (the first time)"
    cites mark 14:68b); the old regex resolved both to the whole verse
    mark:14:68, and `setdefault` gave it to whichever event was processed
    first, leaving the other with an incomplete or empty span. Confirmed by
    the Addendum 7 oracle round-trip: purity and B-cubed came back at 86.3%
    and 0.832 F1 against a perfect input instead of 100%/1.0, traced to 21 of
    169 events (6 of them left with zero mapped verses) affected by this. The
    official parser has no such gap, since half-verse splitting is exactly
    what it was built for (thesis Section 5.2 / the IJCNN-ported
    VerseSplitter).
    """
    corpus = Corpus(data_dir=data)
    ch = chrono_mod.load(corpus, data_dir=data)
    v2e: Dict[str, int] = {}
    e2day: Dict[int, str] = {}
    for e in ch.events:
        e2day[e.event_id] = e.day
        for book, keys in e.verse_keys.items():
            for b, c, v in keys:
                v2e.setdefault(f"{b}:{c}:{v}", e.event_id)
    return v2e, e2day


def bcubed(events: List[dict], v2e: Dict[str, int]) -> dict:
    """B-cubed precision, recall and F1 over the verse-level clustering.

    Purity, above, is a precision measure with no recall term, and that makes it
    gameable in one specific direction: an aligner made conservative enough to
    stop matching across documents purifies whatever little remains matched, and
    purity rises while the system gets worse. Any change that raises purity must
    therefore be read against recall, and B-cubed is the standard pairing --
    computed against the curated events as the gold clustering, per verse.

    Recall is the binding side on this corpus, so the instrument matters: the
    baseline scores P 0.62 / R 0.41, and a change that trades recall for
    precision moves along the axis that is already strong.
    """
    sysc: Dict[str, str] = {}
    for r in events:
        cid = str(r.get("marker") or r.get("cluster"))
        for s in (r.get("sources") or []):
            if isinstance(s, dict):
                for v in (s.get("verses") or []):
                    if v in v2e:
                        sysc[v] = cid
    common = [v for v in sysc if v in v2e]
    if not common:
        return {}
    gold: Dict[str, set] = {}
    syss: Dict[str, set] = {}
    G: Dict[int, Set[str]] = {}
    S: Dict[str, Set[str]] = {}
    for v in common:
        G.setdefault(v2e[v], set()).add(v)
        S.setdefault(sysc[v], set()).add(v)
    p = r = 0.0
    for v in common:
        g, s = G[v2e[v]], S[sysc[v]]
        inter = len(g & s)
        p += inter / len(s)
        r += inter / len(g)
    n = len(common)
    p /= n
    r /= n
    return {
        "verses": n, "gold_clusters": len(G), "system_clusters": len(S),
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
    }


def load_events(path: Path) -> List[dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d
    for f in ("events", "records"):
        if isinstance(d.get(f), list):
            return d[f]
    raise SystemExit(f"{path}: cannot find the event list")


def analyse(path: Path, v2e: Dict[str, int],
            e2day: Dict[int, str]) -> Optional[dict]:
    events = load_events(path)
    pure = impure = skipped = daymix = 0
    amp = Counter()
    touched = Counter()
    worst: List[dict] = []

    for rec in events:
        srcs = [s for s in (rec.get("sources") or []) if isinstance(s, dict)]
        if len(srcs) < 2:
            continue
        per: List[Tuple[str, str, Set[int]]] = []
        for s in srcs:
            ids = {v2e[v] for v in (s.get("verses") or []) if v in v2e}
            per.append((s.get("gospel", "?"), s.get("ref", ""), ids))
        allids: Set[int] = set().union(*[p[2] for p in per]) if per else set()
        if not allids:
            skipped += 1
            continue

        counts: Counter = Counter()
        for _g, _r, ids in per:
            counts.update(ids)
        modal = counts.most_common(1)[0][0]
        intruders = [(g, r, sorted(ids)) for g, r, ids in per if modal not in ids]

        spread = max(allids) - min(allids)
        amp[spread] += 1
        touched[len(allids)] += 1
        days = {e2day.get(i, "") for i in allids if e2day.get(i)}
        mixed = len(days) > 1
        daymix += mixed

        if intruders:
            impure += 1
            if mixed and spread >= 8:
                worst.append({
                    "cluster": rec.get("marker") or rec.get("cluster"),
                    "events": sorted(allids), "days": sorted(days),
                    "dominant": modal,
                    "witnesses": [{"gospel": g, "ref": r, "events": sorted(ids)}
                                  for g, r, ids in per],
                })
        else:
            pure += 1

    n = pure + impure
    if not n:
        return None
    cum, dist = 0, {}
    for k in sorted(amp):
        cum += amp[k]
        dist[str(k)] = {"clusters": amp[k], "cumulative_share": round(cum / n, 4)}
    worst.sort(key=lambda w: -(max(w["events"]) - min(w["events"])))
    return {
        "source": str(path),
        "bcubed": bcubed(events, v2e),
        "multi_witness_clusters": n,
        "skipped_unmappable": skipped,
        "pure": pure,
        "impure": impure,
        "purity": round(pure / n, 4),
        "day_mixing": round(daymix / n, 4),
        "events_touched_per_cluster": {str(k): v for k, v in sorted(touched.items())},
        "amplitude": dist,
        "worst_merges": worst[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("curation", nargs="+", help="one or more curation.json paths")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--json", default="", help="write the full report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    v2e, e2day = verse_to_event(Path(args.data))
    if not args.quiet:
        print(f"chronology maps {len(v2e)} verses to {len(e2day)} curated events\n")

    reports = []
    for p in args.curation:
        r = analyse(Path(p), v2e, e2day)
        if r is None:
            print(f"{p}: no multi-witness clusters"); continue
        reports.append(r)
        print("=" * 72)
        print(p)
        print(f"  multi-witness clusters   {r['multi_witness_clusters']}")
        print(f"  PURITY                   {r['purity']:.1%}   "
              f"({r['pure']} pure / {r['impure']} impure)")
        print(f"  mixes different days     {r['day_mixing']:.1%}")
        print(f"  events touched/cluster   {r['events_touched_per_cluster']}")
        b = r.get("bcubed") or {}
        if b:
            print(f"  B-cubed                  P {b['precision']:.4f}  "
                  f"R {b['recall']:.4f}  F1 {b['f1']:.4f}   "
                  f"({b['system_clusters']} system / {b['gold_clusters']} gold)")
            print("       ^ read purity against B-cubed recall: a conservative "
                  "aligner raises one and lowers the other")
        if not args.quiet:
            print("  amplitude (max-min curated event id):")
            for k, v in r["amplitude"].items():
                print(f"     {k:>4s}  {v['clusters']:3d}   cumulative "
                      f"{v['cumulative_share']:.1%}")
            if r["worst_merges"]:
                print("  worst merges (different days, amplitude >= 8):")
                for w in r["worst_merges"][:5]:
                    print(f"     {w['cluster']}  events {w['events']}  "
                          f"days {w['days']}")

    if args.json and reports:
        Path(args.json).write_text(json.dumps(reports, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
