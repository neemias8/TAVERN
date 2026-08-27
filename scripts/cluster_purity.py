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

This script reads the chronology and therefore belongs to Stage 6. It is a
standalone script so that no import path exists from stages 1-5.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ("matthew", "mark", "luke", "john")
XML = {"matthew": "EnglishNIVMatthew40_PW.xml", "mark": "EnglishNIVMark41_PW.xml",
       "luke": "EnglishNIVLuke42_PW.xml", "john": "EnglishNIVJohn43_PW.xml"}
REF = re.compile(r"(\d+):(\d+)(?:\s*-\s*(?:(\d+):)?(\d+))?")


def corpus_verses(data: Path) -> Dict[str, List[Tuple[int, int]]]:
    """The verses that actually exist, per book, in canonical order.

    Ranges in the chronology cross chapter boundaries ("21:33-22:5"), and
    expanding those needs the real chapter lengths rather than a guess.
    """
    out: Dict[str, List[Tuple[int, int]]] = {}
    for book, name in XML.items():
        path = data / name
        if not path.exists():
            raise SystemExit(f"missing corpus file: {path}")
        seq: List[Tuple[int, int]] = []
        chapter = None
        for el in ET.parse(path).getroot().iter():
            tag = el.tag.lower()
            num = el.get("number") or el.get("num") or el.get("id")
            if tag.endswith("chapter") and num and num.isdigit():
                chapter = int(num)
            elif tag.endswith("verse"):
                v = num if (num and num.isdigit()) else None
                if v is None:
                    continue
                ch = chapter
                if ch is None:
                    continue
                seq.append((ch, int(v)))
        out[book] = seq
    return out


def verse_to_event(data: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map `book:chapter:verse` to the curated event that cites it."""
    verses = corpus_verses(data)
    chron = data / "ChronologyOfTheFourGospels_PW.xml"
    if not chron.exists():
        raise SystemExit(f"missing chronology: {chron}")

    v2e: Dict[str, int] = {}
    e2day: Dict[int, str] = {}
    for ev in ET.parse(chron).getroot().iter("event"):
        try:
            eid = int(ev.get("id"))
        except (TypeError, ValueError):
            continue
        e2day[eid] = (ev.findtext("day") or "").strip()
        for book in BOOKS:
            el = ev.find(book)
            text = (el.text or "").strip() if el is not None else ""
            if not text:
                continue
            seq = verses.get(book, [])
            for part in re.split(r"[;,]", text):
                for m in REF.finditer(part):
                    c1, v1 = int(m.group(1)), int(m.group(2))
                    c2 = int(m.group(3)) if m.group(3) else c1
                    v2 = int(m.group(4)) if m.group(4) else v1
                    try:
                        i = seq.index((c1, v1))
                        j = seq.index((c2, v2))
                    except ValueError:
                        # a citation outside the corpus scope; recorded by the
                        # thesis as the Luke 13:34-35 case and skipped here
                        continue
                    for ch, vv in seq[i:j + 1]:
                        v2e.setdefault(f"{book}:{ch}:{vv}", eid)
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
                    "witnesses": [{"gospel": g, "ref": r, "events": ids}
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
