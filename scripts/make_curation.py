#!/usr/bin/env python3
"""
Turn `outputs/<tag>/curation.json` into artefacts a human expert can work with.

    python scripts/make_curation.py outputs/union/curation.json consolidations/

Writes three files:

  consolidated.txt   the narrative on its own, for reading straight through
  curation.md        one section per event: every source account beside the
                     consolidation derived from it, with the verse addresses
                     and a blank verdict block to fill in
  curation.csv       the same, one row per event, for a spreadsheet

The point of the per-event layout is that consolidation quality is not
judgeable from the finished narrative alone. An expert has to see what went in
to say whether what came out is faithful, complete and correctly placed --- and
those are three separate judgements, so the verdict block asks for them
separately.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BOOKS = {"matthew": "Matthew", "mark": "Mark", "luke": "Luke", "john": "John"}


def main(src: str, dest: str) -> int:
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    events = data["events"]
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)

    (out / "consolidated.txt").write_text(
        " ".join(e["consolidated"] for e in events) + "\n", encoding="utf-8")

    md = [
        "# TAVERN — consolidation for expert curation",
        "",
        f"- Fusion backbone: **`{data.get('backbone')}`**"
        + (f" — {data['fallback_note']}" if data.get("fallback_note") else ""),
        f"- Candidate canonical events: **{len(events)}**",
        f"- Order: induced by the framework. **No curated harmony was used.**",
        "",
        "`Day` is the day index the anchor scaffold assigns, derived from the",
        "temporal expressions in the text; it is not taken from any harmony.",
        "Day 0 is the Passover / Day of Preparation, so negative values are the",
        "days before it and +1, +2 are the Sabbath and the first day of the week.",
        "",
        "For each event: the source accounts first, then the consolidation, then",
        "a verdict block. Please edit the verdict lines in place.",
        "",
        "---",
        "",
    ]
    rows = []
    for e in events:
        day = e.get("scaffold_day")
        day_s = "—" if day is None else f"{int(day):+d}"
        md.append(f"## {e['marker']}  ·  position {e['position']}  ·  day {day_s}"
                  + ("  ·  **CONFLICT DETECTED**" if e["conflicted"] else ""))
        md.append("")
        for src_acc in e["sources"]:
            md.append(f"**{BOOKS.get(src_acc['gospel'], src_acc['gospel'])} "
                      f"{src_acc['ref']}**"
                      + ("  *(selected as most representative)*"
                         if src_acc["gospel"] == e["selected_source"] else ""))
            md.append("")
            md.append("> " + src_acc["text"].replace("\n", " "))
            md.append("")
        md.append("**Consolidation**")
        md.append("")
        md.append(e["consolidated"])
        md.append("")
        md.append("```")
        md.append("FAITHFUL?    yes / no  — does it assert anything the sources do not?")
        md.append("COMPLETE?    yes / no  — is any detail from any account missing?")
        md.append("PLACEMENT?   ok / early / late / wrong day")
        md.append("NOTES:")
        md.append("```")
        md.append("")
        md.append("---")
        md.append("")
        rows.append({
            "marker": e["marker"],
            "position": e["position"],
            "scaffold_day": "" if day is None else int(day),
            "conflict": "yes" if e["conflicted"] else "",
            "selected_source": e["selected_source"],
            "sources": "; ".join(
                f"{BOOKS.get(s['gospel'], s['gospel'])} {s['ref']}"
                for s in e["sources"]),
            "source_text": "  ||  ".join(
                f"[{BOOKS.get(s['gospel'], s['gospel'])}] {s['text']}"
                for s in e["sources"]),
            "consolidated": e["consolidated"],
            "faithful": "", "complete": "", "placement": "", "notes": "",
        })

    (out / "curation.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "curation.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(events)} events -> {out}/")
    for f in ("consolidated.txt", "curation.md", "curation.csv"):
        print(f"  {f}  ({(out / f).stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
