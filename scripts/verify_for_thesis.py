#!/usr/bin/env python3
"""
Fix one definition of content-word coverage and one canonical text digest,
so that Sections 8.8/10.11's three percentages and denominator, and the
human-evaluation booklets built from `outputs/ancoragem/curation.json`, each
have a single, reproducible, citable procedure behind them -- which, before
this script, none of them did (sixteen tried definitions gave a coverage
range, not a point, and no digest of the booklet-generating artefact had
ever been recorded).

Content-word coverage (fixed, not tuned):
  - lowercase the text, take maximal runs of [a-z] as tokens (re.findall)
  - a token is a content word unless it is in scikit-learn's
    ENGLISH_STOP_WORDS (318 entries) -- no custom stopword list
  - no lemmatization, no frequency cutoff, no length cutoff: each is one
    more choice to justify, and across sixteen tested definitions none
    changed the reference < extractive < abstractive ordering
  - coverage(output, sources) = |types(output) & types(sources)| / |types(sources)|

Canonical per-event text digest (fixed, not tuned):
  - one line per event: "<marker>\\t<consolidated text, whitespace collapsed>"
  - lines sorted, joined with "\\n", encoded UTF-8, SHA-256
  - moves only if the text moves -- not with key ordering, float formatting,
    or file metadata, which is why a whole-file digest was the wrong test

    python scripts/verify_for_thesis.py \\
        --abstractive outputs/ancoragem/curation.json \\
        --extractive  outputs/extractive/curation.json \\
        --sources     data/combined_documents.txt \\
        --reference   data/Golden_Sample.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z]+")


def content_types(text: str) -> set:
    return {t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS}


def coverage(output_types: set, source_types: set) -> float:
    return len(output_types & source_types) / len(source_types)


def load_events(path: str) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))["events"]


def canonical_digest(events: list) -> tuple:
    lines = []
    total_len = 0
    for ev in events:
        text = re.sub(r"\s+", " ", ev["consolidated"]).strip()
        total_len += len(text)
        lines.append(f"{ev['marker']}\t{text}")
    lines.sort()
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), len(events), total_len


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstractive", required=True)
    ap.add_argument("--extractive", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    sources_text = Path(args.sources).read_text(encoding="utf-8", errors="replace")
    reference_text = Path(args.reference).read_text(encoding="utf-8", errors="replace")
    abst_events = load_events(args.abstractive)
    extr_events = load_events(args.extractive)
    abst_text = " ".join(ev["consolidated"] for ev in abst_events)
    extr_text = " ".join(ev["consolidated"] for ev in extr_events)

    src_types = content_types(sources_text)
    ref_types = content_types(reference_text)
    abst_types = content_types(abst_text)
    extr_types = content_types(extr_text)

    print("=== content-word coverage (scikit-learn ENGLISH_STOP_WORDS, "
          "[a-z]+ types, no lemmatization/frequency/length cutoff) ===")
    print(f"source content-word types (denominator) = {len(src_types)}")
    print(f"reference   coverage = {coverage(ref_types, src_types):.4f}")
    print(f"abstractive coverage = {coverage(abst_types, src_types):.4f}")
    print(f"extractive  coverage = {coverage(extr_types, src_types):.4f}")
    print(f"ordering holds (abstractive > reference > extractive): "
          f"{coverage(abst_types, src_types) > coverage(ref_types, src_types) > coverage(extr_types, src_types)}")

    print("\n=== canonical per-event text digest ===")
    for label, events in (("abstractive", abst_events), ("extractive", extr_events)):
        digest, n, total_len = canonical_digest(events)
        print(f"{label:12s} {digest}")
        print(f"             {n} events, {total_len} chars (sum of "
              f"whitespace-collapsed per-event text)")

    sibling = Path(args.abstractive).parent / "consolidated.txt"
    if sibling.exists():
        serialized_len = len(sibling.read_text(encoding="utf-8", errors="replace"))
        print(f"\n{sibling} length (cross-check against the thesis's "
              f"128,837): {serialized_len}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
