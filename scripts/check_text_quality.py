#!/usr/bin/env python3
"""
Regression guard over an abstractive run's `curation.json`, on two axes
(tavern/stage6_evaluation/text_quality.py).

    python scripts/check_text_quality.py outputs/ollama_p11/curation.json
    python scripts/check_text_quality.py outputs/ollama_p11/curation.json \
        --max-fraction 0.01 --max-infidelity 0.02

1. Glued words: the decoding artefact of a mistuned `repeat_penalty`.
   Measured with repeat_penalty=1.5, the bug this guards against: 3/249 =
   1.2%.
2. Fusion fidelity, on three axes: events that lose a detail the accounts
   carry, that invent one they do not, or that state the same wording twice.
   Measured against the pre-fix `ancoragem` artifacts, which had no repair
   path, including a fabricated aviation accident on Matthew 24:20. That
   run's `curation.json` does not pass this check, and it is meant to be run
   against it -- that failure is the "before" the fix is measured from.

Exit code 0 and "PASS" only if both are at or below their thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern.stage6_evaluation import text_quality


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("curation_json", type=Path,
                    help="outputs/<tag>/curation.json from an abstractive run")
    ap.add_argument("--max-fraction", type=float, default=0.01,
                    help="glued-word events allowed")
    ap.add_argument("--max-infidelity", type=float, default=0.02,
                    help="events allowed to lose or invent detail")
    args = ap.parse_args()

    data = json.loads(args.curation_json.read_text(encoding="utf-8"))
    events = data["events"]
    texts = [e["consolidated"] for e in events if e.get("consolidated")]
    if not texts:
        print(f"FAIL: no consolidated text found in {args.curation_json}")
        return 1

    failed = False
    try:
        report = text_quality.assert_below_threshold(
            texts, max_fraction=args.max_fraction)
        print(f"PASS glued    : {report.corrupted}/{report.total} events "
              f"({report.fraction:.1%}), at or below "
              f"{args.max_fraction:.1%}")
    except AssertionError as exc:
        print(f"FAIL glued    : {exc}")
        failed = True

    fid = text_quality.scan_fidelity(events)
    print(f"     fidelity : median detail recall {fid.median_recall:.3f}, "
          f"novel content {fid.median_novel:.3f}, "
          f"internal repetition {fid.median_redundancy:.3f}")
    print(f"                {fid.dropped} drop detail, {fid.invented} invent, "
          f"{fid.repeated} repeat themselves")
    if fid.by_fusion:
        print("     fusion   : "
              + ", ".join(f"{k}={v}" for k, v in sorted(fid.by_fusion.items())))
    fb = [e for e in events if e.get("fusion") == "union"]
    if fb:
        fbr = text_quality.scan_fidelity(fb)
        print(f"     fallback : {len(fb)} events the backbone gave up on, "
              f"{fbr.failing} of them still repeating themselves "
              f"(union does not deduplicate; not counted below)")
    try:
        rep = text_quality.assert_fidelity_below(
            events, max_fraction=args.max_infidelity)
        print(f"PASS fidelity : {rep.failing}/{rep.total} fused events "
              f"({rep.fraction:.1%}) fail the premise, at or below "
              f"{args.max_infidelity:.1%}")
    except AssertionError as exc:
        print(f"FAIL fidelity : {exc}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
