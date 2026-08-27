#!/usr/bin/env python3
"""
Regression guard: fail if too many generated events show the glued-word
decoding artefact (tavern/stage6_evaluation/text_quality.py).

    python scripts/check_text_quality.py outputs/ollama_p11/curation.json
    python scripts/check_text_quality.py outputs/ollama_p11/curation.json \
        --max-fraction 0.01

Exit code 0 and "PASS" if the corrupted-event fraction is at or below the
threshold, exit code 1 and "FAIL" with examples otherwise. Measured with
repeat_penalty=1.5 (the bug this guards against): 3/249 = 1.2%.
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
    ap.add_argument("--max-fraction", type=float, default=0.01)
    args = ap.parse_args()

    data = json.loads(args.curation_json.read_text(encoding="utf-8"))
    texts = [e["consolidated"] for e in data["events"] if e.get("consolidated")]
    if not texts:
        print(f"FAIL: no consolidated text found in {args.curation_json}")
        return 1

    try:
        report = text_quality.assert_below_threshold(
            texts, max_fraction=args.max_fraction)
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: {report.corrupted}/{report.total} events "
          f"({report.fraction:.1%}) glued, at or below "
          f"{args.max_fraction:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
