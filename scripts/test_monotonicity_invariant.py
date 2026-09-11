#!/usr/bin/env python3
"""
Addendum 16, Task 3 -- lock the structural claim of thesis Sections 6.4.3,
10.5.2 and 11.2: the adopted (progressive-profile) configuration's feedback
arc set never fires, because the alignment is monotone by construction.

If someone later changes the alignment and monotonicity is lost, this
fails before the thesis's published claim does. Fast: extractive backbone,
no GNN -- `removed_arcs` and `conflicts` are Stage 3 figures, unaffected by
backbone or graph-propagation choice.

    python scripts/test_monotonicity_invariant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline
from tavern.config import TavernConfig

EXPECTED_REMOVED_ARCS = 0
EXPECTED_CONFLICTS = 96


def main() -> int:
    cfg = TavernConfig(tag="test_monotonicity", backbone="extractive")
    res = pipeline.run(cfg, with_gnn=False, write=False, verify=True)
    stats = res.stage3.stats()
    removed_arcs = stats["removed_arcs"]
    conflicts = stats["conflicts"]

    print(f"removed_arcs = {removed_arcs}  (expected {EXPECTED_REMOVED_ARCS})")
    print(f"conflicts    = {conflicts}  (expected {EXPECTED_CONFLICTS})")

    ok = True
    if removed_arcs != EXPECTED_REMOVED_ARCS:
        print("FAIL: removed_arcs != 0 -- the profile alignment is no "
             "longer monotone, or the tournament changed. Sections 6.4.3, "
             "10.5.2 and 11.2's published claim no longer holds as stated.")
        ok = False
    if conflicts != EXPECTED_CONFLICTS:
        print("FAIL: conflicts changed from the ancoragem run's own count "
             "-- re-verify before assuming this is fine.")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
