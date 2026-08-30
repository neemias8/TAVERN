#!/usr/bin/env python3
"""
Addendum 9, Task 5 -- the fraction of multi-witness clusters that have at
least one shared predicate/TIMEX3 term across two DIFFERENT books' member
units. This is exactly the condition behind `predicate_similarity`'s cosine
being nonzero (every IDF weight is strictly positive, so a shared term is
necessary and sufficient for cosine > 0): the addendum's Fact 1 measured 116
of 159 canonical clusters (73%) with NO such term, i.e. 43/159 (27%) WITH one.

This script could not reproduce that baseline. Checked two ways against the
canonical run's own `outputs/canonical/stage3/timeline.json` (the real
Cluster.members, not curation.json's post-merge verse spans, which pools
several verses per witness and would only inflate the fraction further):
with predicates and TIMEX3 labels together, and with predicates alone (they
give the same answer here -- no cross-book pair in this corpus shares a
TIMEX3 term without also sharing a predicate). Both give 139/159 = 87.4%,
not 27%. The likely reason: `preds` includes ordinary narrative verbs (SAY,
COME, GO) that are common enough that almost any two witnesses share ONE,
even about unrelated events -- raw set intersection is a much weaker test
than the IDF-weighted cosine `predicate_similarity` actually computes, where
a shared common verb contributes almost nothing. A stricter, unverified
guess at the addendum's actual method: restrict to the pairs `Clustering.
scores` actually recorded as the alignment's chosen match, rather than every
cross-book combination inside a merged, multi-verse cluster. Untested here --
`Clustering.scores` is not serialised to stage3/timeline.json, so checking it
means re-running the pipeline live. Reported as an open discrepancy, not
resolved by picking a number: see DATA_PROVENANCE.md / the Addendum 9 report.

    python scripts/predicate_evidence_fraction.py outputs/canonical outputs/ancoragem
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline
from tavern.config import TavernConfig
from tavern.stage3_anchoring_alignment.local_timeline import segment_corpus


def unit_terms(cfg: TavernConfig) -> Dict[str, Set[str]]:
    corpus, pericopes, segments, chains, structs, reports = pipeline.prepare(
        cfg, verify=True)
    timelines = segment_corpus(structs, corpus, pericopes, chains)
    units = {}
    for tl in timelines.values():
        for u in tl.units:
            units[u.unit_id] = u
    return ({uid: set(u.preds) | {f"T:{t}" for t in u.timex_preds}
            for uid, u in units.items()},
           {uid: u.book for uid, u in units.items()})


def check(path: Path, terms: Dict[str, Set[str]], book_of: Dict[str, str]):
    d = json.loads(path.read_text(encoding="utf-8"))
    total = with_evidence = 0
    for cl in d["clusters"]:
        if len(set(cl["books"])) < 2:
            continue
        total += 1
        members = cl["members"]
        found = any(
            book_of.get(members[i]) != book_of.get(members[j])
            and terms.get(members[i], set()) & terms.get(members[j], set())
            for i in range(len(members)) for j in range(i + 1, len(members)))
        with_evidence += bool(found)
    return with_evidence, total


def main() -> int:
    tags = sys.argv[1:] or ["outputs/canonical", "outputs/ancoragem"]
    cfg = TavernConfig(tag="predicate_evidence_check", backbone="extractive")
    terms, book_of = unit_terms(cfg)
    for tag in tags:
        p = Path(tag) / "stage3" / "timeline.json"
        if not p.exists():
            print(f"{tag}: no {p}, skipped")
            continue
        we, tot = check(p, terms, book_of)
        print(f"{tag:24s} {we}/{tot} = {we / tot:.4f}  "
             f"(no shared term: {tot - we}/{tot} = {(tot - we) / tot:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
