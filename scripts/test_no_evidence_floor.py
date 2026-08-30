#!/usr/bin/env python3
"""
Addendum 9, Task 4 -- the regression the diagnostic predicts should fail.

Two event units that share no predicate term, no non-ubiquitous entity, no
class evidence and no anchor evidence (positions further apart than
ANCHOR_BAND) must score BELOW MATCH_THRESHOLD. Before Addendum 9's fix this
fails: participant_similarity falls through to a bare Jaccard over
{JESUS, DISCIPLES} (== 1.0 for two units that only mention the ubiquitous
pair) and modal_compatibility returns 1.0 when neither side has a modal type,
so 0.25*1.0 + 0.10*1.0 = 0.35 >= 0.34 clears the threshold on zero linguistic
evidence.

    python scripts/test_no_evidence_floor.py

Exits 0 and prints PASS if the score is below threshold, 1 and FAIL with the
actual score otherwise. Run before Task 1-3's changes to confirm this FAILS
(the proof the defect exists), then after to confirm it PASSES.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern.stage3_anchoring_alignment import event_coref as ec
from tavern.stage3_anchoring_alignment.local_timeline import EventUnit
from tavern.stage3_anchoring_alignment.scaffold import Scaffold


def make_unit(uid: str, book: str, verse: int) -> EventUnit:
    return EventUnit(
        unit_id=uid, book=book, pericope_id=f"{book}-p1",
        verse_keys=[(book, 1, verse)],
        preds=["SAY" if uid == "a" else "GO"],  # disjoint predicates
        entities={"JESUS", "DISCIPLES"},         # only the ubiquitous pair
        classes={"OCCURRENCE"} if uid == "a" else {"STATE"},  # disjoint
        modal_types=set(),                       # no modal evidence
        tenses={"PAST"}, eligible_fraction=1.0,
    )


def main() -> int:
    a = make_unit("a", "matthew", 1)
    b = make_unit("b", "mark", 1)

    sc = Scaffold()
    sc.unit_position["a"] = 0.0
    sc.unit_position["b"] = 0.0 + ec.ANCHOR_BAND + 1.0  # forces anchor_compatibility == 0.0

    idf = ec.PredicateIDF([a, b])
    vectors = {a.unit_id: idf.vector(a), b.unit_id: idf.vector(b)}

    entity_idf = ec.EntityIDF([a, b]) if hasattr(ec, "EntityIDF") else None
    if entity_idf is not None:
        s = ec.score(a, b, sc, vectors, entity_idf)
        part = ec.participant_similarity(a, b, entity_idf)
    else:
        s = ec.score(a, b, sc, vectors)             # pre-Task-3 signature
        part = ec.participant_similarity(a, b)

    print(f"predicate={ec.predicate_similarity(a, b, vectors):.4f}  "
          f"participants={part:.4f}  "
          f"anchor={ec.anchor_compatibility(a, b, sc):.4f}  "
          f"modal={ec.modal_compatibility(a, b):.4f}  "
          f"class={ec.class_agreement(a, b):.4f}")
    print(f"score = {s:.4f}  (MATCH_THRESHOLD = {ec.MATCH_THRESHOLD})")

    if s < ec.MATCH_THRESHOLD:
        print("PASS: no-evidence pair scores below threshold")
        return 0
    print("FAIL: no-evidence pair clears the threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
