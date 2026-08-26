"""
Stage 6 - the held-out chronology.

THIS IS THE ONLY MODULE IN THE FRAMEWORK THAT READS THE CHRONOLOGY FILE.

The chronology is treated as two distinct resources (thesis Section 9.1). Its
event descriptions and verse references are unavailable to the pipeline: the
framework must discover events and align them itself. Its event ordering is
available only here, as the reference against which the induced ordering is
compared. The separation is enforced in code rather than by convention: the
loader calls `assert_no_chronology_import`, which inspects the call stack and
raises if the load originates in any of stages 1 to 5.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from ..config import (BOOK_ORDER, CHRONOLOGY_FILE, DATA_DIR,
                      assert_no_chronology_import)
from ..stage1_preprocessing.corpus import Corpus, ReferenceParser


@dataclass
class CanonicalEvent:
    event_id: int
    day: str
    description: str
    when_where: str
    refs: Dict[str, str] = field(default_factory=dict)
    verse_keys: Dict[str, List[Tuple[str, int, int]]] = field(
        default_factory=dict)
    texts: Dict[str, str] = field(default_factory=dict)

    @property
    def books(self) -> List[str]:
        return [b for b in BOOK_ORDER if self.verse_keys.get(b)]

    @property
    def n_versions(self) -> int:
        return len(self.books)

    @property
    def all_keys(self) -> FrozenSet[Tuple[str, int, int]]:
        out: Set[Tuple[str, int, int]] = set()
        for ks in self.verse_keys.values():
            out |= set(ks)
        return frozenset(out)

    def contested(self) -> bool:
        return self.n_versions > 1


@dataclass
class Chronology:
    events: List[CanonicalEvent] = field(default_factory=list)

    def by_id(self, eid: int) -> Optional[CanonicalEvent]:
        return next((e for e in self.events if e.event_id == eid), None)

    def order(self) -> List[int]:
        return [e.event_id for e in self.events]

    def rank(self) -> Dict[int, int]:
        return {e.event_id: i for i, e in enumerate(self.events)}

    def contested(self) -> List[CanonicalEvent]:
        return [e for e in self.events if e.contested()]

    def day_distribution(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.events:
            out[e.day] = out.get(e.day, 0) + 1
        return out

    def version_distribution(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for e in self.events:
            out[e.n_versions] = out.get(e.n_versions, 0) + 1
        return out

    def versions_per_book(self) -> Dict[str, int]:
        out = {b: 0 for b in BOOK_ORDER}
        for e in self.events:
            for b in e.books:
                out[b] += 1
        return out

    def __len__(self) -> int:
        return len(self.events)


def load(corpus: Corpus, data_dir: Path = DATA_DIR) -> Chronology:
    assert_no_chronology_import()
    path = Path(data_dir) / CHRONOLOGY_FILE
    root = ET.parse(path).getroot()
    ch = Chronology()
    for node in root.findall(".//event"):
        eid = int(node.get("id"))
        desc = _text(node, "description")
        ev = CanonicalEvent(
            event_id=eid,
            day=_text(node, "day"),
            description=desc,
            when_where=_text(node, "when_where"),
        )
        for book in BOOK_ORDER:
            ref = _text(node, book)
            if not ref:
                continue
            ev.refs[book] = ref
            keys = corpus.span_keys(book, ref)
            if keys:
                ev.verse_keys[book] = keys
                ev.texts[book] = corpus.span_text(book, ref)
        ch.events.append(ev)
    return ch


def _text(node, tag: str) -> str:
    el = node.find(tag)
    if el is None or el.text is None:
        return ""
    return el.text.strip()
