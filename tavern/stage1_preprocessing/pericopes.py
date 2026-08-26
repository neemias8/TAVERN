"""
Stage 1 - pericope layer.

Attaches the 91 titled narrative units to the document model by verse
reference (thesis Section 5.2). The layer is read from the
NIV_*_PW_with_pericopes.xml files, which carry the newer schema.

Two data defects in those files are handled here rather than by editing the
user's data:

  1. They are 66 verses shorter than the digest-verified verse files
     (1,179 vs 1,245), because the last pericope of Mark, Luke, Matthew and
     John stops before the end of the book. Since Mark 16:9-20 and
     Luke 24:13-53 are cited by the chronology, the final pericope of each
     book is extended to the last verse present in the verse files.
  2. A verse falling in no declared pericope is attached to the nearest
     preceding one; if there is none, a synthetic unit is opened.

Both repairs are reported, so the pericope layer's provenance is auditable.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import BOOK_CODE, DATA_DIR, PERICOPE_FILES
from .corpus import Corpus


@dataclass
class Pericope:
    pericope_id: str
    book: str
    title: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    verse_keys: List[Tuple[str, int, int]] = field(default_factory=list)
    synthetic: bool = False
    extended: bool = False

    def contains(self, chapter: int, verse: int) -> bool:
        return self.start <= (chapter, verse) <= self.end


@dataclass
class PericopeLayer:
    pericopes: List[Pericope] = field(default_factory=list)
    by_verse: Dict[Tuple[str, int, int], str] = field(default_factory=dict)
    repairs: List[str] = field(default_factory=list)

    def by_id(self, pid: str) -> Optional[Pericope]:
        for p in self.pericopes:
            if p.pericope_id == pid:
                return p
        return None

    def of_verse(self, key: Tuple[str, int, int]) -> Optional[str]:
        return self.by_verse.get(key)

    def for_book(self, book: str) -> List[Pericope]:
        return [p for p in self.pericopes if p.book == book]

    def __len__(self) -> int:
        return len(self.pericopes)


def load_pericopes(corpus: Corpus, data_dir: Path = DATA_DIR) -> PericopeLayer:
    layer = PericopeLayer()
    for book in corpus.books:
        path = Path(data_dir) / PERICOPE_FILES[book]
        raw: List[Pericope] = []
        if path.exists():
            root = ET.parse(path).getroot()
            n = 0
            for chapter in root.findall("chapter"):
                for per in chapter.findall("pericope"):
                    n += 1
                    raw.append(Pericope(
                        pericope_id=f"{BOOK_CODE[book]}_p{n:03d}",
                        book=book,
                        title=(per.get("title") or "").strip(),
                        start=(int(per.get("start_chapter")),
                               int(per.get("start_verse"))),
                        end=(int(per.get("end_chapter")),
                             int(per.get("end_verse"))),
                    ))
        raw.sort(key=lambda p: p.start)

        verses = corpus[book].verses
        if not verses:
            continue
        last_key = (verses[-1].chapter, verses[-1].number)

        # repair 1: extend the final pericope to the last verse of the book
        if raw and raw[-1].end < last_key:
            layer.repairs.append(
                f"{book}: final pericope '{raw[-1].title}' extended from "
                f"{raw[-1].end[0]}:{raw[-1].end[1]} to {last_key[0]}:{last_key[1]}"
            )
            raw[-1].end = last_key
            raw[-1].extended = True

        # repair 2: verses outside every declared pericope
        synthetic: List[Pericope] = []
        for v in verses:
            ck = (v.chapter, v.number)
            if any(p.contains(*ck) for p in raw):
                continue
            host = None
            for p in reversed(raw):
                if p.start <= ck:
                    host = p
                    break
            if host is not None:
                host.end = max(host.end, ck)
                host.extended = True
            else:
                synthetic.append(Pericope(
                    pericope_id=f"{BOOK_CODE[book]}_p000",
                    book=book,
                    title=f"[untitled opening of {book}]",
                    start=ck, end=ck, synthetic=True,
                ))
        if synthetic:
            merged = synthetic[0]
            merged.end = max(s.end for s in synthetic)
            raw.insert(0, merged)
            layer.repairs.append(
                f"{book}: synthetic opening pericope for "
                f"{merged.start[0]}:{merged.start[1]}-{merged.end[0]}:{merged.end[1]}"
            )

        raw.sort(key=lambda p: p.start)
        for p in raw:
            p.verse_keys = [v.key for v in verses if p.contains(v.chapter, v.number)]
            for k in p.verse_keys:
                layer.by_verse[k] = p.pericope_id
            layer.pericopes.append(p)

    return layer
