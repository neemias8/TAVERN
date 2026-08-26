"""
Stage 1 - document model.

Parses the two XML schema variants that occur in the resource (thesis
Section 5.2): the older form nesting <book> inside <testament>, and the newer
form carrying the book as an attribute of the root. Verse text always comes
from the older, digest-verified files; the newer files supply only the
pericope layer.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..config import (BOOK_ORDER, DATA_DIR, GOSPEL_FILES, GOSPEL_SCOPE)


@dataclass
class Verse:
    book: str
    chapter: int
    number: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.book}:{self.chapter}:{self.number}"

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.book, self.chapter, self.number)


@dataclass
class Document:
    book: str
    verses: List[Verse] = field(default_factory=list)
    _index: Dict[Tuple[int, int], Verse] = field(default_factory=dict, repr=False)

    def add(self, v: Verse) -> None:
        self.verses.append(v)
        self._index[(v.chapter, v.number)] = v

    def get(self, chapter: int, number: int) -> Optional[Verse]:
        return self._index.get((chapter, number))

    def text(self) -> str:
        return " ".join(v.text for v in self.verses)

    def __len__(self) -> int:
        return len(self.verses)


class VerseSplitter:
    """Partial verse references (e.g. '14a', '14b').

    Ported unchanged from the IJCNN implementation so that verse-level
    segmentation matches the published baselines exactly.
    """

    @staticmethod
    def smart_split(text: str, part: Optional[str]) -> str:
        if not text or part not in ("a", "b"):
            return text
        tokens = re.split(r"([.;:?!]+)\s+", text)
        sentences: List[str] = []
        current = ""
        for token in tokens:
            if re.match(r"[.;:?!]+", token):
                current += token
                sentences.append(current.strip())
                current = ""
            else:
                current += token
        if current:
            sentences.append(current.strip())
        if not sentences:
            return text
        mid = max(1, len(sentences) // 2)
        return " ".join(sentences[:mid]) if part == "a" else " ".join(sentences[mid:])


class ReferenceParser:
    """Verse-reference strings, including cross-chapter ranges.

    Ported from the IJCNN implementation, with one correction: the
    cross-chapter branch there iterated `range(v1, 100)` regardless of the
    real chapter length, which silently produced non-existent addresses. Here
    the caller supplies a bound.
    """

    @staticmethod
    def parse(ref_str: str, chapter_len=None) -> List[Tuple[int, int, Optional[str]]]:
        refs: List[Tuple[int, int, Optional[str]]] = []
        if not ref_str:
            return refs
        ref_str = ref_str.strip()
        cross = re.match(r"^(\d+):(\d+)\s*-\s*(\d+):(\d+)$", ref_str)
        if cross:
            c1, v1, c2, v2 = map(int, cross.groups())
            last = chapter_len(c1) if chapter_len else 100
            for v in range(v1, last + 1):
                refs.append((c1, v, None))
            for c in range(c1 + 1, c2):
                lc = chapter_len(c) if chapter_len else 0
                for v in range(1, lc + 1):
                    refs.append((c, v, None))
            for v in range(1, v2 + 1):
                refs.append((c2, v, None))
            return refs
        if ":" not in ref_str:
            return refs
        try:
            chapter_part, verses_part = ref_str.split(":", 1)
            chapter = int(chapter_part.strip())
        except ValueError:
            return refs
        if "-" in verses_part:
            start_s, end_s = verses_part.split("-", 1)
            sv, sp = ReferenceParser._token(start_s)
            ev, ep = ReferenceParser._token(end_s)
            if sv is None or ev is None:
                return refs
            for v in range(sv, ev + 1):
                p = None
                if v == sv:
                    p = sp
                if v == ev:
                    p = ep
                refs.append((chapter, v, p))
        else:
            v, p = ReferenceParser._token(verses_part)
            if v is not None:
                refs.append((chapter, v, p))
        return refs

    @staticmethod
    def _token(tok: str):
        m = re.match(r"\s*(\d+)([ab]?)", tok)
        if not m:
            return None, None
        return int(m.group(1)), (m.group(2) or None)


class Corpus:
    """The four Gospels, restricted to the canonical Passion Week scope."""

    def __init__(self, data_dir: Path = DATA_DIR, books: Iterable[str] = BOOK_ORDER,
                 restrict_scope: bool = True):
        self.data_dir = Path(data_dir)
        self.documents: Dict[str, Document] = {}
        for book in books:
            self.documents[book] = self._load(book, restrict_scope)

    # -- loading ----------------------------------------------------------
    def _load(self, book: str, restrict_scope: bool) -> Document:
        path = self.data_dir / GOSPEL_FILES[book]
        root = ET.parse(path).getroot()
        book_node = root.find(".//book")
        if book_node is None:            # newer schema: book is an attribute
            book_node = root
        doc = Document(book=book)
        lo, hi = GOSPEL_SCOPE[book]
        for chapter in book_node.findall("chapter"):
            cnum = int(chapter.get("number"))
            if restrict_scope and not (lo <= cnum <= hi):
                continue
            for verse in chapter.findall("verse"):
                vnum = int(verse.get("number"))
                text = (verse.text or "").strip()
                if not text:
                    continue
                doc.add(Verse(book=book, chapter=cnum, number=vnum, text=text))
        return doc

    # -- access -----------------------------------------------------------
    def __getitem__(self, book: str) -> Document:
        return self.documents[book]

    @property
    def books(self) -> List[str]:
        return [b for b in BOOK_ORDER if b in self.documents]

    def chapter_len(self, book: str):
        doc = self.documents[book]

        def _len(c: int) -> int:
            nums = [v.number for v in doc.verses if v.chapter == c]
            return max(nums) if nums else 0

        return _len

    def get_text(self, book: str, chapter: int, number: int,
                 part: Optional[str] = None) -> str:
        v = self.documents[book].get(chapter, number)
        if v is None:
            return ""
        return VerseSplitter.smart_split(v.text, part) if part else v.text

    def span_text(self, book: str, ref_str: str) -> str:
        refs = ReferenceParser.parse(ref_str, self.chapter_len(book))
        out = [self.get_text(book, c, v, p) for c, v, p in refs]
        return " ".join(t for t in out if t)

    def span_keys(self, book: str, ref_str: str) -> List[Tuple[str, int, int]]:
        refs = ReferenceParser.parse(ref_str, self.chapter_len(book))
        return [(book, c, v) for c, v, _ in refs
                if self.documents[book].get(c, v) is not None]

    def verse_count(self) -> Dict[str, int]:
        return {b: len(d) for b, d in self.documents.items()}

    def total_verses(self) -> int:
        return sum(len(d) for d in self.documents.values())

    def word_count(self) -> int:
        return sum(len(v.text.split()) for d in self.documents.values()
                   for v in d.verses)
