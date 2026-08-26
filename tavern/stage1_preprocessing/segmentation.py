"""
Stage 1 - base segmentation layer.

Produces the addressable token layer into which the stand-off annotation of
Stage 2 points (thesis Section 5.2 / 2.x on stand-off annotation).

Punctuation is retained as separate tokens, because ISO 24617-1 admits
punctuation as the target of a <SIGNAL> (range characters such as '-' and '/'),
and a tokeniser that discards them forecloses conformant annotation of those
cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import spacy

from ..config import BOOK_CODE
from .corpus import Corpus, Verse

_NLP = None


def get_nlp(model: str = "en_core_web_sm"):
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(model)
    return _NLP


@dataclass
class Token:
    xml_id: str
    text: str
    lemma: str
    pos: str            # coarse spaCy POS
    tag: str            # fine PTB tag
    dep: str
    head: int           # index into the sentence's token list (local)
    morph: str
    idx_in_verse: int
    verse_key: Tuple[str, int, int]
    sent_id: str
    is_punct: bool = False
    ent_type: str = ""

    @property
    def ref(self) -> str:
        b, c, v = self.verse_key
        return f"{b}:{c}:{v}"


@dataclass
class Sentence:
    sent_id: str
    verse_key: Tuple[str, int, int]
    tokens: List[Token] = field(default_factory=list)
    doc: object = field(default=None, repr=False)   # spaCy Span

    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)


@dataclass
class SegmentedDocument:
    book: str
    sentences: List[Sentence] = field(default_factory=list)
    tokens_by_id: Dict[str, Token] = field(default_factory=dict, repr=False)
    sentences_by_verse: Dict[Tuple[str, int, int], List[Sentence]] = \
        field(default_factory=dict, repr=False)

    def add_sentence(self, s: Sentence) -> None:
        self.sentences.append(s)
        self.sentences_by_verse.setdefault(s.verse_key, []).append(s)
        for t in s.tokens:
            self.tokens_by_id[t.xml_id] = t

    def all_tokens(self) -> Iterator[Token]:
        for s in self.sentences:
            yield from s.tokens

    def token(self, xml_id: str) -> Optional[Token]:
        return self.tokens_by_id.get(xml_id)


class Segmenter:
    """Tokenises each verse and assigns xml:id to every token."""

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = get_nlp(model)

    def segment_corpus(self, corpus: Corpus) -> Dict[str, SegmentedDocument]:
        return {book: self.segment(corpus[book].verses, book)
                for book in corpus.books}

    def segment(self, verses: List[Verse], book: str) -> SegmentedDocument:
        out = SegmentedDocument(book=book)
        texts = [v.text for v in verses]
        for verse, sdoc in zip(verses, self.nlp.pipe(texts, batch_size=64)):
            for si, sent in enumerate(sdoc.sents):
                sent_id = f"s_{BOOK_CODE.get(book, book[:2])}{verse.chapter}_{verse.number}_{si}"
                toks: List[Token] = []
                base = sent.start
                for ti, tok in enumerate(sent):
                    xid = f"tk_{BOOK_CODE.get(book, book[:2])}{verse.chapter}_{verse.number}_{si}_{ti}"
                    toks.append(Token(
                        xml_id=xid,
                        text=tok.text,
                        lemma=tok.lemma_.lower(),
                        pos=tok.pos_,
                        tag=tok.tag_,
                        dep=tok.dep_,
                        head=tok.head.i - base,
                        morph=str(tok.morph),
                        idx_in_verse=tok.i,
                        verse_key=verse.key,
                        sent_id=sent_id,
                        is_punct=tok.is_punct,
                        ent_type=tok.ent_type_,
                    ))
                out.add_sentence(Sentence(sent_id=sent_id, verse_key=verse.key,
                                          tokens=toks, doc=sent))
        return out
