"""
Stage 2 - <SIGNAL> annotation (thesis Section 6.2.6).

A <SIGNAL> is emitted for each function word signalling a temporal relation,
with @pred carrying its normalised form. Signals carry no relation type of
their own; the relation is recorded on the link that references the signal.

The inventory covers temporal prepositions and subordinators, temporal adverbs
and, as the standard requires, punctuation functioning as a range marker.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..stage1_preprocessing.segmentation import SegmentedDocument, Sentence, Token
from .model import AnnotationStructure, Signal

#: surface -> (@pred, TLINK relType implied when this signal relates two events)
SIGNAL_LEXICON: Dict[str, Tuple[str, Optional[str]]] = {
    # prepositions and subordinators
    "before": ("BEFORE", "BEFORE"),
    "after": ("AFTER", "AFTER"),
    "during": ("DURING", "IS_INCLUDED"),
    "while": ("WHILE", "SIMULTANEOUS"),
    "when": ("WHEN", "IS_INCLUDED"),
    "whenever": ("WHENEVER", "IS_INCLUDED"),
    "until": ("UNTIL", "ENDS"),
    "till": ("UNTIL", "ENDS"),
    "since": ("SINCE", "BEGINS"),
    "as": ("AS", "SIMULTANEOUS"),
    "once": ("ONCE", "AFTER"),
    "by": ("BY", "BEFORE"),
    "within": ("WITHIN", "IS_INCLUDED"),
    "throughout": ("THROUGHOUT", "IS_INCLUDED"),
    "on": ("ON", "IS_INCLUDED"),
    "at": ("AT", "IS_INCLUDED"),
    "in": ("IN", "IS_INCLUDED"),
    "for": ("FOR", None),
    "from": ("FROM", "BEGINS"),
    "to": ("TO", "ENDS"),
    # adverbs
    "then": ("THEN", "AFTER"),
    "now": ("NOW", "SIMULTANEOUS"),
    "immediately": ("IMMEDIATELY", "IAFTER"),
    "at once": ("AT_ONCE", "IAFTER"),
    "straightway": ("IMMEDIATELY", "IAFTER"),
    "meanwhile": ("MEANWHILE", "SIMULTANEOUS"),
    "afterwards": ("AFTERWARDS", "AFTER"),
    "afterward": ("AFTERWARDS", "AFTER"),
    "later": ("LATER", "AFTER"),
    "earlier": ("EARLIER", "BEFORE"),
    "already": ("ALREADY", "BEFORE"),
    "still": ("STILL", "SIMULTANEOUS"),
    "yet": ("YET", "BEFORE"),
    "soon": ("SOON", "AFTER"),
    "shortly": ("SHORTLY", "AFTER"),
    "finally": ("FINALLY", "AFTER"),
    "next": ("NEXT", "AFTER"),
    "first": ("FIRST", "BEFORE"),
    "previously": ("PREVIOUSLY", "BEFORE"),
    "beforehand": ("BEFOREHAND", "BEFORE"),
}

MULTIWORD_SIGNALS: Dict[Tuple[str, ...], Tuple[str, Optional[str]]] = {
    ("as", "soon", "as"): ("AS_SOON_AS", "IAFTER"),
    ("at", "once"): ("AT_ONCE", "IAFTER"),
    ("as", "long", "as"): ("AS_LONG_AS", "SIMULTANEOUS"),
    ("no", "sooner"): ("NO_SOONER", "IAFTER"),
    ("in", "the", "meantime"): ("MEANWHILE", "SIMULTANEOUS"),
    ("from", "then", "on"): ("FROM_THEN_ON", "AFTER"),
    ("up", "to"): ("UP_TO", "ENDS"),
}

#: Punctuation functioning as a range marker (A.2.3.2).
RANGE_PUNCTUATION = {"-", "--", "/", "–", "—"}

#: Dependency labels a genuine temporal signal takes.
_SIGNAL_DEPS = {"mark", "prep", "advmod", "case", "cc", "npadvmod", "prt",
                "agent", "dative", "pcomp", "det"}


class SignalTagger:
    def tag(self, doc: SegmentedDocument, struct: AnnotationStructure) -> None:
        for sent in doc.sentences:
            self._tag_sentence(sent, struct)

    def _tag_sentence(self, sent: Sentence, struct: AnnotationStructure) -> None:
        toks = sent.tokens
        lowers = [t.text.lower() for t in toks]
        claimed: Set[int] = set()

        # multiword first
        for words, (pred, _rel) in MULTIWORD_SIGNALS.items():
            n = len(words)
            for i in range(len(lowers) - n + 1):
                if tuple(lowers[i:i + n]) == words and not (
                        set(range(i, i + n)) & claimed):
                    claimed.update(range(i, i + n))
                    self._emit(struct, toks[i:i + n], pred, sent)

        for i, tok in enumerate(toks):
            if i in claimed:
                continue
            low = lowers[i]
            if tok.text in RANGE_PUNCTUATION:
                # a range marker only between two numerals
                if 0 < i < len(toks) - 1 and toks[i - 1].pos == "NUM" \
                        and toks[i + 1].pos == "NUM":
                    self._emit(struct, [tok], "RANGE", sent)
                continue
            entry = SIGNAL_LEXICON.get(low)
            if entry is None:
                continue
            if not self._is_temporal_use(tok, toks, i, low):
                continue
            claimed.add(i)
            self._emit(struct, [tok], entry[0], sent)

    def _emit(self, struct: AnnotationStructure, span: Sequence[Token],
              pred: str, sent: Sentence) -> None:
        sid = struct.next_id("s")
        struct.add_signal(Signal(
            xml_id=sid,
            target=[t.xml_id for t in span],
            pred=pred,
            verse_key=span[0].verse_key,
            sent_id=sent.sent_id,
            text=" ".join(t.text for t in span),
        ))

    # -- disambiguation ---------------------------------------------------
    def _is_temporal_use(self, tok: Token, toks: Sequence[Token], i: int,
                         low: str) -> bool:
        """Only a temporal use is a <SIGNAL>. 'in the temple' is spatial,
        'in the morning' is temporal; 'as they approached' is temporal,
        'as a prophet' is not."""
        if tok.dep not in _SIGNAL_DEPS and tok.pos not in ("ADV", "SCONJ",
                                                           "ADP"):
            return False
        if low in {"before", "after", "during", "while", "until", "till",
                   "since", "then", "meanwhile", "afterwards", "afterward",
                   "immediately", "later", "earlier", "already", "soon",
                   "shortly", "finally", "previously", "beforehand",
                   "whenever", "throughout", "once"}:
            return True
        if low in {"on", "at", "in", "by", "within", "from", "to", "for",
                   "as", "next", "first", "now", "still", "yet"}:
            return self._governs_temporal(tok, toks, i)
        return True

    @staticmethod
    def _governs_temporal(tok: Token, toks: Sequence[Token], i: int) -> bool:
        from .event_tagger import TEMPORAL_NOUNS
        from .biblical_calendar import DAY_PARTS, FEASTS
        window = toks[i + 1:i + 6]
        for t in window:
            if t.lemma in TEMPORAL_NOUNS:
                return True
            if t.text.lower() in DAY_PARTS or t.text.lower() in FEASTS:
                return True
        # subordinating 'as'/'when' clause introducing a finite verb
        if tok.dep == "mark":
            return True
        return False
