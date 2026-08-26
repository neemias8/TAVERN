"""
Stage 2 - <TIMEX3> annotation (thesis Section 6.2.3).

Detection combines the domain lexicon of `biblical_calendar` with syntactic
patterns. The pattern requiring most care is the relative day expression --
"the next day", "on the third day", "six days before the Passover" -- which is
DATE with @temporalFunction="true", an @anchorTimeID and a @value
underspecified in exactly the positions the expression does not supply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..stage1_preprocessing.pericopes import PericopeLayer
from ..stage1_preprocessing.segmentation import SegmentedDocument, Sentence, Token
from .biblical_calendar import (DAY_PARTS, DURATION_UNITS, FEASTS,
                                NIGHT_WATCHES, NUMBER_WORDS, ORDINAL_HOURS,
                                ORDINAL_NUMBER)
from .enums import TimexType
from .model import AnnotationStructure, Timex3

# Recurrence markers -> SET
SET_PATTERNS = {
    "every day": ("P1D", "EVERY", None),
    "each day": ("P1D", "EVERY", None),
    "daily": ("P1D", None, "1D"),
    "day after day": ("P1D", "EVERY", None),
    "every year": ("P1Y", "EVERY", None),
    "yearly": ("P1Y", None, "1Y"),
    "every sabbath": ("P1W", "EVERY", None),
    "night and day": ("P1D", "EVERY", None),
    "day and night": ("P1D", "EVERY", None),
}

# Deictic day expressions resolved against the anchoring hierarchy
RELATIVE_DAY = {
    "the next day": 1, "the following day": 1, "the day after": 1,
    "the third day": 3, "the second day": 2, "the day before": -1,
    "yesterday": -1, "today": 0, "tomorrow": 1, "that day": 0,
    "that same day": 0, "the same day": 0, "this day": 0,
    "the first day": 1, "the last day": None, "the great day": None,
    "the day of judgement": None, "the day of judgment": None,
}

# Multi-word expressions worth matching before single tokens.
_ORDINAL_HOUR_RE = re.compile(
    r"\b(?:about |around )?the (first|second|third|fourth|fifth|sixth|seventh"
    r"|eighth|ninth|tenth|eleventh|twelfth) hour\b", re.I)

_N_DAYS_BEFORE_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|forty|\d+)\s+"
    r"(day|days|hour|hours|week|weeks|year|years)\s+"
    r"(before|after|later|earlier)\b", re.I)

_DURATION_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|a|an|\d+)\s+"
    r"(second|seconds|minute|minutes|hour|hours|day|days|week|weeks|"
    r"month|months|year|years)\b", re.I)


@dataclass
class TimexCandidate:
    start: int
    end: int                # exclusive
    surface: str
    timex_type: TimexType
    pred: Optional[str] = None
    kind: str = "other"     # feast | hour | daypart | duration | set | relday | watch
    payload: Optional[dict] = None


class TimexTagger:
    def __init__(self, pericopes: Optional[PericopeLayer] = None):
        self.pericopes = pericopes

    def tag(self, doc: SegmentedDocument, struct: AnnotationStructure) -> None:
        for sent in doc.sentences:
            for cand in self._candidates(sent):
                toks = sent.tokens[cand.start:cand.end]
                if not toks:
                    continue
                tid = struct.next_id("t")
                tx = Timex3(
                    xml_id=tid,
                    timex_type=cand.timex_type,
                    target=[t.xml_id for t in toks],
                    pred=cand.pred,
                    verse_key=toks[0].verse_key,
                    sent_id=sent.sent_id,
                    pericope_id=(self.pericopes.of_verse(toks[0].verse_key)
                                 if self.pericopes else None),
                    text=cand.surface,
                )
                tx_kind = cand.kind
                tx_payload = cand.payload or {}
                tx_payload["kind"] = tx_kind
                struct.add_timex(tx)
                # normalisation is performed by timex_normalizer, which needs
                # the detection metadata
                struct.__dict__.setdefault("_timex_meta", {})[tid] = tx_payload

    # -- candidate detection ----------------------------------------------
    def _candidates(self, sent: Sentence) -> List[TimexCandidate]:
        toks = sent.tokens
        lowers = [t.text.lower() for t in toks]
        text = " ".join(lowers)
        taken: Set[int] = set()
        out: List[TimexCandidate] = []

        def claim(c: TimexCandidate) -> None:
            rng = set(range(c.start, c.end))
            if rng & taken:
                return
            taken.update(rng)
            out.append(c)

        # 1. ordinal hours ("the third hour")
        for m in self._match_phrase(lowers, _ORDINAL_HOUR_RE, text):
            s, e, surface = m
            ordinal = None
            for o in ORDINAL_HOURS:
                if o in surface.lower():
                    ordinal = o
                    break
            claim(TimexCandidate(s, e, surface, TimexType.TIME,
                                 pred=f"HOUR_{(ordinal or '').upper()}",
                                 kind="hour", payload={"ordinal": ordinal}))

        # 2. "N days before/after ..."
        for m in self._match_phrase(lowers, _N_DAYS_BEFORE_RE, text):
            s, e, surface = m
            parts = surface.lower().split()
            n = NUMBER_WORDS.get(parts[0])
            if n is None:
                try:
                    n = int(parts[0])
                except ValueError:
                    n = None
            unit = DURATION_UNITS.get(parts[1], "D")
            direction = -1 if parts[2] in ("before", "earlier") else 1
            # the whole phrase is a DATE with a temporal function; the offset is
            # carried by @anchorTimeID plus the duration, never by @value
            claim(TimexCandidate(s, e, surface, TimexType.DATE,
                                 pred="OFFSET", kind="relday",
                                 payload={"offset": (n or 0) * direction,
                                          "unit": unit, "count": n}))

        # 3. named feasts
        for surf, feast in FEASTS.items():
            for s, e in self._find_ngram(lowers, surf.split()):
                if self._is_event_reading(toks, s, e):
                    continue
                claim(TimexCandidate(s, e, " ".join(lowers[s:e]),
                                     feast.timex_type, pred=feast.pred,
                                     kind="feast", payload={"feast": surf}))

        # 4. night watches
        for surf in NIGHT_WATCHES:
            for s, e in self._find_ngram(lowers, surf.split()):
                v, pred = NIGHT_WATCHES[surf]
                claim(TimexCandidate(s, e, surf, TimexType.TIME, pred=pred,
                                     kind="watch", payload={"watch": surf}))

        # 5. recurrence -> SET
        for surf, (val, quant, freq) in SET_PATTERNS.items():
            for s, e in self._find_ngram(lowers, surf.split()):
                claim(TimexCandidate(s, e, surf, TimexType.SET, pred="RECUR",
                                     kind="set",
                                     payload={"value": val, "quant": quant,
                                              "freq": freq}))

        # 6. relative day expressions
        for surf, off in sorted(RELATIVE_DAY.items(), key=lambda kv: -len(kv[0])):
            for s, e in self._find_ngram(lowers, surf.split()):
                claim(TimexCandidate(s, e, surf, TimexType.DATE, pred="RELDAY",
                                     kind="relday",
                                     payload={"offset": off, "unit": "D",
                                              "deictic": surf in
                                              ("today", "tomorrow",
                                               "yesterday")}))

        # 7. parts of the day
        for surf in sorted(DAY_PARTS, key=lambda s: -len(s.split())):
            for s, e in self._find_ngram(lowers, surf.split()):
                val, pred, pos = DAY_PARTS[surf]
                claim(TimexCandidate(s, e, surf, TimexType.TIME, pred=pred,
                                     kind="daypart",
                                     payload={"time": val, "position": pos,
                                              "surface": surf}))

        # 8. bare durations ("three days", "forty days", "three hours")
        for m in self._match_phrase(lowers, _DURATION_RE, text):
            s, e, surface = m
            parts = surface.lower().split()
            n = NUMBER_WORDS.get(parts[0])
            if n is None:
                try:
                    n = int(parts[0])
                except ValueError:
                    n = None
            unit = DURATION_UNITS.get(parts[1], "D")
            claim(TimexCandidate(s, e, surface, TimexType.DURATION,
                                 pred="DURATION", kind="duration",
                                 payload={"count": n, "unit": unit}))

        out.sort(key=lambda c: c.start)
        return out

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _find_ngram(lowers: Sequence[str], words: Sequence[str]):
        n = len(words)
        for i in range(len(lowers) - n + 1):
            if list(lowers[i:i + n]) == list(words):
                yield i, i + n

    @staticmethod
    def _match_phrase(lowers: Sequence[str], regex: re.Pattern, joined: str):
        """Map a regex match over the joined token string back to token
        indices. Tokens are joined by single spaces, so offsets are recoverable.
        """
        offsets = []
        pos = 0
        for w in lowers:
            offsets.append((pos, pos + len(w)))
            pos += len(w) + 1
        for m in regex.finditer(joined):
            s_char, e_char = m.span()
            s = next((i for i, (a, b) in enumerate(offsets) if a >= s_char), None)
            e = None
            for i, (a, b) in enumerate(offsets):
                if b <= e_char:
                    e = i + 1
            if s is not None and e is not None and e > s:
                yield s, e, m.group(0)

    @staticmethod
    def _is_event_reading(toks: Sequence[Token], s: int, e: int) -> bool:
        """'Passover' is ambiguous between the date and the meal. The
        distinction is resolved by the governing predicate: 'before the
        Passover' is temporal, 'eat the Passover' is an event
        (Appendix A, Section A.6).
        """
        surface = " ".join(t.text.lower() for t in toks[s:e])
        if "passover" not in surface:
            return False
        head_idx = toks[e - 1].head
        if 0 <= head_idx < len(toks):
            if toks[head_idx].lemma in {"eat", "prepare", "sacrifice", "keep",
                                        "celebrate", "kill"}:
                return True
        for i in range(max(0, s - 3), s):
            if toks[i].lemma in {"eat", "prepare", "sacrifice", "keep",
                                 "celebrate", "kill"}:
                return True
        return False
