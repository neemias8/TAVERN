"""
Stage 3 - the anchor scaffold (thesis Section 6.4.1).

An expression is ANCHORABLE if its normalised value, together with its anchor
chain, fixes its position relative to the week independently of the document in
which it occurs. "Six days before the Passover" is anchorable; "then" is not.

The scaffold is a shared day axis whose origin is the Passover / Day of
Preparation, whose position in the liturgical year the domain profile supplies
(Appendix A, Table tab:prof-feasts). Nothing here consults the chronology: the
liturgical calendar is external world knowledge declared by the profile, and
every position is derived from the annotation's own values and anchor chains.

    day  -6  ...  -1    0                 +1        +2
                     Passover /       Sabbath   first day
                     Preparation                of the week

Three kinds of evidence are combined.

  ABSOLUTE   a named feast, or a relative-day expression whose chain reaches
             one ("six days before the Passover"), fixes a day outright.
  INCREMENT  a narration-internal day shift. Two sources: a forward
             relative-day expression ("the next day"), and the onset of
             evening -- because under the profile the day boundary lies at
             sunset, so "when evening came" opens a new day interval.
  WITHIN-DAY an ordinal hour or a part of the day fixes a position inside
             whichever day it lands in, and says nothing about which day.

Absolute constraints and increments are solved together: the cumulative
increment count gives each unit a relative day, and the offset that aligns it
to the absolute constraints is taken as the median over those constraints, so
that a single mis-normalised expression cannot displace a whole document.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..stage2_temporal_annotation.model import AnnotationStructure, Timex3
from .local_timeline import EventUnit, LocalTimeline

#: Day offsets the profile's feast lexicon fixes on the shared axis.
FEAST_DAY: Dict[str, float] = {
    "PASSOVER": 0.0,
    "UNLEAVENED_BREAD": 0.0,
    "PREPARATION": 0.0,
    "SABBATH": 1.0,
    "FIRST_DAY": 2.0,
}

#: Position within the sunset-to-sunset day interval (Table tab:prof-parts).
DAYPART_POSITION: Dict[str, float] = {
    "EVENING": 0.02, "SUNSET": 0.02, "NIGHT": 0.13, "MIDNIGHT": 0.25,
    "COCKCROW": 0.38, "MORNING_WATCH": 0.45, "DAWN": 0.50, "MORNING": 0.58,
    "NOON": 0.71, "AFTERNOON": 0.83, "LATE": 0.95,
}

#: Ordinal-hour predicate -> position within the day interval. The daylight
#: half of the interval begins at 0.50 and each temporal hour is 1/24 of it.
HOUR_POSITION: Dict[str, float] = {
    f"HOUR_{name.upper()}": 0.50 + (i / 24.0)
    for i, name in enumerate(["first", "second", "third", "fourth", "fifth",
                              "sixth", "seventh", "eighth", "ninth", "tenth",
                              "eleventh", "twelfth"])
}

#: Dayparts whose onset opens a new day interval under the profile.
DAY_OPENING = {"EVENING", "SUNSET"}

DEFAULT_MIN_DAY = -6.0
DEFAULT_MAX_DAY = 2.0


@dataclass
class Anchor:
    anchor_id: str
    book: str
    unit_id: Optional[str]
    unit_index: Optional[int]
    verse_key: Tuple[str, int, int]
    surface: str
    pred: Optional[str]
    kind: str                       # feast | offset | hour | daypart
    absolute_day: Optional[float] = None
    increment: int = 0
    within_day: Optional[float] = None
    in_narration: bool = True
    position: Optional[float] = None


@dataclass
class Scaffold:
    anchors: List[Anchor] = field(default_factory=list)
    unit_position: Dict[str, float] = field(default_factory=dict)
    unit_interval: Dict[str, int] = field(default_factory=dict)
    boundaries: List[float] = field(default_factory=list)
    day_of_unit: Dict[str, int] = field(default_factory=dict)

    def interval_of(self, uid: str) -> Optional[int]:
        return self.unit_interval.get(uid)

    def position_of(self, uid: str) -> Optional[float]:
        return self.unit_position.get(uid)

    def anchorable_verses(self) -> int:
        return len({a.verse_key for a in self.anchors})


# ---------------------------------------------------------------------------
def build(structs: Dict[str, AnnotationStructure],
          timelines: Dict[str, LocalTimeline],
          enabled: bool = True) -> Scaffold:
    sc = Scaffold()
    if not enabled:
        # ablation: induction without the anchorable expressions. Documents are
        # registered against each other by relative position alone.
        for book, tl in timelines.items():
            n = max(1, len(tl.units) - 1)
            for i, u in enumerate(tl.units):
                p = DEFAULT_MIN_DAY + (DEFAULT_MAX_DAY - DEFAULT_MIN_DAY) * i / n
                sc.unit_position[u.unit_id] = p
                sc.day_of_unit[u.unit_id] = math.floor(p)
                u.day_index = float(math.floor(p))
                u.time_position = p - math.floor(p)
                sc.unit_interval[u.unit_id] = 0
                u.anchor_interval = 0
        return sc

    per_doc: Dict[str, List[Anchor]] = {}
    for book, struct in structs.items():
        tl = timelines[book]
        per_doc[book] = _document_anchors(struct, book, tl)
        sc.anchors.extend(per_doc[book])

    days = [a.absolute_day for a in sc.anchors if a.absolute_day is not None]
    lo = min(days) if days else DEFAULT_MIN_DAY
    hi = max(days) if days else DEFAULT_MAX_DAY

    for book, tl in timelines.items():
        _solve_document(tl, per_doc[book], sc, lo, hi)

    sc.boundaries = sorted({a.position for a in sc.anchors
                            if a.position is not None})
    _assign_intervals(sc, timelines)
    return sc


# ---------------------------------------------------------------------------
def _document_anchors(struct: AnnotationStructure, book: str,
                      tl: LocalTimeline) -> List[Anchor]:
    unit_index: Dict[Tuple[str, int, int], int] = {}
    unit_id: Dict[Tuple[str, int, int], str] = {}
    for i, u in enumerate(tl.units):
        for k in u.verse_keys:
            unit_index[k] = i
            unit_id[k] = u.unit_id

    meta = struct.__dict__.get("_timex_meta", {})
    out: List[Anchor] = []
    for tx in struct.timexes.values():
        if not tx.anchorable or not tx.verse_key:
            continue
        info = meta.get(tx.xml_id, {})
        narration = _is_narration(struct, tx)
        a = Anchor(
            anchor_id=tx.xml_id, book=book,
            unit_id=unit_id.get(tx.verse_key),
            unit_index=unit_index.get(tx.verse_key),
            verse_key=tx.verse_key, surface=tx.text, pred=tx.pred,
            kind=info.get("kind", "other"), in_narration=narration,
        )
        if tx.pred in FEAST_DAY:
            a.kind = "feast"
            if narration:
                a.absolute_day = FEAST_DAY[tx.pred]
        elif tx.pred in HOUR_POSITION:
            a.kind = "hour"
            a.within_day = HOUR_POSITION[tx.pred]
        elif tx.pred in DAYPART_POSITION:
            a.kind = "daypart"
            a.within_day = DAYPART_POSITION[tx.pred]
            if tx.pred in DAY_OPENING and narration:
                a.increment = 1
        elif a.kind == "relday":
            off = info.get("offset")
            base = _named_feast_in_scope(struct, tx)
            if base is not None and off is not None and narration \
                    and DEFAULT_MIN_DAY <= FEAST_DAY[base] + off <= DEFAULT_MAX_DAY:
                a.absolute_day = FEAST_DAY[base] + off
                a.kind = "offset-resolved"
            elif off == 1 and narration:
                # only a next-day expression is a day increment; "the third
                # day" is a prediction, not a step in the narration
                a.increment = 1
                a.kind = "offset-increment"
        out.append(a)
    out.sort(key=lambda a: (a.verse_key[1], a.verse_key[2]))

    # chain the unresolved forward offsets: "the next day" is one day after the
    # narrative reference time it anchors to, so an offset following a resolved
    # anchor is itself resolved (thesis Section 6.2.3)
    last: Optional[float] = None
    for a in out:
        if a.absolute_day is not None:
            last = a.absolute_day
        elif a.kind == "offset-increment" and last is not None:
            v = last + a.increment
            if DEFAULT_MIN_DAY <= v <= DEFAULT_MAX_DAY:
                a.absolute_day = v
                a.kind = "offset-chained"
                last = v
    return out


def _is_narration(struct: AnnotationStructure, tx: Timex3) -> bool:
    """True when the expression occurs in narration rather than direct speech.

    The test is the timeline eligibility of the expression's governing event --
    the annotated event nearest to it in its own sentence. An expression in the
    scope of a reporting event is governed by a subordinated event and is
    therefore not narration, which is what keeps "on the third day he will
    rise" out of the day axis (Sections 6.2.5, 6.3.1).
    """
    host = struct.timexes.get(tx.anchor_time_id) if tx.anchor_time_id else None
    if host is not None and host.anchor_level == 3:
        return False
    gov = _governing_event(struct, tx)
    if gov is None:
        return True
    return gov in struct.eligible


def _governing_event(struct: AnnotationStructure, tx: Timex3):
    if not tx.target:
        return None
    anchor = _token_order(tx.target[0])
    best, best_d = None, None
    for ev in struct.events.values():
        if ev.sent_id != tx.sent_id or not ev.target:
            continue
        d = abs(_token_order(ev.target[0]) - anchor)
        if best_d is None or d < best_d:
            best, best_d = ev.xml_id, d
    return best


def _token_order(xml_id: str) -> int:
    try:
        parts = xml_id.split("_")
        chap = int("".join(ch for ch in parts[1] if ch.isdigit()))
        return (chap * 10 ** 9 + int(parts[2]) * 10 ** 6
                + int(parts[3]) * 10 ** 3 + int(parts[4]))
    except (IndexError, ValueError):
        return 0


def _named_feast_in_scope(struct: AnnotationStructure,
                          tx: Timex3) -> Optional[str]:
    for other in struct.timexes.values():
        if other.xml_id == tx.xml_id or other.sent_id != tx.sent_id:
            continue
        if other.pred in FEAST_DAY:
            return other.pred
    return None


# ---------------------------------------------------------------------------
def _solve_document(tl: LocalTimeline, anchors: List[Anchor], sc: Scaffold,
                    global_lo: float, global_hi: float) -> None:
    """Register the document onto the shared day axis.

    Phase 1 pins the units carrying an absolute day constraint and interpolates
    between them; the opening and closing stretches run out to the days the
    corpus's anchors actually reach, rather than being extrapolated at an
    arbitrary rate. Phase 2 adds the within-day anchors as further pins, now
    that the day each one falls in is known, and re-interpolates. This is what
    makes "the third hour", "the sixth hour" and "the ninth hour" do real work
    on the crucifixion day rather than being averaged away.
    """
    n = len(tl.units)
    if n == 0:
        return

    # --- phase 1: the day axis from absolute constraints -----------------
    pins: Dict[int, float] = {}
    for a in anchors:
        if a.absolute_day is None or a.unit_index is None:
            continue
        i = a.unit_index
        if i not in pins or a.absolute_day < pins[i]:
            pins[i] = a.absolute_day

    positions = _interpolate_pins(pins, n, global_lo, global_hi)

    # --- phase 2: within-day anchors -------------------------------------
    for a in anchors:
        if a.within_day is None or a.unit_index is None:
            continue
        i = a.unit_index
        day = math.floor(positions[i] + 1e-9)
        v = day + a.within_day
        if i not in pins or v < pins[i]:
            pins[i] = v
    positions = _interpolate_pins(pins, n, global_lo, global_hi)

    for i, u in enumerate(tl.units):
        p = positions[i]
        sc.unit_position[u.unit_id] = p
        d = math.floor(p + 1e-9)
        sc.day_of_unit[u.unit_id] = d
        u.day_index = float(d)
        u.time_position = p - d

    for a in anchors:
        if a.unit_id in sc.unit_position:
            a.position = sc.unit_position[a.unit_id]


def _interpolate_pins(pins: Dict[int, float], n: int, lo: float,
                      hi: float) -> List[float]:
    """Piecewise-linear registration through a non-decreasing set of pins."""
    if not pins:
        return [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]

    # Only strictly increasing pins are retained. Two consecutive pins of equal
    # value -- the Day of Preparation and the Passover both fall on day 0 --
    # would otherwise flatten everything between them onto a single point,
    # losing the order of a whole day's narrative. Keeping the first occurrence
    # instead spreads that stretch across the interval between the two
    # surrounding anchors, which is what Section 6.4.1 means by a block of
    # narrative being confined to an interval.
    mono: List[Tuple[int, float]] = []
    run: Optional[float] = None
    for i in sorted(pins):
        v = pins[i]
        if run is None or v > run:
            mono.append((i, v))
            run = v

    out = [0.0] * n
    first_i, first_v = mono[0]
    last_i, last_v = mono[-1]

    for i in range(n):
        if i < first_i:
            out[i] = (min(lo, first_v) + (first_v - min(lo, first_v))
                      * i / first_i) if first_i else first_v
        elif i > last_i:
            span = (n - 1) - last_i
            top = max(hi, last_v)
            out[i] = last_v + (top - last_v) * (i - last_i) / span if span \
                else last_v
        else:
            out[i] = last_v
            for (a_i, a_v), (b_i, b_v) in zip(mono, mono[1:]):
                if a_i <= i <= b_i:
                    out[i] = a_v if b_i == a_i else \
                        a_v + (b_v - a_v) * (i - a_i) / (b_i - a_i)
                    break
    return out


def _assign_intervals(sc: Scaffold, timelines) -> None:
    bounds = sc.boundaries
    for tl in timelines.values():
        for u in tl.units:
            p = sc.unit_position.get(u.unit_id)
            if p is None:
                sc.unit_interval[u.unit_id] = -1
                continue
            idx = sum(1 for b in bounds if b <= p)
            sc.unit_interval[u.unit_id] = idx
            u.anchor_interval = idx
