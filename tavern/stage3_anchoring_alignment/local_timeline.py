"""
Stage 3 - local timelines and event units (thesis Sections 6.3.3, 7.2.1).

Two things happen here.

First, event mentions are grouped into EVENT UNITS. A graph node is one *event
version*: "the description of a candidate canonical event in one document"
(Section 7.2.1), which is a verse span rather than a single verb. Segmentation
is driven by the annotation itself: a unit boundary is opened where the
annotation asserts a new temporal position (an anchorable <TIMEX3>), where a
clause-initial temporal <SIGNAL> marks a shift, where the participant set
turns over, or at a pericope boundary.

Second, the closed network over eligible events induces a partial order over
the units of each document. Where the network is not total the partial order is
retained as such: forcing a total order here would fabricate precedence
relations the text does not support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from ..config import BOOK_CODE
from ..stage1_preprocessing.coref import EntityChains
from ..stage1_preprocessing.corpus import Corpus
from ..stage1_preprocessing.pericopes import PericopeLayer
from ..stage2_temporal_annotation.closure import implies_before
from ..stage2_temporal_annotation.model import AnnotationStructure, Event

#: Signals whose clause-initial occurrence marks a narrative shift.
SHIFT_SIGNALS = {"THEN", "AFTER", "AFTERWARDS", "WHEN", "IMMEDIATELY",
                 "AT_ONCE", "LATER", "NEXT", "MEANWHILE", "AS_SOON_AS",
                 "FINALLY", "NOW", "ONCE"}

MAX_UNIT_VERSES = 6
MIN_UNIT_VERSES = 1
ENTITY_TURNOVER = 0.60


@dataclass
class EventUnit:
    unit_id: str
    book: str
    pericope_id: Optional[str]
    verse_keys: List[Tuple[str, int, int]]
    event_ids: List[str] = field(default_factory=list)
    eligible_ids: List[str] = field(default_factory=list)
    timex_ids: List[str] = field(default_factory=list)
    timex_preds: List[str] = field(default_factory=list)
    anchorable_timex: bool = False
    signal_preds: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    text: str = ""
    preds: List[str] = field(default_factory=list)
    classes: Set[str] = field(default_factory=set)
    types: Set[str] = field(default_factory=set)
    tenses: Set[str] = field(default_factory=set)
    aspects: Set[str] = field(default_factory=set)
    pos_tags: Set[str] = field(default_factory=set)
    neg_fraction: float = 0.0
    modal_depth: float = 0.0
    modal_types: Set[str] = field(default_factory=set)
    eligible_fraction: float = 0.0
    day_index: Optional[float] = None
    time_position: Optional[float] = None
    anchor_interval: Optional[int] = None
    level_profile: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def ref(self) -> str:
        c0, v0 = self.verse_keys[0][1], self.verse_keys[0][2]
        c1, v1 = self.verse_keys[-1][1], self.verse_keys[-1][2]
        if (c0, v0) == (c1, v1):
            return f"{self.book.capitalize()} {c0}:{v0}"
        if c0 == c1:
            return f"{self.book.capitalize()} {c0}:{v0}-{v1}"
        return f"{self.book.capitalize()} {c0}:{v0}-{c1}:{v1}"

    @property
    def key_set(self) -> FrozenSet[Tuple[str, int, int]]:
        return frozenset(self.verse_keys)

    def __len__(self) -> int:
        return len(self.verse_keys)


@dataclass
class LocalTimeline:
    book: str
    units: List[EventUnit] = field(default_factory=list)
    unit_of_event: Dict[str, str] = field(default_factory=dict)
    order: Set[Tuple[str, str]] = field(default_factory=set)   # (before, after)
    order_confidence: Dict[Tuple[str, str], float] = field(default_factory=dict)
    order_level: Dict[Tuple[str, str], int] = field(default_factory=dict)
    nrt_chain: Set[Tuple[str, str]] = field(default_factory=set)

    def by_id(self, uid: str) -> Optional[EventUnit]:
        return next((u for u in self.units if u.unit_id == uid), None)


def segment(struct: AnnotationStructure, corpus: Corpus,
            pericopes: PericopeLayer, chains: EntityChains) -> LocalTimeline:
    book = struct.book
    tl = LocalTimeline(book=book)
    verses = corpus[book].verses

    events_by_verse: Dict[Tuple[str, int, int], List[Event]] = {}
    for ev in struct.events.values():
        if ev.verse_key:
            events_by_verse.setdefault(ev.verse_key, []).append(ev)
    timex_by_verse: Dict[Tuple[str, int, int], List] = {}
    for tx in struct.timexes.values():
        if tx.verse_key:
            timex_by_verse.setdefault(tx.verse_key, []).append(tx)
    signals_by_verse: Dict[Tuple[str, int, int], List] = {}
    for sg in struct.signals.values():
        if sg.verse_key:
            signals_by_verse.setdefault(sg.verse_key, []).append(sg)

    n = 0
    current: List[Tuple[str, int, int]] = []
    current_entities: Set[str] = set()
    current_pericope: Optional[str] = None

    def flush():
        nonlocal current, current_entities, current_pericope, n
        if not current:
            return
        n += 1
        uid = f"{BOOK_CODE[book]}_u{n:03d}"
        unit = _build_unit(uid, book, current_pericope, current, struct,
                           corpus, events_by_verse, timex_by_verse,
                           signals_by_verse, chains)
        tl.units.append(unit)
        for eid in unit.event_ids:
            tl.unit_of_event[eid] = uid
        current = []
        current_entities = set()

    for v in verses:
        pid = pericopes.of_verse(v.key)
        ents = chains.by_verse.get(v.key, set())
        boundary = False
        if current and pid != current_pericope:
            boundary = True
        elif current:
            if len(current) >= MAX_UNIT_VERSES:
                boundary = True
            elif any(t.anchorable for t in timex_by_verse.get(v.key, [])):
                boundary = True
            elif any(s.pred in SHIFT_SIGNALS
                     for s in signals_by_verse.get(v.key, [])[:2]):
                boundary = True
            elif current_entities and ents:
                overlap = len(current_entities & ents) / max(1, len(ents))
                if overlap < 1 - ENTITY_TURNOVER:
                    boundary = True
        if boundary and len(current) >= MIN_UNIT_VERSES:
            flush()
        current_pericope = pid
        current.append(v.key)
        current_entities |= ents
    flush()

    _induce_order(tl, struct)
    return tl


def _build_unit(uid, book, pid, keys, struct, corpus, events_by_verse,
                timex_by_verse, signals_by_verse, chains) -> EventUnit:
    event_ids, eligible, timexes, sig_preds, preds = [], [], [], set(), []
    timex_preds: List[str] = []
    anchorable = False
    classes, types_, tenses, aspects, pos_tags = set(), set(), set(), set(), set()
    modal_types = set()
    depths = []
    negs = 0
    for k in keys:
        for ev in events_by_verse.get(k, []):
            event_ids.append(ev.xml_id)
            preds.append(ev.pred)
            classes.add(str(ev.event_class))
            types_.add(str(ev.event_type))
            tenses.add(str(ev.tense))
            aspects.add(str(ev.aspect))
            pos_tags.add(str(ev.pos))
            if str(ev.polarity) == "NEG":
                negs += 1
            path = struct.modal_paths.get(ev.xml_id, [])
            depths.append(len(path))
            modal_types |= set(path)
            if ev.xml_id in struct.eligible:
                eligible.append(ev.xml_id)
        for tx in timex_by_verse.get(k, []):
            timexes.append(tx.xml_id)
            if tx.pred:
                timex_preds.append(tx.pred)
            if tx.anchorable:
                anchorable = True
        for sg in signals_by_verse.get(k, []):
            sig_preds.add(sg.pred)

    text = " ".join(corpus[book].get(c, v).text for _b, c, v in keys
                    if corpus[book].get(c, v))
    entities = set()
    for k in keys:
        entities |= chains.by_verse.get(k, set())

    # assertion profile: share of the unit's incident relations at each level
    prof = [0.0] * 5
    total = 0
    eset = set(event_ids)
    for l in struct.tlinks:
        if l.source in eset or l.target_id in eset:
            lv = min(max(l.level, 1), 5)
            prof[lv - 1] += 1
            total += 1
    if total:
        prof = [p / total for p in prof]

    return EventUnit(
        unit_id=uid, book=book, pericope_id=pid, verse_keys=list(keys),
        event_ids=event_ids, eligible_ids=eligible, timex_ids=timexes,
        timex_preds=timex_preds, anchorable_timex=anchorable,
        signal_preds=sig_preds, entities=entities, text=text, preds=preds,
        classes=classes, types=types_, tenses=tenses, aspects=aspects,
        pos_tags=pos_tags,
        neg_fraction=(negs / len(event_ids)) if event_ids else 0.0,
        modal_depth=(sum(depths) / len(depths)) if depths else 0.0,
        modal_types=modal_types,
        eligible_fraction=(len(eligible) / len(event_ids)) if event_ids else 0.0,
        level_profile=tuple(prof),
    )


def _induce_order(tl: LocalTimeline, struct: AnnotationStructure) -> None:
    """Project the closed network over eligible events onto the units."""
    net = struct.closed_network
    prov = struct.network_provenance
    ident = struct.identity_classes or {}

    rep_to_units: Dict[str, Set[str]] = {}
    for eid, uid in tl.unit_of_event.items():
        rep = ident.get(eid, eid)
        rep_to_units.setdefault(rep, set()).add(uid)

    for (i, j), rel in net.items():
        if not implies_before(frozenset(rel)):
            continue
        for ui in rep_to_units.get(i, ()):
            for uj in rep_to_units.get(j, ()):
                if ui == uj:
                    continue
                meta = prov.get((i, j), {})
                conf = meta.get("confidence", 0.35)
                lvl = meta.get("level", 5)
                key = (ui, uj)
                if key not in tl.order or tl.order_confidence.get(key, 0) < conf:
                    tl.order.add(key)
                    tl.order_confidence[key] = conf
                    tl.order_level[key] = lvl

    # document order is itself evidence of narrative order between units of the
    # same pericope; recorded at level-4 confidence so that any asserted
    # relation outranks it
    for a, b in zip(tl.units, tl.units[1:]):
        if a.pericope_id == b.pericope_id:
            key = (a.unit_id, b.unit_id)
            if key not in tl.order:
                tl.order.add(key)
                tl.order_confidence[key] = 0.35
                tl.order_level[key] = 4

    # the chain of narrative reference times is itself a partial ordering of the
    # pericopes (thesis Section 6.2.5, Level 2): each pericope's empty <TIMEX3>
    # is anchored to that of the preceding pericope, and that anchor chain is
    # contributed to the network here. It is recorded with its own provenance
    # so that Table tab:res-cascade can report it apart from the event-level
    # cascade.
    for a, b in zip(tl.units, tl.units[1:]):
        if a.pericope_id != b.pericope_id:
            key = (a.unit_id, b.unit_id)
            tl.order.add(key)
            tl.order_confidence[key] = max(tl.order_confidence.get(key, 0.0),
                                           0.70)
            tl.order_level[key] = 2
            tl.nrt_chain.add(key)


def segment_corpus(structs: Dict[str, AnnotationStructure], corpus: Corpus,
                   pericopes: PericopeLayer,
                   chains: Dict[str, EntityChains]) -> Dict[str, LocalTimeline]:
    return {book: segment(structs[book], corpus, pericopes, chains[book])
            for book in structs}
