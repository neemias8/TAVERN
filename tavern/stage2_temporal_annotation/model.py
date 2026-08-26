"""
Stage 2 - element dataclasses and the AnnotationStructure (M, L).

One dataclass per element type, plus the pair (M, L) of the abstract syntax
(thesis Section 2.x, sec:bg-abstract). The abstract syntax's components tau
(set-theoretic type), N (iteration count) and PN (distribution period) have no
realisation in the concrete XML syntax; they are carried here because the
reasoning layer needs iteration counts to interpret expressions such as
"before the rooster crows twice", and are serialised through a
private-namespace attribute so the document remains schema-valid
(thesis Section 6.2.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .enums import (ALinkRel, Aspect, CONFIDENCE_OF_LEVEL, EventClass,
                    EventType, FunctionInDocument, MLinkRel, Mod, POS,
                    Polarity, SLinkRel, TLinkRel, Tense, TimexType, VForm)

PRIVATE_NS = "http://tavern.unisinos.br/ns/isotimeml-ext"
PRIVATE_PREFIX = "tvn"


def _strip(ref: Optional[str]) -> Optional[str]:
    """References are emitted with '#' and accepted with or without it."""
    if ref is None:
        return None
    return ref[1:] if ref.startswith("#") else ref


def _hash(ref: Optional[str]) -> Optional[str]:
    if ref is None:
        return None
    return ref if ref.startswith("#") else "#" + ref


# ---------------------------------------------------------------------------
# Markables
# ---------------------------------------------------------------------------

@dataclass
class Event:
    xml_id: str
    target: List[str]                      # token xml:ids (empty = non-consuming)
    pred: str
    event_class: EventClass
    event_type: EventType
    tense: Tense = Tense.NONE
    aspect: Aspect = Aspect.NONE
    vform: VForm = VForm.NONE
    pos: POS = POS.OTHER
    polarity: Polarity = Polarity.POS
    modality: Optional[str] = None
    comment: Optional[str] = None

    # abstract-syntax components without a concrete realisation
    tau: Optional[str] = None              # set-theoretic type
    iteration_count: Optional[int] = None  # N
    distribution_period: Optional[str] = None  # PN

    # provenance (not part of Layer A semantics; serialised in private ns)
    verse_key: Optional[Tuple[str, int, int]] = None
    verse_keys: List[Tuple[str, int, int]] = field(default_factory=list)
    sent_id: Optional[str] = None
    pericope_id: Optional[str] = None
    text: str = ""
    head_token: Optional[str] = None

    @property
    def ref(self) -> str:
        return "#" + self.xml_id

    @property
    def book(self) -> Optional[str]:
        return self.verse_key[0] if self.verse_key else None


@dataclass
class Timex3:
    xml_id: str
    timex_type: TimexType
    target: List[str] = field(default_factory=list)
    value: Optional[str] = None
    temporal_function: bool = False
    function_in_document: FunctionInDocument = FunctionInDocument.NONE
    anchor_time_id: Optional[str] = None
    begin_point: Optional[str] = None
    end_point: Optional[str] = None
    quant: Optional[str] = None
    freq: Optional[str] = None
    mod: Optional[Mod] = None
    pred: Optional[str] = None
    comment: Optional[str] = None

    verse_key: Optional[Tuple[str, int, int]] = None
    sent_id: Optional[str] = None
    pericope_id: Optional[str] = None
    text: str = ""
    # anchoring hierarchy level: 1 document, 2 narrative reference, 3 utterance,
    # 0 for expressions realised in the text
    anchor_level: int = 0
    anchorable: bool = False
    scaffold_rank: Optional[float] = None

    @property
    def ref(self) -> str:
        return "#" + self.xml_id

    @property
    def is_empty(self) -> bool:
        return not self.target


@dataclass
class Signal:
    xml_id: str
    target: List[str]
    pred: str
    verse_key: Optional[Tuple[str, int, int]] = None
    sent_id: Optional[str] = None
    text: str = ""

    @property
    def ref(self) -> str:
        return "#" + self.xml_id


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

@dataclass
class TLink:
    xml_id: str
    rel_type: TLinkRel
    event_id: Optional[str] = None          # exactly one of event_id / time_id
    time_id: Optional[str] = None
    related_to_event: Optional[str] = None
    related_to_time: Optional[str] = None
    signal_id: Optional[str] = None
    origin: str = "asserted"                # 'asserted' | 'closure'
    level: int = 4
    confidence: float = 0.35
    comment: Optional[str] = None

    def __post_init__(self):
        self.event_id = _strip(self.event_id)
        self.time_id = _strip(self.time_id)
        self.related_to_event = _strip(self.related_to_event)
        self.related_to_time = _strip(self.related_to_time)
        self.signal_id = _strip(self.signal_id)

    @property
    def source(self) -> Optional[str]:
        return self.event_id or self.time_id

    @property
    def target_id(self) -> Optional[str]:
        return self.related_to_event or self.related_to_time


@dataclass
class SLink:
    xml_id: str
    rel_type: SLinkRel
    event_id: str
    subordinated_event: str
    signal_id: Optional[str] = None
    comment: Optional[str] = None

    def __post_init__(self):
        self.event_id = _strip(self.event_id)
        self.subordinated_event = _strip(self.subordinated_event)
        self.signal_id = _strip(self.signal_id)


@dataclass
class ALink:
    xml_id: str
    rel_type: ALinkRel
    event_id: str
    related_to_event: str
    signal_id: Optional[str] = None
    comment: Optional[str] = None

    def __post_init__(self):
        self.event_id = _strip(self.event_id)
        self.related_to_event = _strip(self.related_to_event)
        self.signal_id = _strip(self.signal_id)


@dataclass
class MLink:
    xml_id: str
    event_id: str
    related_to_time: str
    rel_type: MLinkRel = MLinkRel.MEASURES
    signal_id: Optional[str] = None

    def __post_init__(self):
        self.event_id = _strip(self.event_id)
        self.related_to_time = _strip(self.related_to_time)
        self.signal_id = _strip(self.signal_id)


@dataclass
class Confidence:
    xml_id: str
    target: str            # xml:id of the annotated element
    value: float
    annotator: str = "TAVERN"

    def __post_init__(self):
        self.target = _strip(self.target)


# ---------------------------------------------------------------------------
# The pair (M, L)
# ---------------------------------------------------------------------------

@dataclass
class AnnotationStructure:
    """The pair (M, L) of the abstract syntax: markables and links."""

    book: str
    events: Dict[str, Event] = field(default_factory=dict)
    timexes: Dict[str, Timex3] = field(default_factory=dict)
    signals: Dict[str, Signal] = field(default_factory=dict)
    tlinks: List[TLink] = field(default_factory=list)
    slinks: List[SLink] = field(default_factory=list)
    alinks: List[ALink] = field(default_factory=list)
    mlinks: List[MLink] = field(default_factory=list)
    confidences: List[Confidence] = field(default_factory=list)

    profile: str = "semaf-time-biblical"
    projection_mode: str = "relative"

    # Layer B products, filled by veridicality.py / closure.py
    modal_paths: Dict[str, List[str]] = field(default_factory=dict)
    eligible: Set[str] = field(default_factory=set)
    closed_network: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    network_provenance: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    conflicts: List[dict] = field(default_factory=list)
    identity_classes: Dict[str, str] = field(default_factory=dict)

    _counters: Dict[str, int] = field(default_factory=dict, repr=False)

    # -- id allocation ----------------------------------------------------
    def next_id(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}{n}"

    # -- mutation ---------------------------------------------------------
    def add_event(self, e: Event) -> Event:
        self.events[e.xml_id] = e
        return e

    def add_timex(self, t: Timex3) -> Timex3:
        self.timexes[t.xml_id] = t
        return t

    def add_signal(self, s: Signal) -> Signal:
        self.signals[s.xml_id] = s
        return s

    def add_tlink(self, l: TLink) -> TLink:
        self.tlinks.append(l)
        return l

    # -- queries ----------------------------------------------------------
    def event_list(self) -> List[Event]:
        return list(self.events.values())

    def events_in_sentence(self, sent_id: str) -> List[Event]:
        return [e for e in self.events.values() if e.sent_id == sent_id]

    def events_in_pericope(self, pid: str) -> List[Event]:
        return [e for e in self.events.values() if e.pericope_id == pid]

    def eligible_events(self) -> List[Event]:
        return [e for e in self.events.values() if e.xml_id in self.eligible]

    def asserted_tlinks(self) -> List[TLink]:
        return [l for l in self.tlinks if l.origin == "asserted"]

    def derived_tlinks(self) -> List[TLink]:
        return [l for l in self.tlinks if l.origin == "closure"]

    def element(self, xml_id: str):
        return (self.events.get(xml_id) or self.timexes.get(xml_id)
                or self.signals.get(xml_id))

    def counts(self) -> Dict[str, int]:
        verbal = sum(1 for e in self.events.values() if e.pos == POS.VERB)
        nominal = sum(1 for e in self.events.values() if e.pos == POS.NOUN)
        states = sum(1 for e in self.events.values()
                     if e.pos in (POS.ADJECTIVE, POS.OTHER, POS.PREP))
        return {
            "event_verbal": verbal,
            "event_nominal": nominal,
            "event_state": states,
            "event_total": len(self.events),
            "timex": sum(1 for t in self.timexes.values() if not t.is_empty),
            "timex_empty": sum(1 for t in self.timexes.values() if t.is_empty),
            "signal": len(self.signals),
            "tlink_asserted": len(self.asserted_tlinks()),
            "tlink_derived": len(self.derived_tlinks()),
            "slink": len(self.slinks),
            "alink": len(self.alinks),
            "mlink": len(self.mlinks),
            "eligible": len(self.eligible),
            "subordinated": len(self.events) - len(self.eligible),
        }
