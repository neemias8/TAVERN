"""
Stage 2 - reader for conformant .tml documents.

Lenient on input, per the governing principle of Appendix B, Section B.1:
accepts @id as well as xml:id (defect S7), references with or without the '#'
prefix, the nominal <ALINK> relation types of A.3.4.1, the abbreviated
@aspect spellings, PARTICIPLE for PASTPART, DURING_INV, and <CERTAINTY>
(preserved, never emitted).

This module exists so that the published annotation is sufficient to reproduce
the pipeline: Stage 3 can be re-executed from the .tml documents alone.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from .enums import (ALinkRel, Aspect, EventClass, EventType,
                    FunctionInDocument, MLinkRel, Mod, POS, Polarity,
                    SLinkRel, TLinkRel, Tense, TimexType, normalise_alink,
                    normalise_aspect, normalise_vform)
from .model import (PRIVATE_NS, ALink, AnnotationStructure, Confidence, Event,
                    MLink, SLink, Signal, TLink, Timex3)

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _id(el) -> str:
    return el.get(XML_ID) or el.get("id") or ""


def _ref(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    v = v.strip()
    if v.startswith("#range("):
        inner = v[7:-1]
        return inner.split(",")[0].strip()
    return v[1:] if v.startswith("#") else v


def _targets(v: Optional[str]) -> List[str]:
    if not v:
        return []
    v = v.strip()
    if v.startswith("#range("):
        inner = v[7:-1]
        return [p.strip().lstrip("#") for p in inner.split(",")]
    return [p.strip().lstrip("#") for p in v.split() if p.strip()]


def _priv(el, name: str) -> Optional[str]:
    return el.get(f"{{{PRIVATE_NS}}}{name}")


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def read(path: Path) -> AnnotationStructure:
    root = ET.parse(path).getroot()
    book = _priv(root, "document") or Path(path).stem
    struct = AnnotationStructure(book=book)
    struct.profile = _priv(root, "profile") or struct.profile
    struct.projection_mode = _priv(root, "projectionMode") or "relative"

    for el in root:
        tag = _local(el.tag)
        if tag == "EVENT":
            _read_event(el, struct)
        elif tag == "TIMEX3":
            _read_timex(el, struct)
        elif tag == "SIGNAL":
            struct.add_signal(Signal(
                xml_id=_id(el), target=_targets(el.get("target")),
                pred=el.get("pred") or "",
                verse_key=_verse(el),
            ))
        elif tag == "TLINK":
            _read_tlink(el, struct)
        elif tag == "SLINK":
            struct.slinks.append(SLink(
                xml_id=_id(el), rel_type=SLinkRel(el.get("relType")),
                event_id=_ref(el.get("eventID")),
                subordinated_event=_ref(el.get("subordinatedEvent")),
                signal_id=_ref(el.get("signalID")),
            ))
        elif tag == "ALINK":
            rel = normalise_alink(el.get("relType"))
            if rel is None:
                continue
            struct.alinks.append(ALink(
                xml_id=_id(el), rel_type=rel,
                event_id=_ref(el.get("eventID")),
                related_to_event=_ref(el.get("relatedToEvent")),
                signal_id=_ref(el.get("signalID")),
            ))
        elif tag == "MLINK":
            struct.mlinks.append(MLink(
                xml_id=_id(el),
                event_id=_ref(el.get("eventID")),
                related_to_time=_ref(el.get("relatedToTime")
                                     or el.get("relatedToEvent")),
                rel_type=MLinkRel.MEASURES,     # absence means MEASURES (R7)
                signal_id=_ref(el.get("signalID")),
            ))
        elif tag == "CONFIDENCE":
            struct.confidences.append(Confidence(
                xml_id=_id(el), target=_ref(el.get("target")) or "",
                value=float(el.get("value") or 1.0),
                annotator=el.get("annotator") or "",
            ))
        elif tag == "CERTAINTY":
            # defect S3: accepted and preserved, never emitted
            struct.__dict__.setdefault("_certainty", []).append(dict(el.attrib))

    # rebuild Layer B state carried in the private namespace
    for eid, ev in struct.events.items():
        pass
    return struct


def _verse(el):
    v = _priv(el, "verse")
    if not v:
        return None
    parts = v.split(":")
    if len(parts) != 3:
        return None
    return (parts[0], int(parts[1]), int(parts[2]))


def _read_event(el, struct: AnnotationStructure) -> None:
    eid = _id(el)
    ev = Event(
        xml_id=eid,
        target=_targets(el.get("target")),
        pred=el.get("pred") or "",
        event_class=EventClass(el.get("class") or "OCCURRENCE"),
        event_type=EventType(el.get("type") or "TRANSITION"),
        tense=Tense(el.get("tense") or "NONE"),
        aspect=normalise_aspect(el.get("aspect")),
        vform=normalise_vform(el.get("vform")),
        pos=POS(el.get("pos") or "OTHER"),
        polarity=Polarity(el.get("polarity") or "POS"),
        modality=el.get("modality"),
        verse_key=_verse(el),
        pericope_id=_priv(el, "pericope"),
        sent_id=_priv(el, "sentence"),
        tau=_priv(el, "tau"),
        iteration_count=int(_priv(el, "N")) if _priv(el, "N") else None,
        distribution_period=_priv(el, "PN"),
    )
    if ev.verse_key:
        ev.verse_keys = [ev.verse_key]
    struct.add_event(ev)
    path = _priv(el, "modalPath")
    struct.modal_paths[eid] = path.split() if path else []
    if (_priv(el, "eligible") or "true") == "true":
        struct.eligible.add(eid)


def _read_timex(el, struct: AnnotationStructure) -> None:
    mod = el.get("mod")
    struct.add_timex(Timex3(
        xml_id=_id(el),
        timex_type=TimexType(el.get("type") or "DATE"),
        target=_targets(el.get("target")),
        value=el.get("value"),
        temporal_function=(el.get("temporalFunction") or "false") == "true",
        function_in_document=FunctionInDocument(
            el.get("functionInDocument") or "NONE"),
        anchor_time_id=_ref(el.get("anchorTimeID")),
        begin_point=_ref(el.get("beginPoint")),
        end_point=_ref(el.get("endPoint")),
        quant=el.get("quant"), freq=el.get("freq"),
        mod=Mod(mod) if mod else None,
        pred=el.get("pred"),
        verse_key=_verse(el),
        anchor_level=int(_priv(el, "anchorLevel") or 0),
        anchorable=(_priv(el, "anchorable") or "false") == "true",
    ))


def _read_tlink(el, struct: AnnotationStructure) -> None:
    rel = el.get("relType") or "BEFORE"
    struct.add_tlink(TLink(
        xml_id=_id(el),
        rel_type=TLinkRel(rel),
        event_id=_ref(el.get("eventID")),
        time_id=_ref(el.get("timeID")),
        related_to_event=_ref(el.get("relatedToEvent")),
        related_to_time=_ref(el.get("relatedToTime")),
        signal_id=_ref(el.get("signalID")),
        origin=_priv(el, "origin") or "asserted",
        level=int(_priv(el, "level") or 4),
        confidence=float(_priv(el, "confidence") or 0.35),
    ))
