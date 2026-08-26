"""
Stage 2 - serialisation (thesis Section 6.2.8).

Layer A is serialised as one .tml document per Gospel, with <isoTimeML> as the
root element (repairing schema defect S8) and every element carrying xml:id
(S7). References are emitted with the '#' prefix.

Components of the abstract syntax with no concrete realisation (tau, N, PN) and
the pipeline's own provenance are emitted in a private namespace, so the
document remains valid under the corrected schema while remaining sufficient to
reproduce the pipeline (thesis Section 5.8: no information exists only in the
JSON projection).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from ..stage1_preprocessing.segmentation import SegmentedDocument
from .enums import NEVER_EMITTED_TLINK, Polarity
from .model import (PRIVATE_NS, PRIVATE_PREFIX, AnnotationStructure, Event,
                    Signal, TLink, Timex3)

ISO_NS = "http://www.iso.org/ns/semaf-time"
ET.register_namespace("", ISO_NS)
ET.register_namespace(PRIVATE_PREFIX, PRIVATE_NS)

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _p(name: str) -> str:
    return f"{{{PRIVATE_NS}}}{name}"


def _refs(ids: List[str]) -> str:
    if len(ids) == 1:
        return "#" + ids[0]
    if len(ids) > 1:
        return f"#range({ids[0]},{ids[-1]})"
    return ""


def serialise(struct: AnnotationStructure, seg: SegmentedDocument,
              out_path: Path) -> Path:
    root = ET.Element(f"{{{ISO_NS}}}isoTimeML")
    root.set(_p("profile"), struct.profile)
    root.set(_p("projectionMode"), struct.projection_mode)
    root.set(_p("document"), struct.book)

    for ev in struct.events.values():
        el = ET.SubElement(root, "EVENT")
        el.set(XML_ID, ev.xml_id)
        if ev.target:
            el.set("target", _refs(ev.target))
        el.set("pred", ev.pred)
        el.set("class", str(ev.event_class))
        el.set("type", str(ev.event_type))
        el.set("tense", str(ev.tense))
        el.set("aspect", str(ev.aspect))
        if str(ev.vform) != "NONE":
            el.set("vform", str(ev.vform))
        el.set("pos", str(ev.pos))
        if ev.polarity == Polarity.NEG:
            el.set("polarity", "NEG")
        if ev.modality:
            el.set("modality", ev.modality)
        # abstract-syntax components without concrete realisation
        if ev.tau:
            el.set(_p("tau"), ev.tau)
        if ev.iteration_count is not None:
            el.set(_p("N"), str(ev.iteration_count))
        if ev.distribution_period:
            el.set(_p("PN"), ev.distribution_period)
        # provenance: keyed on book:chapter:verse (stand-off distribution)
        if ev.verse_key:
            el.set(_p("verse"), f"{ev.verse_key[0]}:{ev.verse_key[1]}:"
                                f"{ev.verse_key[2]}")
        if ev.pericope_id:
            el.set(_p("pericope"), ev.pericope_id)
        if ev.sent_id:
            el.set(_p("sentence"), ev.sent_id)
        el.set(_p("eligible"), "true" if ev.xml_id in struct.eligible
               else "false")
        path = struct.modal_paths.get(ev.xml_id) or []
        if path:
            el.set(_p("modalPath"), " ".join(path))

    for tx in struct.timexes.values():
        el = ET.SubElement(root, "TIMEX3")
        el.set(XML_ID, tx.xml_id)
        if tx.target:
            el.set("target", _refs(tx.target))
        el.set("type", str(tx.timex_type))
        if tx.value is not None:
            el.set("value", tx.value)
        el.set("temporalFunction", "true" if tx.temporal_function else "false")
        if str(tx.function_in_document) != "NONE":
            el.set("functionInDocument", str(tx.function_in_document))
        if tx.anchor_time_id:
            el.set("anchorTimeID", "#" + tx.anchor_time_id)
        if tx.begin_point:
            el.set("beginPoint", "#" + tx.begin_point)
        if tx.end_point:
            el.set("endPoint", "#" + tx.end_point)
        if tx.quant:
            el.set("quant", tx.quant)
        if tx.freq:
            el.set("freq", tx.freq)
        if tx.mod:
            el.set("mod", str(tx.mod))
        if tx.pred:
            el.set("pred", tx.pred)
        if tx.verse_key:
            el.set(_p("verse"), f"{tx.verse_key[0]}:{tx.verse_key[1]}:"
                                f"{tx.verse_key[2]}")
        if tx.anchor_level:
            el.set(_p("anchorLevel"), str(tx.anchor_level))
        if tx.anchorable:
            el.set(_p("anchorable"), "true")
        if tx.comment:
            el.set(_p("comment"), tx.comment)

    for sg in struct.signals.values():
        el = ET.SubElement(root, "SIGNAL")
        el.set(XML_ID, sg.xml_id)
        el.set("target", _refs(sg.target))
        el.set("pred", sg.pred)
        if sg.verse_key:
            el.set(_p("verse"), f"{sg.verse_key[0]}:{sg.verse_key[1]}:"
                                f"{sg.verse_key[2]}")

    for l in struct.tlinks:
        if l.rel_type in NEVER_EMITTED_TLINK:
            continue
        el = ET.SubElement(root, "TLINK")
        el.set(XML_ID, l.xml_id)
        if l.event_id:
            el.set("eventID", "#" + l.event_id)
        if l.time_id:
            el.set("timeID", "#" + l.time_id)
        if l.related_to_event:
            el.set("relatedToEvent", "#" + l.related_to_event)
        if l.related_to_time:
            el.set("relatedToTime", "#" + l.related_to_time)
        if l.signal_id:
            el.set("signalID", "#" + l.signal_id)
        el.set("relType", str(l.rel_type))
        el.set(_p("origin"), l.origin)
        el.set(_p("level"), str(l.level))
        el.set(_p("confidence"), f"{l.confidence:.2f}")

    for sl in struct.slinks:
        el = ET.SubElement(root, "SLINK")
        el.set(XML_ID, sl.xml_id)
        el.set("eventID", "#" + sl.event_id)
        el.set("subordinatedEvent", "#" + sl.subordinated_event)
        if sl.signal_id:
            el.set("signalID", "#" + sl.signal_id)
        el.set("relType", str(sl.rel_type))

    for al in struct.alinks:
        el = ET.SubElement(root, "ALINK")
        el.set(XML_ID, al.xml_id)
        el.set("eventID", "#" + al.event_id)
        el.set("relatedToEvent", "#" + al.related_to_event)
        if al.signal_id:
            el.set("signalID", "#" + al.signal_id)
        el.set("relType", str(al.rel_type))

    for ml in struct.mlinks:
        el = ET.SubElement(root, "MLINK")
        el.set(XML_ID, ml.xml_id)
        el.set("eventID", "#" + ml.event_id)
        el.set("relatedToTime", "#" + ml.related_to_time)
        if ml.signal_id:
            el.set("signalID", "#" + ml.signal_id)
        el.set("relType", str(ml.rel_type))

    for cf in struct.confidences:
        el = ET.SubElement(root, "CONFIDENCE")
        el.set(XML_ID, cf.xml_id)
        el.set("target", "#" + cf.target)
        el.set("value", f"{cf.value:.2f}")
        el.set("annotator", cf.annotator)

    _indent(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8",
                               xml_declaration=True)
    return out_path


def serialise_token_layer(seg: SegmentedDocument, out_path: Path) -> Path:
    """The base segmentation layer, so the stand-off annotation can be resolved
    by a recipient holding a licensed copy of the source text."""
    root = ET.Element("tokenLayer", {"document": seg.book})
    for sent in seg.sentences:
        s = ET.SubElement(root, "s", {XML_ID: sent.sent_id,
                                      "verse": f"{sent.verse_key[0]}:"
                                               f"{sent.verse_key[1]}:"
                                               f"{sent.verse_key[2]}"})
        for t in sent.tokens:
            ET.SubElement(s, "tk", {XML_ID: t.xml_id, "pos": t.pos,
                                    "lemma": t.lemma})
    _indent(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8",
                               xml_declaration=True)
    return out_path


def json_projection(struct: AnnotationStructure) -> dict:
    """The JSON projection of Table tab:fw-contracts.

    Derived, never primary: it carries pre-computed modal context paths and the
    transitive closure so downstream stages need not recompute them.
    """
    return {
        "document": struct.book,
        "profile": struct.profile,
        "projection_mode": struct.projection_mode,
        "events": {
            eid: {
                "pred": e.pred, "class": str(e.event_class),
                "type": str(e.event_type), "tense": str(e.tense),
                "aspect": str(e.aspect), "pos": str(e.pos),
                "polarity": str(e.polarity),
                "verse": list(e.verse_key) if e.verse_key else None,
                "pericope": e.pericope_id, "text": e.text,
                "modal_path": struct.modal_paths.get(eid, []),
                "eligible": eid in struct.eligible,
            } for eid, e in struct.events.items()
        },
        "timexes": {
            tid: {
                "type": str(t.timex_type), "value": t.value,
                "temporalFunction": t.temporal_function,
                "anchorTimeID": t.anchor_time_id, "mod": str(t.mod)
                if t.mod else None, "pred": t.pred,
                "anchor_level": t.anchor_level, "anchorable": t.anchorable,
                "verse": list(t.verse_key) if t.verse_key else None,
                "text": t.text,
            } for tid, t in struct.timexes.items()
        },
        "signals": {s.xml_id: {"pred": s.pred, "text": s.text}
                    for s in struct.signals.values()},
        "tlinks": [
            {"id": l.xml_id, "source": l.source, "target": l.target_id,
             "relType": str(l.rel_type), "origin": l.origin,
             "level": l.level, "confidence": l.confidence,
             "signal": l.signal_id}
            for l in struct.tlinks
        ],
        "slinks": [{"id": s.xml_id, "gov": s.event_id,
                    "sub": s.subordinated_event, "relType": str(s.rel_type)}
                   for s in struct.slinks],
        "alinks": [{"id": a.xml_id, "gov": a.event_id,
                    "arg": a.related_to_event, "relType": str(a.rel_type)}
                   for a in struct.alinks],
        "mlinks": [{"id": m.xml_id, "event": m.event_id,
                    "time": m.related_to_time} for m in struct.mlinks],
        "closed_network": [
            {"i": i, "j": j, "allen": sorted(r),
             **(struct.network_provenance.get((i, j), {}))}
            for (i, j), r in struct.closed_network.items()
        ],
        "conflicts": struct.conflicts,
        "identity_classes": struct.identity_classes,
    }


def write_json_projection(struct: AnnotationStructure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(json_projection(struct), fh, indent=1)
    return out_path


def _indent(elem, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (elem.tail or "").strip():
            elem.tail = pad
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad
