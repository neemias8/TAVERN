"""
Stage 2 - temporal annotation with ISO 24617-1:2012.

Layer A produces a conformant stand-off .tml instance per document; Layer B
derives from Layer A alone the structures the pipeline requires. Nothing in
Layer B consults the text again.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from ..config import TavernConfig
from ..stage1_preprocessing.pericopes import PericopeLayer
from ..stage1_preprocessing.segmentation import SegmentedDocument
from . import closure as closure_mod
from . import veridicality
from .event_tagger import EventTagger
from .link_inference import (ALinkInferrer, MLinkInferrer, SLinkInferrer,
                             TLinkInferrer)
from .model import AnnotationStructure
from .serializer import (serialise, serialise_token_layer,
                         write_json_projection)
from .signal_tagger import SignalTagger
from .timex_normalizer import TimexNormalizer
from .timex_tagger import TimexTagger
from .validator import ValidationReport, enforce_accessibility, validate

__all__ = ["annotate_document", "annotate_corpus", "AnnotationStructure"]


def annotate_document(seg: SegmentedDocument, pericopes: PericopeLayer,
                      cfg: TavernConfig) -> Tuple[AnnotationStructure,
                                                  ValidationReport]:
    struct = AnnotationStructure(book=seg.book,
                                 projection_mode=cfg.projection_mode)

    # --- Layer A ---------------------------------------------------------
    EventTagger(pericopes).tag(seg, struct)
    TimexTagger(pericopes).tag(seg, struct)
    SignalTagger().tag(seg, struct)
    TimexNormalizer(pericopes, cfg.projection_mode,
                    cfg.absolute_year).normalise(struct)

    SLinkInferrer().infer(seg, struct)
    _reclassify_unsubordinating_reporters(struct)
    ALinkInferrer().infer(seg, struct)
    MLinkInferrer().infer(seg, struct)

    # the veridicality partition must precede level 4 of the cascade, since
    # level 4 applies only to narrative-world events
    veridicality.partition(struct, enabled=cfg.use_veridicality)
    TLinkInferrer(cfg.cascade_levels).infer(seg, struct)

    # --- Layer B ---------------------------------------------------------
    veridicality.partition(struct, enabled=cfg.use_veridicality)
    result = closure_mod.close(struct, enabled=cfg.use_closure)
    closure_mod.apply_to_struct(struct, result)

    enforce_accessibility(struct)
    report = validate(struct)
    return struct, report


def _reclassify_unsubordinating_reporters(struct: AnnotationStructure) -> int:
    """A REPORTING or PERCEPTION event that governs no <SLINK> is demoted to
    OCCURRENCE.

    A.3.3.1.2 requires one <SLINK> for each reporting or perception event, and
    an <SLINK> requires a subordinated *event*. Where the complement of a
    reporting verb contains no event-denoting expression -- 'he answered them',
    'they saw the fig tree' -- the class assignment rests on the lexical test
    alone, which Section 6.2.2 already declares insufficient for the classes
    defined by argument structure. The same reasoning is applied here, so that
    constraint 7 of Appendix B is satisfied by the annotation rather than
    excused.
    """
    from .enums import EventClass, EventType
    governors = {sl.event_id for sl in struct.slinks}
    demoted = 0
    for ev in struct.events.values():
        if ev.event_class in (EventClass.REPORTING, EventClass.PERCEPTION) \
                and ev.xml_id not in governors:
            ev.event_class = EventClass.OCCURRENCE
            ev.event_type = EventType.TRANSITION
            ev.comment = ("demoted from REPORTING/PERCEPTION: no subordinated "
                          "event (A.3.3.1.2)")
            demoted += 1
    return demoted


def annotate_corpus(segments: Dict[str, SegmentedDocument],
                    pericopes: PericopeLayer, cfg: TavernConfig,
                    write: bool = True):
    structs: Dict[str, AnnotationStructure] = {}
    reports: Dict[str, ValidationReport] = {}
    out = cfg.run_dir() / "annotation"
    for book, seg in segments.items():
        struct, rep = annotate_document(seg, pericopes, cfg)
        structs[book] = struct
        reports[book] = rep
        if write:
            serialise(struct, seg, out / f"{book}.tml")
            serialise_token_layer(seg, out / f"{book}.tokens.xml")
            write_json_projection(struct, out / f"{book}.json")
    return structs, reports
