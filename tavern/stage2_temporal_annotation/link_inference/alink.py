"""
<ALINK> inference (thesis Section 6.2.7).

Aspectual links are inferred from the five subclasses of the ASPECTUAL event
class. Verbal relation types of H.3.1 are emitted (conformance resolution R11);
REINITIATES is emitted even though A.3.4.1 omits it, because the ASPECTUAL
class does define a reinitiation subclass (Appendix B, Section B.5).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ...stage1_preprocessing.segmentation import SegmentedDocument, Token
from ..enums import ALinkRel, EventClass, normalise_alink
from ..event_tagger import ASPECTUAL_LEMMAS
from ..model import ALink, AnnotationStructure, Event

_COMPLEMENT_DEPS = {"xcomp", "ccomp", "acl", "pcomp", "oprd", "dobj", "obj"}


class ALinkInferrer:
    def infer(self, doc: SegmentedDocument, struct: AnnotationStructure) -> None:
        by_sent: Dict[str, List[Event]] = {}
        for ev in struct.events.values():
            by_sent.setdefault(ev.sent_id or "", []).append(ev)

        for sent in doc.sentences:
            events = by_sent.get(sent.sent_id, [])
            aspectual = [e for e in events
                         if e.event_class == EventClass.ASPECTUAL]
            if not aspectual:
                continue
            toks = sent.tokens
            idx = {t.xml_id: i for i, t in enumerate(toks)}
            heads = {idx[e.head_token]: e for e in events
                     if e.head_token in idx}
            for gov in aspectual:
                gi = idx.get(gov.head_token)
                if gi is None:
                    continue
                rel = normalise_alink(ASPECTUAL_LEMMAS.get(gov.pred.lower()))
                if rel is None:
                    continue
                arg = self._argument(toks, heads, gi, gov)
                if arg is None:
                    continue
                struct.alinks.append(ALink(
                    xml_id=struct.next_id("al"),
                    rel_type=rel,
                    event_id=gov.xml_id,
                    related_to_event=arg.xml_id,
                ))

    @staticmethod
    def _argument(toks: Sequence[Token], heads: Dict[int, Event], gi: int,
                  gov: Event) -> Optional[Event]:
        for i, t in enumerate(toks):
            if t.head == gi and t.dep in _COMPLEMENT_DEPS and i in heads:
                cand = heads[i]
                if cand.xml_id != gov.xml_id:
                    return cand
        # nominal argument: 'the beginning of the sorrows'
        for i, t in enumerate(toks):
            if t.head == gi and t.dep in ("prep", "pobj") and i in heads:
                cand = heads[i]
                if cand.xml_id != gov.xml_id:
                    return cand
        return None
