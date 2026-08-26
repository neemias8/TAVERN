"""
<MLINK> inference (thesis Section 6.2.7).

Measure links come from durations governed by a preposition of extent
("for three hours"). <MLINK> is the only conformant means of recording that a
duration MEASURES an event rather than standing in a temporal relation to it.

@relType is emitted always with the value MEASURES; absence is interpreted as
MEASURES on input, since 7.3.5.4 describes the relation as inherent to the
element (conformance resolution R7). @relatedToTime is admitted and used on
emission, repairing schema defect S6.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ...stage1_preprocessing.segmentation import SegmentedDocument, Token
from ..enums import MLinkRel, TimexType
from ..model import AnnotationStructure, Event, MLink, Timex3

#: Prepositions of extent.
EXTENT_PREDS = {"FOR", "THROUGHOUT", "DURING", "OVER", "WITHIN"}
EXTENT_SURFACES = {"for", "throughout", "over", "during", "within", "in"}


class MLinkInferrer:
    def infer(self, doc: SegmentedDocument, struct: AnnotationStructure) -> None:
        for sent in doc.sentences:
            toks = sent.tokens
            idx = {t.xml_id: i for i, t in enumerate(toks)}
            durations = [t for t in struct.timexes.values()
                         if t.sent_id == sent.sent_id
                         and t.timex_type == TimexType.DURATION and t.target]
            if not durations:
                continue
            events = [e for e in struct.events.values()
                      if e.sent_id == sent.sent_id and e.head_token in idx]
            heads = {idx[e.head_token]: e for e in events}
            for tx in durations:
                tpos = [idx[k] for k in tx.target if k in idx]
                if not tpos:
                    continue
                first = min(tpos)
                gov_prep = None
                for back in range(1, 4):
                    p = first - back
                    if p < 0:
                        break
                    if toks[p].text.lower() in EXTENT_SURFACES:
                        gov_prep = p
                        break
                if gov_prep is None:
                    continue
                ev = self._governing_event(toks, heads, gov_prep)
                if ev is None:
                    continue
                sig = next((s.xml_id for s in struct.signals.values()
                            if s.sent_id == sent.sent_id and s.target
                            and idx.get(s.target[0]) == gov_prep), None)
                struct.mlinks.append(MLink(
                    xml_id=struct.next_id("ml"),
                    event_id=ev.xml_id,
                    related_to_time=tx.xml_id,
                    rel_type=MLinkRel.MEASURES,
                    signal_id=sig,
                ))

    @staticmethod
    def _governing_event(toks: Sequence[Token], heads: Dict[int, Event],
                         start: int) -> Optional[Event]:
        cur = start
        for _ in range(6):
            if cur in heads:
                return heads[cur]
            nxt = toks[cur].head
            if nxt == cur or nxt < 0 or nxt >= len(toks):
                return None
            cur = nxt
        return None
