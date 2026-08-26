"""
<SLINK> inference (thesis Section 6.2.7).

Subordination links are inferred from event class, following the standard's
normative interaction rules (A.3.3.1.2):

  * a REPORTING or PERCEPTION event yields an EVIDENTIAL or NEG_EVIDENTIAL
    link to each event in its complement, with polarity determining which;
  * an I_STATE yields INTENSIONAL, FACTIVE or COUNTER_FACTIVE according to the
    lexical class of its predicate;
  * conditional constructions yield CONDITIONAL.

The standard's requirement that one <SLINK> be introduced for EACH REPORTING or
PERCEPTION event is what makes this layer dense in this corpus, and is the
reason the veridicality partition is available at all.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from ...stage1_preprocessing.segmentation import SegmentedDocument, Token
from ..enums import EventClass, Polarity, SLinkRel
from ..event_tagger import (COUNTER_FACTIVE_LEMMAS, FACTIVE_LEMMAS)
from ..model import AnnotationStructure, Event, SLink

_COMPLEMENT_DEPS = {"ccomp", "xcomp", "acl", "advcl", "parataxis", "conj",
                    "oprd", "pcomp"}
_CONDITIONAL_MARKERS = {"if", "unless", "whether", "lest", "suppose"}


class SLinkInferrer:
    def infer(self, doc: SegmentedDocument, struct: AnnotationStructure) -> None:
        by_sent: Dict[str, List[Event]] = {}
        for ev in struct.events.values():
            by_sent.setdefault(ev.sent_id or "", []).append(ev)

        for sent in doc.sentences:
            events = by_sent.get(sent.sent_id, [])
            if not events:
                continue
            self._infer_sentence(sent, events, struct)

        # A REPORTING/PERCEPTION event with no complement in its own sentence
        # still governs the direct speech that follows it in the same verse.
        self._link_following_speech(doc, struct, by_sent)

    # -- per sentence -----------------------------------------------------
    def _infer_sentence(self, sent, events: List[Event],
                        struct: AnnotationStructure) -> None:
        toks = sent.tokens
        idx_of = {t.xml_id: i for i, t in enumerate(toks)}
        head_of: Dict[str, int] = {}
        for ev in events:
            if ev.head_token in idx_of:
                head_of[ev.xml_id] = idx_of[ev.head_token]

        for gov in events:
            gi = head_of.get(gov.xml_id)
            if gi is None:
                continue
            for sub in events:
                if sub is gov:
                    continue
                si = head_of.get(sub.xml_id)
                if si is None:
                    continue
                if not self._is_complement(toks, si, gi):
                    continue
                rel = self._relation(gov, sub, toks, si)
                if rel is None:
                    continue
                struct.slinks.append(SLink(
                    xml_id=struct.next_id("sl"),
                    rel_type=rel,
                    event_id=gov.xml_id,
                    subordinated_event=sub.xml_id,
                ))

        # conditional constructions: the marker's clause is subordinated
        for i, tok in enumerate(toks):
            if tok.lemma not in _CONDITIONAL_MARKERS or tok.dep != "mark":
                continue
            clause_head = tok.head
            protasis = [e for e in events
                        if head_of.get(e.xml_id) == clause_head]
            apodosis = [e for e in events
                        if head_of.get(e.xml_id) is not None
                        and head_of[e.xml_id] == toks[clause_head].head]
            for a in apodosis or events[:1]:
                for p in protasis:
                    if a is p:
                        continue
                    struct.slinks.append(SLink(
                        xml_id=struct.next_id("sl"),
                        rel_type=SLinkRel.CONDITIONAL,
                        event_id=a.xml_id,
                        subordinated_event=p.xml_id,
                    ))

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _is_complement(toks: Sequence[Token], sub_i: int, gov_i: int) -> bool:
        seen = 0
        cur = sub_i
        while 0 <= cur < len(toks) and seen < 8:
            dep = toks[cur].dep
            head = toks[cur].head
            if head == gov_i and dep in _COMPLEMENT_DEPS:
                return True
            if head == cur or head < 0 or head >= len(toks):
                return False
            if dep not in _COMPLEMENT_DEPS and seen > 0:
                return False
            cur = head
            seen += 1
        return False

    @staticmethod
    def _relation(gov: Event, sub: Event, toks: Sequence[Token],
                  sub_i: int) -> Optional[SLinkRel]:
        cls = gov.event_class
        neg = gov.polarity == Polarity.NEG
        if cls in (EventClass.REPORTING, EventClass.PERCEPTION):
            return SLinkRel.NEG_EVIDENTIAL if neg else SLinkRel.EVIDENTIAL
        if cls in (EventClass.I_STATE, EventClass.I_ACTION):
            lemma = gov.pred.lower()
            if lemma in COUNTER_FACTIVE_LEMMAS:
                return SLinkRel.COUNTER_FACTIVE
            if lemma in FACTIVE_LEMMAS:
                return (SLinkRel.COUNTER_FACTIVE if neg
                        else SLinkRel.FACTIVE)
            return SLinkRel.INTENSIONAL
        return None

    # -- direct speech following a reporting verb --------------------------
    def _link_following_speech(self, doc: SegmentedDocument,
                              struct: AnnotationStructure,
                              by_sent: Dict[str, List[Event]]) -> None:
        """In this corpus a reporting verb frequently introduces speech that
        the parser places in a separate sentence (or a separate verse). Every
        event inside the quotation is subordinated to the reporting event, as
        A.3.3.1.2 requires.
        """
        existing = {(l.event_id, l.subordinated_event) for l in struct.slinks}
        sents = doc.sentences
        for si, sent in enumerate(sents):
            reporters = [e for e in by_sent.get(sent.sent_id, [])
                         if e.event_class in (EventClass.REPORTING,
                                              EventClass.PERCEPTION)]
            if not reporters:
                continue
            gov = reporters[-1]
            opens_quote = any(t.text in ('"', "``", "“") for t in sent.tokens)
            if not opens_quote:
                continue
            # events inside the same sentence after the quote mark
            qpos = next((i for i, t in enumerate(sent.tokens)
                         if t.text in ('"', "``", "“")), None)
            for ev in by_sent.get(sent.sent_id, []):
                if ev is gov or not ev.target:
                    continue
                pos = next((i for i, t in enumerate(sent.tokens)
                            if t.xml_id == ev.head_token), None)
                if pos is not None and qpos is not None and pos > qpos:
                    key = (gov.xml_id, ev.xml_id)
                    if key not in existing:
                        existing.add(key)
                        struct.slinks.append(SLink(
                            xml_id=struct.next_id("sl"),
                            rel_type=(SLinkRel.NEG_EVIDENTIAL
                                      if gov.polarity == Polarity.NEG
                                      else SLinkRel.EVIDENTIAL),
                            event_id=gov.xml_id,
                            subordinated_event=ev.xml_id,
                        ))
            # continuation sentences that remain inside the quotation
            depth = _quote_delta(sent.tokens)
            j = si + 1
            while depth % 2 == 1 and j < len(sents):
                nxt = sents[j]
                for ev in by_sent.get(nxt.sent_id, []):
                    key = (gov.xml_id, ev.xml_id)
                    if key not in existing:
                        existing.add(key)
                        struct.slinks.append(SLink(
                            xml_id=struct.next_id("sl"),
                            rel_type=(SLinkRel.NEG_EVIDENTIAL
                                      if gov.polarity == Polarity.NEG
                                      else SLinkRel.EVIDENTIAL),
                            event_id=gov.xml_id,
                            subordinated_event=ev.xml_id,
                        ))
                depth += _quote_delta(nxt.tokens)
                j += 1
                if j - si > 12:
                    break


def _quote_delta(tokens) -> int:
    return sum(1 for t in tokens if t.text in ('"', "``", "''", "“", "”"))
