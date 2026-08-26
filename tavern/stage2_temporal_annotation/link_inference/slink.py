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

    # -- direct speech: document-level quotation scope ---------------------
    def _link_following_speech(self, doc: SegmentedDocument,
                               struct: AnnotationStructure,
                               by_sent: Dict[str, List[Event]]) -> None:
        """Subordinate every event inside a quotation to the event reporting it.

        A.3.3.1.2 requires one <SLINK> for each reporting or perception event,
        and the events it governs are the events of its complement. In this
        corpus a complement is frequently a discourse of dozens of verses --
        the Olivet discourse is 97 verses in Matthew and the farewell discourse
        91 in John -- so a sentence-local test finds only its first clause and
        leaves the rest of the discourse in the narrative world. Since that
        material is 30.6% of the corpus (Table tab:ann-discourse), the
        veridicality partition of Section 6.3.1 depends on getting it right.

        The scope is therefore computed over the whole document: quotation depth
        is tracked across every token in order, and each maximal quoted run is
        attributed to the nearest reporting or perception event lying outside
        quotation -- in the run's own sentence where there is one, since English
        allows the reporting clause to follow the quotation ("...?" he asked),
        and otherwise the most recent one before it.
        """
        quote_tokens, runs = _quotation_scope(doc)
        if not runs:
            return

        outside: List[Tuple[int, Event]] = []
        order = _document_order(doc)
        for ev in struct.events.values():
            if ev.event_class not in (EventClass.REPORTING,
                                      EventClass.PERCEPTION):
                continue
            if not ev.target or ev.target[0] in quote_tokens:
                continue
            pos = order.get(ev.head_token or ev.target[0])
            if pos is not None:
                outside.append((pos, ev))
        outside.sort(key=lambda p: p[0])
        if not outside:
            return

        sent_of_event = {ev.xml_id: ev.sent_id for ev in struct.events.values()}
        existing = {(l.event_id, l.subordinated_event) for l in struct.slinks}

        events_by_pos: List[Tuple[int, Event]] = []
        for ev in struct.events.values():
            pos = order.get(ev.head_token or (ev.target[0] if ev.target else ""))
            if pos is not None:
                events_by_pos.append((pos, ev))
        events_by_pos.sort(key=lambda p: p[0])

        import bisect
        reporter_pos = [p for p, _ in outside]

        for start, end, sent_ids in runs:
            gov = None
            # a reporting event in one of the run's own sentences, outside the
            # quotation, whether it precedes or follows the quoted material
            for pos, ev in outside:
                if ev.sent_id in sent_ids:
                    gov = ev
                    break
            if gov is None:
                i = bisect.bisect_right(reporter_pos, start) - 1
                if i >= 0:
                    gov = outside[i][1]
            if gov is None:
                continue
            rel = (SLinkRel.NEG_EVIDENTIAL if gov.polarity == Polarity.NEG
                   else SLinkRel.EVIDENTIAL)
            lo = bisect.bisect_left([p for p, _ in events_by_pos], start)
            for pos, ev in events_by_pos[lo:]:
                if pos > end:
                    break
                if ev.xml_id == gov.xml_id:
                    continue
                key = (gov.xml_id, ev.xml_id)
                if key in existing:
                    continue
                existing.add(key)
                struct.slinks.append(SLink(
                    xml_id=struct.next_id("sl"), rel_type=rel,
                    event_id=gov.xml_id, subordinated_event=ev.xml_id,
                ))

    # -- legacy sentence-local pass (kept for the ablation) ----------------
    def _link_following_speech_local(self, doc: SegmentedDocument,
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


# ---------------------------------------------------------------------------
_OPEN_QUOTES = {'"', "``", "“"}
_CLOSE_QUOTES = {'"', "''", "”"}


def _document_order(doc: SegmentedDocument) -> Dict[str, int]:
    order: Dict[str, int] = {}
    i = 0
    for sent in doc.sentences:
        for tok in sent.tokens:
            order[tok.xml_id] = i
            i += 1
    return order


def _quotation_scope(doc: SegmentedDocument):
    """Quotation depth over the whole document.

    Returns the set of token ids lying inside a quotation, and the maximal
    quoted runs as (start, end, sentence ids) in document token order.

    A straight double quote is both opener and closer, so the state is normally
    toggled. One convention of the translation has to be handled explicitly:
    a speech spanning several paragraphs opens a quotation mark at the start of
    EACH paragraph and closes only at the end of the speech. A verse-initial
    quotation mark encountered while already inside a quotation is therefore a
    continuation, not a close; treating it as a close inverts the state for the
    remainder of the document and puts narration inside the quotation and
    speech outside it. Typographic quotes, where present, are directional and
    need no such rule.
    """
    inside: Set[str] = set()
    runs: List[Tuple[int, int, Set[str]]] = []
    depth = 0
    pos = 0
    start: Optional[int] = None
    sents: Set[str] = set()
    seen_verses: Set[Tuple[str, int, int]] = set()

    def close_run(end_pos: int) -> None:
        nonlocal start, sents
        if start is not None and end_pos >= start:
            runs.append((start, end_pos, set(sents)))
        start = None
        sents = set()

    for sent in doc.sentences:
        for tok in sent.tokens:
            verse_initial = tok.verse_key not in seen_verses
            seen_verses.add(tok.verse_key)
            t = tok.text
            opened = closed = False
            if t in ("\u201c", "``"):
                depth += 1
                opened = True
            elif t in ("\u201d", "''"):
                depth = max(0, depth - 1)
                closed = depth == 0
            elif t == '"':
                if depth and verse_initial:
                    opened = True          # paragraph continuation
                elif depth:
                    depth -= 1
                    closed = depth == 0
                else:
                    depth += 1
                    opened = True
            if opened and depth == 1 and start is None:
                start = pos + 1
                sents = {sent.sent_id}
            elif closed and depth == 0:
                close_run(pos - 1)
            elif depth:
                inside.add(tok.xml_id)
                sents.add(sent.sent_id)
            pos += 1
    close_run(pos - 1)
    return inside, runs
