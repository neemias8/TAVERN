"""
<TLINK> inference by the typed cascade of Table tab:ann-cascade
(thesis Section 6.2.7).

  Lv 1  explicit signal        arguments resolved from the signal's dependency
                               subtree; @relType from @pred; @signalID recorded
  Lv 2  temporal expression    event and <TIMEX3> in the same clause, related by
                               dependency attachment; the governing preposition
                               selects among IS_INCLUDED / SIMULTANEOUS /
                               BEGINS / ENDS
  Lv 3  aspectual predicate    derived from an <ALINK> of type INITIATES or
                               TERMINATES
  Lv 4  narrative progression  adjacent narrative-world events within the SAME
                               pericope, both OCCURRENCE with PAST tense and
                               PERFECTIVE aspect, receive BEFORE
  Lv 5  closure                computed in closure.py

Levels are applied in order and a relation asserted at a higher level is never
overwritten by a lower one. Level 4 is the only place where the narrative-order
assumption survives, and it is constrained on three axes simultaneously.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...stage1_preprocessing.segmentation import SegmentedDocument, Sentence, Token
from ..enums import (Aspect, CONFIDENCE_OF_LEVEL, EventClass, Tense, TLinkRel,
                     TimexType)
from ..model import AnnotationStructure, Event, TLink, Timex3
from ..signal_tagger import SIGNAL_LEXICON, MULTIWORD_SIGNALS

#: Preposition -> relType for level 2 (event/TIMEX3 attachment).
PREP_TO_REL = {
    "ON": TLinkRel.IS_INCLUDED, "AT": TLinkRel.IS_INCLUDED,
    "IN": TLinkRel.IS_INCLUDED, "DURING": TLinkRel.IS_INCLUDED,
    "WITHIN": TLinkRel.IS_INCLUDED, "THROUGHOUT": TLinkRel.IS_INCLUDED,
    "WHEN": TLinkRel.IS_INCLUDED, "WHILE": TLinkRel.SIMULTANEOUS,
    "AS": TLinkRel.SIMULTANEOUS, "NOW": TLinkRel.SIMULTANEOUS,
    "SINCE": TLinkRel.BEGINS, "FROM": TLinkRel.BEGINS,
    "UNTIL": TLinkRel.ENDS, "TO": TLinkRel.ENDS, "BY": TLinkRel.BEFORE,
    "BEFORE": TLinkRel.BEFORE, "AFTER": TLinkRel.AFTER,
}

_SIGNAL_REL = {}
for _s, (_p, _r) in SIGNAL_LEXICON.items():
    if _r:
        _SIGNAL_REL[_p] = TLinkRel(_r)
for _w, (_p, _r) in MULTIWORD_SIGNALS.items():
    if _r:
        _SIGNAL_REL[_p] = TLinkRel(_r)


class TLinkInferrer:
    def __init__(self, levels: Iterable[int] = (1, 2, 3, 4)):
        self.levels = set(levels)

    # -- entry point ------------------------------------------------------
    def infer(self, doc: SegmentedDocument, struct: AnnotationStructure) -> None:
        asserted: Dict[Tuple[str, str], TLink] = {}
        if 1 in self.levels:
            self._level1_signals(doc, struct, asserted)
        if 2 in self.levels:
            self._level2_timex(doc, struct, asserted)
        if 3 in self.levels:
            self._level3_aspectual(struct, asserted)
        if 4 in self.levels:
            self._level4_narrative(struct, asserted)

    # -- level 1 ----------------------------------------------------------
    def _level1_signals(self, doc: SegmentedDocument,
                        struct: AnnotationStructure,
                        asserted: Dict[Tuple[str, str], TLink]) -> None:
        for sent in doc.sentences:
            toks = sent.tokens
            idx = {t.xml_id: i for i, t in enumerate(toks)}
            events = [e for e in struct.events.values()
                      if e.sent_id == sent.sent_id and e.head_token in idx]
            timexes = [t for t in struct.timexes.values()
                       if t.sent_id == sent.sent_id and t.target]
            if not events:
                continue
            for sig in struct.signals.values():
                if sig.sent_id != sent.sent_id:
                    continue
                rel = _SIGNAL_REL.get(sig.pred)
                if rel is None:
                    continue
                si = idx.get(sig.target[0])
                if si is None:
                    continue
                left, right = self._resolve_signal_arguments(
                    toks, idx, si, events, timexes)
                if left is None or right is None:
                    continue
                self._assert(struct, asserted, left, right, rel, 1,
                             signal_id=sig.xml_id)

    def _resolve_signal_arguments(self, toks, idx, si, events, timexes):
        """Arguments resolved from the dependency subtree of the signal.

        The signal's head is one argument (or the event/timex governing it);
        the other is the nearest event or temporal expression on the opposite
        side.
        """
        head = toks[si].head
        # the constituent the signal introduces
        right = self._nearest(events, timexes, idx, si, direction=+1)
        # the constituent the signal attaches to
        left = None
        if 0 <= head < len(toks):
            left = self._element_at(events, timexes, idx, head)
        if left is None:
            left = self._nearest(events, timexes, idx, si, direction=-1)
        if left is not None and right is not None and left[0] == right[0]:
            right = self._nearest(events, timexes, idx, si, direction=+1,
                                  exclude=left[0])
        return left, right

    @staticmethod
    def _element_at(events, timexes, idx, pos):
        for e in events:
            if idx.get(e.head_token) == pos:
                return e.xml_id, "event"
        for t in timexes:
            if any(idx.get(tk) == pos for tk in t.target):
                return t.xml_id, "timex"
        return None

    @staticmethod
    def _nearest(events, timexes, idx, si, direction: int, exclude=None):
        best = None
        best_d = 10 ** 6
        for e in events:
            p = idx.get(e.head_token)
            if p is None or e.xml_id == exclude:
                continue
            d = (p - si) * direction
            if 0 < d < best_d:
                best, best_d = (e.xml_id, "event"), d
        for t in timexes:
            for tk in t.target:
                p = idx.get(tk)
                if p is None or t.xml_id == exclude:
                    continue
                d = (p - si) * direction
                if 0 < d < best_d:
                    best, best_d = (t.xml_id, "timex"), d
        return best

    # -- level 2 ----------------------------------------------------------
    def _level2_timex(self, doc: SegmentedDocument,
                      struct: AnnotationStructure,
                      asserted: Dict[Tuple[str, str], TLink]) -> None:
        for sent in doc.sentences:
            toks = sent.tokens
            idx = {t.xml_id: i for i, t in enumerate(toks)}
            events = [e for e in struct.events.values()
                      if e.sent_id == sent.sent_id and e.head_token in idx]
            timexes = [t for t in struct.timexes.values()
                       if t.sent_id == sent.sent_id and t.target]
            if not events or not timexes:
                continue
            sigs = {s.target[0]: s for s in struct.signals.values()
                    if s.sent_id == sent.sent_id and s.target}
            for tx in timexes:
                tpos = [idx[k] for k in tx.target if k in idx]
                if not tpos:
                    continue
                anchor_ev = self._attached_event(toks, idx, events, tpos)
                if anchor_ev is None:
                    continue
                rel, sig_id = self._select_relation(toks, idx, sigs, tpos, tx)
                self._assert(struct, asserted, (anchor_ev, "event"),
                             (tx.xml_id, "timex"), rel, 2, signal_id=sig_id)

    @staticmethod
    def _attached_event(toks, idx, events, tpos):
        """Follow the dependency chain up from the temporal expression to the
        first event head."""
        heads = {idx[e.head_token]: e.xml_id for e in events
                 if e.head_token in idx}
        for start in tpos:
            cur = start
            for _ in range(6):
                if cur in heads:
                    return heads[cur]
                nxt = toks[cur].head
                if nxt == cur or nxt < 0 or nxt >= len(toks):
                    break
                cur = nxt
        # fall back to the nearest event in the same clause
        best, best_d = None, 10 ** 6
        for p, eid in heads.items():
            d = min(abs(p - t) for t in tpos)
            if d < best_d:
                best, best_d = eid, d
        return best

    @staticmethod
    def _select_relation(toks, idx, sigs, tpos, tx):
        """The governing preposition selects among IS_INCLUDED, SIMULTANEOUS,
        BEGINS and ENDS."""
        first = min(tpos)
        for back in range(1, 4):
            p = first - back
            if p < 0:
                break
            sig = next((s for k, s in sigs.items() if idx.get(k) == p), None)
            if sig is not None:
                rel = PREP_TO_REL.get(sig.pred)
                if rel:
                    return rel, sig.xml_id
        if tx.timex_type == TimexType.DURATION:
            return TLinkRel.SIMULTANEOUS, None
        return TLinkRel.IS_INCLUDED, None

    # -- level 3 ----------------------------------------------------------
    def _level3_aspectual(self, struct: AnnotationStructure,
                          asserted: Dict[Tuple[str, str], TLink]) -> None:
        """An <ALINK> of type INITIATES or TERMINATES imposes a derived
        constraint between the aspectual event and its argument."""
        for al in struct.alinks:
            rel = {"INITIATES": TLinkRel.BEGINS,
                   "REINITIATES": TLinkRel.BEGINS,
                   "TERMINATES": TLinkRel.ENDS,
                   "CULMINATES": TLinkRel.ENDS,
                   "CONTINUES": TLinkRel.IS_INCLUDED}.get(str(al.rel_type))
            if rel is None:
                continue
            self._assert(struct, asserted, (al.event_id, "event"),
                         (al.related_to_event, "event"), rel, 3)

    # -- level 4 ----------------------------------------------------------
    def _level4_narrative(self, struct: AnnotationStructure,
                          asserted: Dict[Tuple[str, str], TLink]) -> None:
        """Adjacent narrative-world events within the same pericope, both
        OCCURRENCE with PAST tense and PERFECTIVE aspect, receive BEFORE.

        This is the only place where the narrative-order assumption survives.
        It applies only to events in the narrative world (so nothing inside a
        parable or a prophecy is ordered against the narration), only within a
        pericope, and only to perfective past occurrences.
        """
        by_pericope: Dict[str, List[Event]] = {}
        for ev in struct.events.values():
            if not self._level4_eligible(ev, struct):
                continue
            by_pericope.setdefault(ev.pericope_id or "?", []).append(ev)

        for pid, evs in by_pericope.items():
            evs.sort(key=_event_order_key)
            for a, b in zip(evs, evs[1:]):
                self._assert(struct, asserted, (a.xml_id, "event"),
                             (b.xml_id, "event"), TLinkRel.BEFORE, 4)

    @staticmethod
    def _level4_eligible(ev: Event, struct: AnnotationStructure) -> bool:
        if struct.eligible and ev.xml_id not in struct.eligible:
            return False
        if ev.event_class != EventClass.OCCURRENCE:
            return False
        if ev.tense != Tense.PAST:
            return False
        if ev.aspect not in (Aspect.PERFECTIVE, Aspect.PERFECTIVE_PROGRESSIVE):
            return False
        return True

    # -- assertion --------------------------------------------------------
    def _assert(self, struct: AnnotationStructure,
                asserted: Dict[Tuple[str, str], TLink],
                left: Tuple[str, str], right: Tuple[str, str],
                rel: TLinkRel, level: int,
                signal_id: Optional[str] = None) -> None:
        lid, lkind = left
        rid, rkind = right
        if lid == rid:
            return
        key = (lid, rid)
        rkey = (rid, lid)
        prior = asserted.get(key) or asserted.get(rkey)
        if prior is not None and prior.level <= level:
            return                          # higher level wins
        if prior is not None:
            struct.tlinks.remove(prior)
            asserted.pop(key, None)
            asserted.pop(rkey, None)
        link = TLink(
            xml_id=struct.next_id("l"),
            rel_type=rel,
            event_id=lid if lkind == "event" else None,
            time_id=lid if lkind == "timex" else None,
            related_to_event=rid if rkind == "event" else None,
            related_to_time=rid if rkind == "timex" else None,
            signal_id=signal_id,
            origin="asserted",
            level=level,
            confidence=CONFIDENCE_OF_LEVEL[level],
        )
        struct.add_tlink(link)
        asserted[key] = link


def _event_order_key(ev: Event):
    vk = ev.verse_key or ("", 0, 0)
    tok = ev.head_token or ""
    try:
        parts = tok.split("_")
        s = int(parts[3]); i = int(parts[4])
    except (IndexError, ValueError):
        s, i = 0, 0
    return (vk[1], vk[2], s, i)
