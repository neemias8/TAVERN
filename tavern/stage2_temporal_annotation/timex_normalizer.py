"""
Stage 2 - normalisation, @temporalFunction and the anchoring hierarchy
(thesis Sections 6.2.4 and 6.2.5).

ISO-TimeML resolves deictic expressions against a <TIMEX3> bearing
functionInDocument="CREATION_TIME". Narrative prose has no creation time in the
relevant sense, and the standard defers narrative-function values. The solution
is a hierarchy of three anchoring levels, built entirely from mechanisms the
standard already provides -- empty <TIMEX3> elements, @temporalFunction and
@anchorTimeID chains:

  Level 1  the document anchor        one per document, holds no information
  Level 2  narrative reference time   one per pericope, chained to the previous
  Level 3  utterance time             one per REPORTING scope

Deictic expressions in narration anchor at level 2; deictic expressions in
direct speech anchor at level 3.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..stage1_preprocessing.pericopes import PericopeLayer
from .biblical_calendar import (PROFILE_NAME, duration_value, feast_of,
                                hour_value, project_absolute)
from .enums import EventClass, FunctionInDocument, Mod, TimexType
from .model import AnnotationStructure, Event, Timex3

#: Expressions whose normalised value plus anchor chain fixes their position
#: relative to the week independently of the document (thesis Section 6.4.1).
ANCHORABLE_KINDS = {"feast", "hour", "watch", "relday", "daypart"}

#: Day-part surfaces too vague to anchor on their own.
NON_ANCHORABLE_DAYPARTS = {"late", "already late", "night", "tonight"}


class TimexNormalizer:
    def __init__(self, pericopes: PericopeLayer, mode: str = "relative",
                 year: int = 30):
        self.pericopes = pericopes
        self.mode = mode
        self.year = year

    # -- entry point ------------------------------------------------------
    def normalise(self, struct: AnnotationStructure) -> None:
        meta: Dict[str, dict] = struct.__dict__.get("_timex_meta", {})
        doc_anchor = self._document_anchor(struct)
        nrt = self._narrative_reference_times(struct, doc_anchor)
        utt = self._utterance_times(struct, nrt)

        for tid, tx in list(struct.timexes.items()):
            if tx.anchor_level > 0:
                continue                        # an anchor element itself
            info = meta.get(tid, {})
            self._value_for(tx, info)
            host = self._host_anchor(struct, tx, nrt, utt, doc_anchor)
            if tx.anchor_time_id is None and self._needs_anchor(tx, info):
                tx.anchor_time_id = host
                tx.temporal_function = True
            tx.anchorable = self._is_anchorable(tx, info)
            if self.mode == "absolute" and tx.value:
                tx.value = project_absolute(tx.value, self.year)
                if "X" not in tx.value:
                    tx.temporal_function = False

    # -- level 1 ----------------------------------------------------------
    def _document_anchor(self, struct: AnnotationStructure) -> str:
        tid = struct.next_id("t")
        struct.add_timex(Timex3(
            xml_id=tid, timex_type=TimexType.DATE, target=[],
            value="XXXX-XX-XX", temporal_function=True,
            function_in_document=FunctionInDocument.CREATION_TIME,
            anchor_level=1,
            comment="document anchor; holds no temporal information "
                    "(thesis Section 6.2.5, Level 1)",
        ))
        return tid

    # -- level 2 ----------------------------------------------------------
    def _narrative_reference_times(self, struct: AnnotationStructure,
                                   doc_anchor: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        prev: Optional[str] = None
        for per in self.pericopes.for_book(struct.book):
            tid = struct.next_id("t")
            struct.add_timex(Timex3(
                xml_id=tid, timex_type=TimexType.DATE, target=[],
                value="XXXX-XX-XX", temporal_function=True,
                anchor_time_id=(prev or doc_anchor),
                anchor_level=2, pericope_id=per.pericope_id,
                comment=f"narrative reference time of '{per.title}' "
                        f"(thesis Section 6.2.5, Level 2)",
            ))
            out[per.pericope_id] = tid
            prev = tid
        return out

    # -- level 3 ----------------------------------------------------------
    def _utterance_times(self, struct: AnnotationStructure,
                         nrt: Dict[str, str]) -> Dict[str, str]:
        """One utterance time per REPORTING event, anchored to the narrative
        reference time of the pericope containing the report."""
        out: Dict[str, str] = {}
        for ev in struct.events.values():
            if ev.event_class not in (EventClass.REPORTING,
                                      EventClass.PERCEPTION):
                continue
            host = nrt.get(ev.pericope_id or "")
            tid = struct.next_id("t")
            struct.add_timex(Timex3(
                xml_id=tid, timex_type=TimexType.TIME, target=[],
                value="XXXX-XX-XXTXX:XX", temporal_function=True,
                anchor_time_id=host, anchor_level=3,
                pericope_id=ev.pericope_id,
                comment=f"utterance time of {ev.xml_id} "
                        f"(thesis Section 6.2.5, Level 3)",
            ))
            out[ev.xml_id] = tid
        return out

    # -- value assignment -------------------------------------------------
    def _value_for(self, tx: Timex3, info: dict) -> None:
        kind = info.get("kind")

        if kind == "hour":
            hv = hour_value(info.get("ordinal") or "")
            if hv:
                tx.value, tx.mod = hv
                tx.temporal_function = True
            return

        if kind == "watch":
            from .biblical_calendar import NIGHT_WATCHES
            v, _pred = NIGHT_WATCHES[info["watch"]]
            tx.value = f"XXXX-XX-XX{v}"
            tx.mod = Mod.APPROX
            tx.temporal_function = True
            return

        if kind == "daypart":
            tx.value = f"XXXX-XX-XX{info['time']}"
            tx.mod = Mod.APPROX
            tx.temporal_function = True
            return

        if kind == "feast":
            feast = feast_of(info["feast"])
            if feast:
                tx.value = feast.value_relative
                tx.pred = feast.pred
                # 'the Sabbath' and 'the first day of the week' are the only
                # two expressions whose value is fully determined without an
                # anchor, since the weekday is intrinsic (Appendix A, A.7)
                tx.temporal_function = "WXX" not in (tx.value or "")
            return

        if kind == "duration":
            tx.value = duration_value(info.get("count"), info.get("unit", "D"))
            tx.temporal_function = False
            return

        if kind == "set":
            tx.value = info.get("value")
            tx.quant = info.get("quant")
            tx.freq = info.get("freq")
            if not tx.quant and not tx.freq:
                tx.quant = "EVERY"        # constraint 4 of Appendix B
            tx.temporal_function = False
            return

        if kind == "relday":
            # A value such as '+P3D' is NOT a legal @value under the standard;
            # the offset is carried by @anchorTimeID plus the duration
            # (thesis Section 6.2.3).
            tx.value = "XXXX-XX-XX"
            tx.temporal_function = True
            return

        if tx.value is None:
            tx.value = ("XXXX-XX-XX" if tx.timex_type == TimexType.DATE
                        else "XXXX-XX-XXTXX:XX"
                        if tx.timex_type == TimexType.TIME else None)
            tx.temporal_function = True

    # -- anchor selection -------------------------------------------------
    def _needs_anchor(self, tx: Timex3, info: dict) -> bool:
        if tx.timex_type in (TimexType.DURATION, TimexType.SET):
            return False
        if tx.value and "X" not in tx.value:
            return False
        return True

    def _host_anchor(self, struct: AnnotationStructure, tx: Timex3,
                     nrt: Dict[str, str], utt: Dict[str, str],
                     doc_anchor: str) -> Optional[str]:
        """Deictic expressions in narration anchor at level 2; those inside the
        scope of a REPORTING event anchor at level 3."""
        governing = self._governing_reporting_event(struct, tx)
        if governing and governing in utt:
            return utt[governing]
        if tx.pericope_id and tx.pericope_id in nrt:
            return nrt[tx.pericope_id]
        return doc_anchor

    @staticmethod
    def _governing_reporting_event(struct: AnnotationStructure,
                                   tx: Timex3) -> Optional[str]:
        """A <TIMEX3> in the same sentence as a REPORTING event, occurring after
        it, is taken to lie in its scope."""
        if not tx.target:
            return None
        best = None
        for ev in struct.events.values():
            if ev.sent_id != tx.sent_id:
                continue
            if ev.event_class not in (EventClass.REPORTING,
                                      EventClass.PERCEPTION):
                continue
            if not ev.target:
                continue
            if _token_index(ev.target[0]) < _token_index(tx.target[0]):
                if best is None or _token_index(ev.target[0]) > \
                        _token_index(struct.events[best].target[0]):
                    best = ev.xml_id
        return best

    # -- anchorability ----------------------------------------------------
    @staticmethod
    def _is_anchorable(tx: Timex3, info: dict) -> bool:
        kind = info.get("kind")
        if kind not in ANCHORABLE_KINDS:
            return False
        if kind == "daypart" and info.get("surface") in NON_ANCHORABLE_DAYPARTS:
            return False
        if kind == "relday" and info.get("offset") is None:
            return False
        if kind == "relday" and info.get("deictic"):
            # 'today' inside direct speech does not fix a position in the week
            return False
        return True


def _token_index(xml_id: str) -> int:
    """Order tokens by their id, which encodes chapter, verse, sentence and
    position: tk_<bk><ch>_<vs>_<sent>_<idx>."""
    try:
        parts = xml_id.split("_")
        chap = int("".join(ch for ch in parts[1] if ch.isdigit()))
        return (chap * 10 ** 9 + int(parts[2]) * 10 ** 6
                + int(parts[3]) * 10 ** 3 + int(parts[4]))
    except (IndexError, ValueError):
        return 0
