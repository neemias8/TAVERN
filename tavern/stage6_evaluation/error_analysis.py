"""
Stage 6 - the error taxonomy for the induced timeline
(thesis Table tab:res-qualerrors).

Seven classes, each counted against the held-out chronology, with a worked
example. The last class -- a subordinated event admitted to the timeline -- is
expected to be zero if the veridicality partition works as specified; a non-zero
count identifies a class of subordination the <SLINK> inference failed to
detect, which is a more useful finding than a clean result.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..config import DISCOURSE_BLOCKS
from .chronology import Chronology


@dataclass
class ErrorClass:
    name: str
    count: int
    example: str = ""


def analyse(chronology: Chronology, clustering, units, induced,
            matching: Dict[int, str], structs, segments=None
            ) -> List[ErrorClass]:
    out: List[ErrorClass] = []
    rank = chronology.rank()

    covered_keys: Set[Tuple[str, int, int]] = set()
    for cl in clustering.clusters:
        for m in cl.members:
            covered_keys |= set(units[m].verse_keys)

    # 1 event not detected: no cluster covers any of the event's verses
    undetected = [e for e in chronology.events
                  if e.all_keys and not (set(e.all_keys) & covered_keys)]
    out.append(ErrorClass("Event not detected", len(undetected),
                          _ex(undetected)))

    # 2 detected but not aligned across documents: the event has versions in
    #   several Gospels, its verses are covered, but no single cluster holds
    #   two of them
    not_aligned = []
    for e in chronology.events:
        if not e.contested() or e.event_id in matching:
            continue
        if set(e.all_keys) & covered_keys:
            not_aligned.append(e)
    for e in chronology.events:
        cid = matching.get(e.event_id)
        if cid and e.contested():
            cl = clustering.by_id(cid)
            if cl is not None and cl.size < 2:
                not_aligned.append(e)
    out.append(ErrorClass("Detected but not aligned across documents",
                          len(not_aligned), _ex(not_aligned)))

    # 3 over-merged: one cluster overlaps two or more canonical events
    #   substantially
    over = []
    for cl in clustering.clusters:
        keys = set()
        for m in cl.members:
            keys |= set(units[m].verse_keys)
        hits = [e for e in chronology.events
                if e.all_keys and len(set(e.all_keys) & keys)
                >= 0.5 * len(set(e.all_keys))]
        if len(hits) >= 2:
            over.append((cl, hits))
    out.append(ErrorClass(
        "Over-merged cluster (two events joined)", len(over),
        f"{over[0][0].cluster_id} covers events "
        f"{[h.event_id for h in over[0][1]][:3]}" if over else ""))

    # 4 under-merged: one canonical event's verses spread over several clusters
    under = []
    for e in chronology.events:
        if not e.all_keys:
            continue
        cids = set()
        for cl in clustering.clusters:
            for m in cl.members:
                if set(units[m].verse_keys) & set(e.all_keys):
                    cids.add(cl.cluster_id)
        if len(cids) >= 3:
            under.append(e)
    out.append(ErrorClass("Under-merged cluster (one event split)",
                          len(under), _ex(under)))

    # 5 / 6 misordering, split by whether the pair crosses a day boundary
    pairs = [(rank[eid], induced.rank.get(cid), eid)
             for eid, cid in matching.items()
             if induced.rank.get(cid) is not None]
    pairs.sort()
    adjacent = 0
    displaced = 0
    adj_ex = disp_ex = ""
    day_of = {e.event_id: e.day for e in chronology.events}
    for i in range(len(pairs) - 1):
        (ri, hi, ei), (rj, hj, ej) = pairs[i], pairs[i + 1]
        if hi <= hj:
            continue
        if day_of.get(ei) == day_of.get(ej):
            adjacent += 1
            if not adj_ex:
                adj_ex = (f"events {ei} and {ej} "
                          f"({day_of.get(ei)}) transposed")
        else:
            displaced += 1
            if not disp_ex:
                disp_ex = (f"event {ei} ({day_of.get(ei)}) placed after "
                           f"{ej} ({day_of.get(ej)})")
    out.append(ErrorClass("Misordered: adjacent transposition", adjacent,
                          adj_ex))
    out.append(ErrorClass("Misordered: displaced across a day boundary",
                          displaced, disp_ex))

    # 7 subordinated event admitted to the timeline. Counted only for events
    # inside the quotation: the narrative frame of a discourse belongs to the
    # narrative world and its reporting event is correctly eligible.
    from ..stage2_temporal_annotation.link_inference.slink import (
        _quotation_scope)
    quoted: Dict[str, set] = {}
    if segments:
        for book, doc in segments.items():
            quoted[book], _r = _quotation_scope(doc)
    leaked = 0
    leak_ex = ""
    for book, c0, v0, c1, v1, label in DISCOURSE_BLOCKS:
        s = structs.get(book)
        if s is None:
            continue
        qt = quoted.get(book)
        for ev in s.events.values():
            if not ev.verse_key:
                continue
            _b, ch, vs = ev.verse_key
            if not ((c0, v0) <= (ch, vs) <= (c1, v1)):
                continue
            inside = (ev.head_token in qt) if qt is not None else \
                ((c0, v0) < (ch, vs))
            if inside and ev.xml_id in s.eligible:
                leaked += 1
                if not leak_ex:
                    leak_ex = (f"{ev.pred} at {book} {ch}:{vs} "
                               f"inside '{label}'")
    out.append(ErrorClass("Subordinated event admitted to the timeline",
                          leaked, leak_ex))
    return out


def _ex(events) -> str:
    if not events:
        return ""
    e = events[0]
    refs = ", ".join(f"{b.capitalize()} {e.refs[b]}" for b in e.books) \
        if e.books else "no version"
    return f"#{e.event_id} \"{e.description}\" ({refs})"
