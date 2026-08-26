"""
Stage 1 - within-document coreference over persons and places.

The resulting entity chains are not used by the annotation stage but by the
cross-document event coreference of Stage 3, where shared participants are one
of the alignment signals (thesis Section 5.2, Section 6.4.2).

No neural coreference model is used. The corpus's referential inventory is
small, closed and highly conventionalised, so a lexicon-driven resolver over
named entities plus a pronoun-to-salient-antecedent heuristic is both
sufficient and deterministic - which matters, because Section 9.7 requires
Stages 1-3 to be deterministic given the corpus.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .segmentation import SegmentedDocument, Token

# Canonical entity lexicon for the Passion Week narrative.
ENTITY_LEXICON: Dict[str, Set[str]] = {
    "JESUS": {"jesus", "christ", "the lord", "son of man", "the nazarene",
              "the galilean", "teacher", "rabbi", "master", "messiah",
              "king of the jews", "son of david", "the christ"},
    "PETER": {"peter", "simon peter", "simon", "cephas"},
    "JUDAS": {"judas", "judas iscariot", "iscariot"},
    "PILATE": {"pilate", "pontius pilate", "the governor"},
    "CAIAPHAS": {"caiaphas", "the high priest"},
    "HEROD": {"herod"},
    "MARY_MAGDALENE": {"mary magdalene"},
    "MARY": {"mary"},
    "MARTHA": {"martha"},
    "LAZARUS": {"lazarus"},
    "BARABBAS": {"barabbas"},
    "JOHN_DISCIPLE": {"john", "the other disciple",
                      "the disciple whom jesus loved"},
    "JAMES": {"james"},
    "THOMAS": {"thomas", "didymus"},
    "PHILIP": {"philip"},
    "ANDREW": {"andrew"},
    "JOSEPH_ARIMATHEA": {"joseph of arimathea", "joseph"},
    "NICODEMUS": {"nicodemus"},
    "SIMON_CYRENE": {"simon of cyrene"},
    "DISCIPLES": {"the disciples", "his disciples", "the twelve", "the eleven",
                  "disciples"},
    "CHIEF_PRIESTS": {"the chief priests", "chief priests"},
    "PHARISEES": {"the pharisees", "pharisees"},
    "SADDUCEES": {"the sadducees", "sadducees"},
    "SCRIBES": {"the scribes", "the teachers of the law", "scribes"},
    "ELDERS": {"the elders", "elders"},
    "SANHEDRIN": {"the sanhedrin", "the council", "the whole council"},
    "SOLDIERS": {"the soldiers", "soldiers", "the guards", "guards"},
    "CROWD": {"the crowd", "the crowds", "the people", "the multitude"},
    "SERVANT_GIRL": {"a servant girl", "the servant girl", "one of the servant girls"},
    "JERUSALEM": {"jerusalem"},
    "BETHANY": {"bethany"},
    "BETHPHAGE": {"bethphage"},
    "MOUNT_OLIVES": {"the mount of olives", "mount of olives", "olivet"},
    "GETHSEMANE": {"gethsemane"},
    "GOLGOTHA": {"golgotha", "the place of the skull", "calvary"},
    "TEMPLE": {"the temple", "the temple courts", "the temple area"},
    "PRAETORIUM": {"the praetorium", "the palace of the roman governor"},
    "TOMB": {"the tomb", "the sepulchre"},
    "COURTYARD": {"the courtyard"},
    "UPPER_ROOM": {"the upper room", "the guest room"},
    "GALILEE": {"galilee"},
    "EMMAUS": {"emmaus"},
}

_SURFACE_TO_ID: Dict[str, str] = {}
for _eid, _surfaces in ENTITY_LEXICON.items():
    for _s in _surfaces:
        _SURFACE_TO_ID[_s] = _eid

_MAX_SURFACE_WORDS = max(len(s.split()) for s in _SURFACE_TO_ID)

_PRONOUN_GROUPS = {
    "he": "SG_M", "him": "SG_M", "his": "SG_M", "himself": "SG_M",
    "she": "SG_F", "her": "SG_F", "hers": "SG_F", "herself": "SG_F",
    "they": "PL", "them": "PL", "their": "PL", "theirs": "PL",
    "themselves": "PL",
}

_PLURAL_ENTITIES = {"DISCIPLES", "CHIEF_PRIESTS", "PHARISEES", "SADDUCEES",
                    "SCRIBES", "ELDERS", "SANHEDRIN", "SOLDIERS", "CROWD"}
_FEMALE_ENTITIES = {"MARY", "MARY_MAGDALENE", "MARTHA", "SERVANT_GIRL"}


@dataclass
class Mention:
    entity_id: str
    tokens: List[str]           # token xml:ids
    surface: str
    verse_key: Tuple[str, int, int]
    is_pronoun: bool = False


@dataclass
class EntityChains:
    mentions: List[Mention] = field(default_factory=list)
    by_token: Dict[str, str] = field(default_factory=dict)
    by_verse: Dict[Tuple[str, int, int], Set[str]] = field(default_factory=dict)

    def entities_in(self, keys) -> Set[str]:
        out: Set[str] = set()
        for k in keys:
            out |= self.by_verse.get(k, set())
        return out

    def entities_of_tokens(self, token_ids) -> Set[str]:
        return {self.by_token[t] for t in token_ids if t in self.by_token}


def resolve(doc: SegmentedDocument) -> EntityChains:
    chains = EntityChains()
    salient: List[str] = []          # most recent first

    for sent in doc.sentences:
        toks = sent.tokens
        i = 0
        while i < len(toks):
            matched = False
            for n in range(min(_MAX_SURFACE_WORDS, len(toks) - i), 0, -1):
                window = toks[i:i + n]
                if any(t.is_punct for t in window):
                    continue
                surface = " ".join(t.text.lower() for t in window)
                surface = re.sub(r"\s+([',])", r"\1", surface)
                eid = _SURFACE_TO_ID.get(surface)
                if eid is None and n > 1:
                    # allow a leading determiner to be dropped
                    if window[0].text.lower() in {"the", "a", "an"}:
                        eid = _SURFACE_TO_ID.get(
                            " ".join(t.text.lower() for t in window[1:]))
                if eid is None and n == 1 and window[0].pos == "PROPN":
                    eid = _SURFACE_TO_ID.get(window[0].text.lower())
                if eid:
                    m = Mention(entity_id=eid,
                                tokens=[t.xml_id for t in window],
                                surface=surface, verse_key=window[0].verse_key)
                    _record(chains, m)
                    if eid in salient:
                        salient.remove(eid)
                    salient.insert(0, eid)
                    i += n
                    matched = True
                    break
            if matched:
                continue
            tok = toks[i]
            low = tok.text.lower()
            if tok.pos == "PRON" and low in _PRONOUN_GROUPS:
                group = _PRONOUN_GROUPS[low]
                target = _pick_antecedent(salient, group)
                if target:
                    _record(chains, Mention(entity_id=target,
                                            tokens=[tok.xml_id],
                                            surface=low,
                                            verse_key=tok.verse_key,
                                            is_pronoun=True))
            i += 1
        salient = salient[:6]
    return chains


def _pick_antecedent(salient: List[str], group: str) -> Optional[str]:
    for eid in salient:
        if group == "PL" and eid in _PLURAL_ENTITIES:
            return eid
        if group == "SG_F" and eid in _FEMALE_ENTITIES:
            return eid
        if group == "SG_M" and eid not in _PLURAL_ENTITIES \
                and eid not in _FEMALE_ENTITIES:
            return eid
    return salient[0] if salient else None


def _record(chains: EntityChains, m: Mention) -> None:
    chains.mentions.append(m)
    for t in m.tokens:
        chains.by_token[t] = m.entity_id
    chains.by_verse.setdefault(m.verse_key, set()).add(m.entity_id)


def resolve_all(docs: Dict[str, SegmentedDocument]) -> Dict[str, EntityChains]:
    return {book: resolve(doc) for book, doc in docs.items()}
