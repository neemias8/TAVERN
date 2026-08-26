"""
Stage 2 - <EVENT> annotation (thesis Section 6.2.2).

Three classes of trigger, corresponding to the values of @pos: verbal events
from finite and non-finite verbs, nominal events from event-denoting nouns
(including deverbal nominalisations), and adjectival/copular states where the
state satisfies the standard's temporal-relevance criteria.

Span extent follows A.2.1.2.3: the minimal chunk including auxiliaries and
negation, not the syntactic head alone.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..stage1_preprocessing.pericopes import PericopeLayer
from ..stage1_preprocessing.segmentation import SegmentedDocument, Sentence, Token
from .enums import (Aspect, EventClass, EventType, POS, Polarity, Tense, VForm)
from .model import AnnotationStructure, Event

# ---------------------------------------------------------------------------
# Lexical evidence for the closed classes
# ---------------------------------------------------------------------------

REPORTING_LEMMAS = {
    "say", "tell", "ask", "answer", "reply", "declare", "announce", "report",
    "speak", "call", "cry", "shout", "exclaim", "command", "order", "warn",
    "explain", "insist", "deny", "confess", "testify", "witness", "proclaim",
    "preach", "teach", "rebuke", "promise", "swear", "protest", "urge",
    "beg", "plead", "accuse", "charge", "read", "write", "sing", "pray",
    "greet", "bless", "curse", "quote", "mention", "state", "add", "note",
}

PERCEPTION_LEMMAS = {
    "see", "hear", "watch", "look", "behold", "observe", "notice", "perceive",
    "feel", "smell", "taste", "witness", "glimpse", "spot", "sight",
}

ASPECTUAL_LEMMAS: Dict[str, str] = {
    # lemma -> ALINK relation type (A.3.4)
    "begin": "INITIATES", "start": "INITIATES", "commence": "INITIATES",
    "set": "INITIATES", "proceed": "INITIATES",
    "finish": "CULMINATES", "complete": "CULMINATES",
    "accomplish": "CULMINATES", "fulfil": "CULMINATES",
    "fulfill": "CULMINATES", "end": "TERMINATES",
    "stop": "TERMINATES", "cease": "TERMINATES", "quit": "TERMINATES",
    "continue": "CONTINUES", "keep": "CONTINUES", "persist": "CONTINUES",
    "resume": "REINITIATES", "restart": "REINITIATES",
    "reinitiate": "REINITIATES", "reignite": "REINITIATES",
}

I_STATE_LEMMAS = {
    "believe", "think", "know", "want", "wish", "hope", "fear", "doubt",
    "suppose", "expect", "intend", "desire", "need", "love", "hate",
    "remember", "forget", "understand", "realise", "realize", "consider",
    "regard", "suspect", "trust", "prefer", "plan", "seek", "long",
}

I_ACTION_LEMMAS = {
    "try", "attempt", "seek", "avoid", "refuse", "agree", "decide",
    "promise", "offer", "prevent", "allow", "permit", "forbid", "cause",
    "make", "let", "force", "compel", "persuade", "convince", "arrange",
    "prepare", "look", "delay", "postpone", "manage", "fail", "succeed",
    "conspire", "plot", "betray", "plan",
}

FACTIVE_LEMMAS = {
    "know", "remember", "realise", "realize", "understand", "regret",
    "see", "notice", "forget", "discover", "find", "recognise", "recognize",
    "reveal", "prove", "show",
}

COUNTER_FACTIVE_LEMMAS = {"pretend", "imagine", "dream", "lie", "falsify"}

# Lexical event nouns and deverbal nominalisations occurring in this corpus.
NOMINAL_EVENT_LEMMAS = {
    "crucifixion", "death", "betrayal", "resurrection", "burial", "arrest",
    "trial", "denial", "arrival", "departure", "entry", "cleansing",
    "anointing", "supper", "meal", "feast", "banquet", "wedding", "sacrifice",
    "offering", "prayer", "blessing", "curse", "teaching", "preaching",
    "healing", "miracle", "sign", "wonder", "judgement", "judgment",
    "condemnation", "sentence", "scourging", "flogging", "mocking",
    "beating", "kiss", "kiss", "cry", "shout", "voice", "earthquake",
    "darkness", "tearing", "rending", "rising", "coming", "return",
    "ascension", "transfiguration", "baptism", "passion", "sorrow",
    "lament", "weeping", "mourning", "escape", "flight", "conspiracy",
    "plot", "plan", "agreement", "covenant", "promise", "command",
    "commandment", "question", "answer", "reply", "accusation", "charge",
    "testimony", "witness", "confession", "release", "riot", "uproar",
    "gathering", "assembly", "council", "meeting", "journey", "procession",
    "triumph", "victory", "salvation", "redemption", "war", "battle",
    "famine", "pestilence", "tribulation", "persecution", "abomination",
    "harvest", "sowing", "reaping", "watch", "vigil", "sleep", "awakening",
    "hour", "work", "deed", "act", "action", "birth", "life",
}

# Nouns that name a temporal region rather than an event; excluded so that the
# timex tagger owns them.
TEMPORAL_NOUNS = {
    "day", "days", "night", "nights", "hour", "hours", "morning", "evening",
    "week", "weeks", "month", "months", "year", "years", "time", "times",
    "sabbath", "passover", "moment", "dawn", "daybreak", "noon", "midnight",
    "afternoon", "season", "period",
}

COPULA_LEMMAS = {"be", "become", "remain", "seem", "appear", "stay"}

MODAL_LEMMAS = {
    "will": "NONE", "shall": "NONE", "would": "POSSIBILITY",
    "can": "POSSIBILITY", "could": "POSSIBILITY", "may": "PERMISSION",
    "might": "POSSIBILITY", "must": "NECESSITY", "should": "OBLIGATION",
    "ought": "OBLIGATION", "need": "NECESSITY",
}

NEGATION_LEMMAS = {"not", "n't", "never", "no", "none", "nothing", "neither",
                   "nor", "cannot"}

# Verbs whose subtree is a reported-speech complement introducing direct speech
_SPEECH_DEPS = {"ccomp", "xcomp", "acl", "advcl", "conj", "parataxis"}


class EventTagger:
    """Emits an <EVENT> for each event-denoting expression."""

    def __init__(self, pericopes: Optional[PericopeLayer] = None):
        self.pericopes = pericopes

    # -- entry point ------------------------------------------------------
    def tag(self, doc: SegmentedDocument, struct: AnnotationStructure) -> None:
        for sent in doc.sentences:
            self._tag_sentence(sent, struct)

    # -- per sentence -----------------------------------------------------
    def _tag_sentence(self, sent: Sentence, struct: AnnotationStructure) -> None:
        toks = sent.tokens
        for i, tok in enumerate(toks):
            trigger = self._classify_trigger(tok, toks, i)
            if trigger is None:
                continue
            pos, cls, etype = trigger
            span = self._span(tok, toks, i, pos)
            if not span:
                continue
            eid = struct.next_id("e")
            ev = Event(
                xml_id=eid,
                target=[t.xml_id for t in span],
                pred=tok.lemma.upper(),
                event_class=cls,
                event_type=etype,
                tense=self._tense(tok, toks, i),
                aspect=self._aspect(tok, toks, i),
                vform=self._vform(tok),
                pos=pos,
                polarity=self._polarity(tok, toks, i),
                modality=self._modality(tok, toks, i),
                verse_key=tok.verse_key,
                verse_keys=[tok.verse_key],
                sent_id=sent.sent_id,
                pericope_id=(self.pericopes.of_verse(tok.verse_key)
                             if self.pericopes else None),
                text=" ".join(t.text for t in span),
                head_token=tok.xml_id,
            )
            struct.add_event(ev)

    # -- trigger identification -------------------------------------------
    def _classify_trigger(self, tok: Token, toks: Sequence[Token], i: int):
        lemma = tok.lemma

        if tok.pos in ("VERB", "AUX"):
            if tok.pos == "AUX" and tok.dep in ("aux", "auxpass", "cop"):
                return None
            if tok.pos == "AUX" and lemma in COPULA_LEMMAS:
                # copular state: only if it carries a temporally relevant
                # predicate complement
                if not self._copular_is_relevant(tok, toks, i):
                    return None
                return POS.ADJECTIVE, EventClass.STATE, EventType.STATE
            cls, etype = self._verb_class(lemma, tok, toks, i)
            return POS.VERB, cls, etype

        if tok.pos == "NOUN":
            if lemma in TEMPORAL_NOUNS:
                return None
            if lemma in NOMINAL_EVENT_LEMMAS or self._is_deverbal(lemma):
                return POS.NOUN, self._noun_class(lemma), EventType.PROCESS
            return None

        if tok.pos == "ADJ":
            # adjectival state: A.2.1.2.2 temporal-relevance criteria
            if self._adj_is_relevant(tok, toks, i):
                return POS.ADJECTIVE, EventClass.STATE, EventType.STATE
            return None

        return None

    def _verb_class(self, lemma: str, tok: Token, toks, i: int):
        if lemma in REPORTING_LEMMAS and self._has_complement(tok, toks, i):
            return EventClass.REPORTING, EventType.PROCESS
        if lemma in PERCEPTION_LEMMAS and self._has_complement(tok, toks, i):
            return EventClass.PERCEPTION, EventType.PROCESS
        if lemma in ASPECTUAL_LEMMAS and self._has_complement(tok, toks, i):
            return EventClass.ASPECTUAL, EventType.PROCESS
        if lemma in I_STATE_LEMMAS:
            # an I_STATE must govern another event; a lexical test alone cannot
            # establish that (thesis Section 6.2.2)
            if self._has_complement(tok, toks, i):
                return EventClass.I_STATE, EventType.STATE
            return EventClass.STATE, EventType.STATE
        if lemma in I_ACTION_LEMMAS and self._has_complement(tok, toks, i):
            return EventClass.I_ACTION, EventType.PROCESS
        if lemma in COPULA_LEMMAS:
            return EventClass.STATE, EventType.STATE
        # OCCURRENCE: TRANSITION where the verb denotes a change of state,
        # PROCESS where it denotes an activity (A.2.1.3.3).
        etype = EventType.PROCESS if tok.tag == "VBG" else EventType.TRANSITION
        return EventClass.OCCURRENCE, etype

    def _noun_class(self, lemma: str) -> EventClass:
        if lemma in {"prayer", "teaching", "preaching", "question", "answer",
                     "reply", "testimony", "confession", "accusation",
                     "command", "commandment", "promise", "lament", "cry",
                     "shout", "voice"}:
            return EventClass.REPORTING
        if lemma in {"plot", "plan", "conspiracy", "agreement", "covenant"}:
            return EventClass.I_ACTION
        return EventClass.OCCURRENCE

    @staticmethod
    def _is_deverbal(lemma: str) -> bool:
        return bool(re.search(r"(tion|sion|ment|ance|ence|ure|al|ing)$", lemma)) \
            and len(lemma) > 6

    def _has_complement(self, tok: Token, toks: Sequence[Token], i: int) -> bool:
        for j, t in enumerate(toks):
            if j != i and t.head == i and t.dep in _SPEECH_DEPS:
                return True
        # a following quotation mark is evidence of direct speech
        for j in range(i + 1, min(i + 4, len(toks))):
            if toks[j].text in ('"', "``", "''", "“"):
                return True
        return False

    def _copular_is_relevant(self, tok: Token, toks: Sequence[Token],
                             i: int) -> bool:
        """A copular clause is annotated when its complement is temporally
        relevant: it contains a temporal expression, or an adjective/noun
        predicate that holds of the narrative present.

        'it was the third hour' (Mark 15:25) is the motivating case: a tagger
        restricted to verbs produces no event there and leaves the <TIMEX3>
        unanchored (thesis Section 6.2.2).
        """
        for j, t in enumerate(toks):
            if t.head == i and t.dep in ("attr", "acomp", "oprd", "npadvmod",
                                         "advmod", "nsubj", "attr"):
                if t.lemma in TEMPORAL_NOUNS or t.pos in ("ADJ", "NOUN", "NUM"):
                    return True
        return False

    def _adj_is_relevant(self, tok: Token, toks: Sequence[Token],
                         i: int) -> bool:
        if tok.dep in ("acomp", "attr", "oprd"):
            return True
        return False

    # -- span extent ------------------------------------------------------
    def _span(self, tok: Token, toks: Sequence[Token], i: int,
              pos: POS) -> List[Token]:
        """A.2.1.2.3 - minimal chunk including auxiliaries and negation."""
        idx = {i}
        if pos == POS.VERB or pos == POS.ADJECTIVE:
            for j, t in enumerate(toks):
                if t.head == i and t.dep in ("aux", "auxpass", "neg", "cop",
                                             "prt"):
                    idx.add(j)
                if t.head == i and t.dep == "advmod" and t.lemma in NEGATION_LEMMAS:
                    idx.add(j)
        if pos == POS.NOUN:
            for j, t in enumerate(toks):
                if t.head == i and t.dep in ("compound",):
                    idx.add(j)
        lo, hi = min(idx), max(idx)
        span = [t for k, t in enumerate(toks) if lo <= k <= hi
                and not (t.is_punct and k != i)]
        return span or [tok]

    # -- attributes -------------------------------------------------------
    def _tense(self, tok: Token, toks: Sequence[Token], i: int) -> Tense:
        """@tense is read from morphology. The default is NOT PAST: direct
        speech and prophecy are pervasive, and 'the Son of Man will be
        delivered' is FUTURE regardless of the reporting verb's tense
        (thesis Section 6.2.2).
        """
        aux_lemmas = [t.lemma for t in toks if t.head == i
                      and t.dep in ("aux", "auxpass")]
        if any(a in ("will", "shall") for a in aux_lemmas):
            return Tense.FUTURE
        if "going" in [t.text.lower() for t in toks if t.head == i]:
            return Tense.FUTURE
        morph = tok.morph
        if "Tense=Past" in morph:
            return Tense.PAST
        if "Tense=Pres" in morph:
            return Tense.PRESENT
        if tok.tag in ("VB",) and any(t.dep == "aux" and t.lemma == "to"
                                      for t in toks if t.head == i):
            return Tense.INFINITIVE
        if tok.tag in ("VBD", "VBN"):
            return Tense.PAST
        if tok.tag in ("VBZ", "VBP"):
            return Tense.PRESENT
        if tok.tag in ("VB", "VBG"):
            return Tense.NONE
        for a in aux_lemmas:
            if a in ("have", "be", "do"):
                return Tense.PAST
        return Tense.NONE

    def _aspect(self, tok: Token, toks: Sequence[Token], i: int) -> Aspect:
        aux = [(t.lemma, t.tag) for t in toks if t.head == i
               and t.dep in ("aux", "auxpass")]
        has_have = any(l == "have" for l, _ in aux)
        prog = tok.tag == "VBG" or any(l == "be" for l, _ in aux) and tok.tag == "VBG"
        if has_have and prog:
            return Aspect.PERFECTIVE_PROGRESSIVE
        if has_have:
            return Aspect.PERFECTIVE
        if prog:
            return Aspect.PROGRESSIVE
        if tok.tag == "VBD":
            return Aspect.PERFECTIVE
        return Aspect.NONE

    def _vform(self, tok: Token) -> VForm:
        if tok.tag == "VBG":
            return VForm.PRESPART
        if tok.tag == "VBN":
            return VForm.PASTPART
        if tok.tag == "VB":
            return VForm.INFINITIVE
        return VForm.NONE

    def _polarity(self, tok: Token, toks: Sequence[Token], i: int) -> Polarity:
        """Scoped to the event's own dependency subtree rather than to the
        sentence, so that a single negation does not mark every event of a long
        verse as negative (thesis Section 6.2.2)."""
        for t in toks:
            if t.head == i and (t.dep == "neg" or t.lemma in NEGATION_LEMMAS
                                and t.dep in ("advmod", "det", "neg")):
                return Polarity.NEG
        return Polarity.POS

    def _modality(self, tok: Token, toks: Sequence[Token],
                  i: int) -> Optional[str]:
        for t in toks:
            if t.head == i and t.dep in ("aux", "auxpass"):
                m = MODAL_LEMMAS.get(t.lemma)
                if m and m != "NONE":
                    return m
        return None
