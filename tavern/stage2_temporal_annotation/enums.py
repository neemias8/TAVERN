"""
Stage 2 - closed value sets and the conformance resolutions of Appendix B.

The standard's normative annexes disagree on some thirty points. Every
disagreement is resolved here, once, with a reference to the clause of
ISO 24617-1:2012 on which the resolution rests. The governing principle is
asymmetric: STRICT ON OUTPUT, LENIENT ON INPUT (Appendix B, Section B.1).
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Optional


class StrEnum(str, Enum):
    def __str__(self) -> str:      # pragma: no cover
        return self.value


# ---------------------------------------------------------------------------
# <EVENT>
# ---------------------------------------------------------------------------

class EventClass(StrEnum):
    """A.2.1.3.2 - the seven-way classification."""
    OCCURRENCE = "OCCURRENCE"
    STATE = "STATE"
    REPORTING = "REPORTING"
    PERCEPTION = "PERCEPTION"
    ASPECTUAL = "ASPECTUAL"
    I_STATE = "I_STATE"
    I_ACTION = "I_ACTION"


class EventType(StrEnum):
    """A.2.1.3.3 - required, closed (conformance resolution R1)."""
    STATE = "STATE"
    PROCESS = "PROCESS"
    TRANSITION = "TRANSITION"


class Tense(StrEnum):
    PAST = "PAST"
    PRESENT = "PRESENT"
    FUTURE = "FUTURE"
    INFINITIVE = "INFINITIVE"
    NONE = "NONE"


class Aspect(StrEnum):
    """Canonical form is the long one; the short form is an input alias (R9)."""
    NONE = "NONE"
    PROGRESSIVE = "PROGRESSIVE"
    PERFECTIVE = "PERFECTIVE"
    PERFECTIVE_PROGRESSIVE = "PERFECTIVE_PROGRESSIVE"
    IMPERFECTIVE = "IMPERFECTIVE"
    IMPERFECTIVE_PROGRESSIVE = "IMPERFECTIVE_PROGRESSIVE"


ASPECT_ALIASES = {
    "PERFECTIVE_PROG": Aspect.PERFECTIVE_PROGRESSIVE,
    "IMPERFECTIVE_PROG": Aspect.IMPERFECTIVE_PROGRESSIVE,
}


class VForm(StrEnum):
    """A.2.1.3.10 spelling adopted; att.linguistic spelling accepted (R8)."""
    NONE = "NONE"
    INFINITIVE = "INFINITIVE"
    GERUND = "GERUND"
    PASTPART = "PASTPART"
    PRESPART = "PRESPART"


VFORM_ALIASES = {"PARTICIPLE": VForm.PASTPART}


class POS(StrEnum):
    VERB = "VERB"
    NOUN = "NOUN"
    ADJECTIVE = "ADJECTIVE"
    PREP = "PREP"
    OTHER = "OTHER"


class Polarity(StrEnum):
    POS = "POS"
    NEG = "NEG"


class Modality(StrEnum):
    NONE = "NONE"
    OBLIGATION = "OBLIGATION"
    PERMISSION = "PERMISSION"
    POSSIBILITY = "POSSIBILITY"
    NECESSITY = "NECESSITY"
    VOLITION = "VOLITION"


# ---------------------------------------------------------------------------
# <TIMEX3>
# ---------------------------------------------------------------------------

class TimexType(StrEnum):
    """A.2.2.3.3.1 - required, closed set of four (conformance resolution R2)."""
    DATE = "DATE"
    TIME = "TIME"
    DURATION = "DURATION"
    SET = "SET"


class FunctionInDocument(StrEnum):
    """A.2.2.3.8. Narrative-function values are deferred by the standard;
    the anchoring hierarchy of Section 6.2.5 uses NONE plus anchor chains."""
    NONE = "NONE"
    CREATION_TIME = "CREATION_TIME"
    EXPIRATION_TIME = "EXPIRATION_TIME"
    MODIFICATION_TIME = "MODIFICATION_TIME"
    PUBLICATION_TIME = "PUBLICATION_TIME"
    RELEASE_TIME = "RELEASE_TIME"
    RECEPTION_TIME = "RECEPTION_TIME"


class Mod(StrEnum):
    """Only START and EQUAL_OR_LESS are attested in the standard itself
    (A.2.2.3.5); the remainder are declared by the biblical domain profile
    (Appendix A, Table tab:prof-mod)."""
    START = "START"
    EQUAL_OR_LESS = "EQUAL_OR_LESS"
    APPROX = "APPROX"
    MID = "MID"
    END = "END"
    ON_OR_BEFORE = "ON_OR_BEFORE"
    ON_OR_AFTER = "ON_OR_AFTER"


ISO_NORMATIVE_MOD: FrozenSet[str] = frozenset({"START", "EQUAL_OR_LESS"})


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

class TLinkRel(StrEnum):
    """A.3.2 - fourteen relation types. DURING_INV appears in the schema
    (H.3.8) and in no normative guideline: accepted on input, never emitted
    (Appendix B, Section B.5)."""
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    IBEFORE = "IBEFORE"
    IAFTER = "IAFTER"
    INCLUDES = "INCLUDES"
    IS_INCLUDED = "IS_INCLUDED"
    DURING = "DURING"
    DURING_INV = "DURING_INV"
    SIMULTANEOUS = "SIMULTANEOUS"
    IDENTITY = "IDENTITY"
    BEGINS = "BEGINS"
    BEGUN_BY = "BEGUN_BY"
    ENDS = "ENDS"
    ENDED_BY = "ENDED_BY"


NEVER_EMITTED_TLINK = frozenset({TLinkRel.DURING_INV})

TLINK_INVERSE: Dict[str, str] = {
    "BEFORE": "AFTER", "AFTER": "BEFORE",
    "IBEFORE": "IAFTER", "IAFTER": "IBEFORE",
    "INCLUDES": "IS_INCLUDED", "IS_INCLUDED": "INCLUDES",
    "DURING": "DURING_INV", "DURING_INV": "DURING",
    "SIMULTANEOUS": "SIMULTANEOUS", "IDENTITY": "IDENTITY",
    "BEGINS": "BEGUN_BY", "BEGUN_BY": "BEGINS",
    "ENDS": "ENDED_BY", "ENDED_BY": "ENDS",
}


class SLinkRel(StrEnum):
    """Six types: A.3.3.1.1 enumerates five, H.3.6 adds CONDITIONAL, which
    A.3.3.2.2 independently mandates (conformance resolution R12).
    INTENSIONAL replaces TimeML's MODAL."""
    INTENSIONAL = "INTENSIONAL"
    EVIDENTIAL = "EVIDENTIAL"
    NEG_EVIDENTIAL = "NEG_EVIDENTIAL"
    FACTIVE = "FACTIVE"
    COUNTER_FACTIVE = "COUNTER_FACTIVE"
    CONDITIONAL = "CONDITIONAL"


class ALinkRel(StrEnum):
    """Verbal forms of H.3.1 (conformance resolution R11). REINITIATES is in
    the schema but not in the A.3.4.1 enumeration; it is emitted because the
    ASPECTUAL class does define a reinitiation subclass (Appendix B, B.5)."""
    INITIATES = "INITIATES"
    CULMINATES = "CULMINATES"
    TERMINATES = "TERMINATES"
    CONTINUES = "CONTINUES"
    REINITIATES = "REINITIATES"


ALINK_ALIASES = {
    "INITIATION": ALinkRel.INITIATES,
    "CULMINATION": ALinkRel.CULMINATES,
    "TERMINATION": ALinkRel.TERMINATES,
    "CONTINUATION": ALinkRel.CONTINUES,
}


class MLinkRel(StrEnum):
    MEASURES = "MEASURES"


# ---------------------------------------------------------------------------
# Cascade provenance (thesis Table tab:ann-cascade)
# ---------------------------------------------------------------------------

class CascadeLevel(int, Enum):
    SIGNAL = 1
    TIMEX = 2
    ASPECTUAL = 3
    NARRATIVE = 4
    CLOSURE = 5


CONFIDENCE_OF_LEVEL: Dict[int, float] = {
    1: 0.90,   # explicit signal            - high
    2: 0.85,   # explicit temporal expression - high
    3: 0.65,   # aspectual predicate        - medium
    4: 0.35,   # narrative progression      - low
    5: 0.50,   # closure                    - derived (no independent evidence)
}

CONFIDENCE_LABEL: Dict[int, str] = {
    1: "high", 2: "high", 3: "medium", 4: "low", 5: "derived",
}


# ---------------------------------------------------------------------------
# SLINK interaction rules (A.3.3.1.2) - used by the validator, constraint 8
# ---------------------------------------------------------------------------

#: CONDITIONAL is structurally derived (A.3.3.2.2) rather than licensed by the
#: governor's class, so it is admissible under every class.
SLINK_ALLOWED_BY_CLASS: Dict[str, FrozenSet[str]] = {
    "REPORTING": frozenset({"EVIDENTIAL", "NEG_EVIDENTIAL", "CONDITIONAL"}),
    "PERCEPTION": frozenset({"EVIDENTIAL", "NEG_EVIDENTIAL", "CONDITIONAL"}),
    "I_STATE": frozenset({"INTENSIONAL", "FACTIVE", "COUNTER_FACTIVE",
                          "CONDITIONAL"}),
    "I_ACTION": frozenset({"INTENSIONAL", "FACTIVE", "COUNTER_FACTIVE",
                           "CONDITIONAL"}),
    "OCCURRENCE": frozenset({"CONDITIONAL"}),
    "STATE": frozenset({"CONDITIONAL"}),
    "ASPECTUAL": frozenset({"CONDITIONAL"}),
}

#: Subordination types that preserve timeline eligibility (Equation eq:eligible)
ELIGIBILITY_PRESERVING = frozenset({"FACTIVE"})


def normalise_aspect(raw: Optional[str]) -> Aspect:
    if not raw:
        return Aspect.NONE
    raw = raw.upper()
    if raw in ASPECT_ALIASES:
        return ASPECT_ALIASES[raw]
    try:
        return Aspect(raw)
    except ValueError:
        return Aspect.NONE


def normalise_vform(raw: Optional[str]) -> VForm:
    if not raw:
        return VForm.NONE
    raw = raw.upper()
    if raw in VFORM_ALIASES:
        return VFORM_ALIASES[raw]
    try:
        return VForm(raw)
    except ValueError:
        return VForm.NONE


def normalise_alink(raw: Optional[str]) -> Optional[ALinkRel]:
    if not raw:
        return None
    raw = raw.upper()
    if raw in ALINK_ALIASES:
        return ALINK_ALIASES[raw]
    try:
        return ALinkRel(raw)
    except ValueError:
        return None
