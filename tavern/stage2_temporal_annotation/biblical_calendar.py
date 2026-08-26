"""
Stage 2 - the biblical domain profile (Appendix A, `semaf-time-biblical`).

Extends the value space of ISO 24617-1 to cover a luni-solar liturgical
calendar, temporal hours reckoned from sunrise, Roman night watches and a day
boundary at sunset. Affects @value, @mod and @pred only; adds no element and
no attribute, and extends no value set that the standard closes.

Two projection modes. Relative mode (default) assigns no absolute value;
temporal information resides in the anchor chain. Absolute mode projects onto
a configured Julian year and is not used for any result in the thesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .enums import Mod, TimexType

PROFILE_NAME = "semaf-time-biblical"

# --- Table tab:prof-hours: temporal hours (interval-in-progress reading) ----
ORDINAL_HOURS: Dict[str, str] = {
    "first": "T06:00", "second": "T07:00", "third": "T08:00",
    "fourth": "T09:00", "fifth": "T10:00", "sixth": "T11:00",
    "seventh": "T12:00", "eighth": "T13:00", "ninth": "T14:00",
    "tenth": "T15:00", "eleventh": "T16:00", "twelfth": "T17:00",
}

ORDINAL_NUMBER: Dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12,
}

# --- Table tab:prof-parts: parts of the sunset-to-sunset day interval ------
DAY_PARTS: Dict[str, Tuple[str, str, float]] = {
    # surface -> (@value time component, @pred, position in the day interval
    #             expressed as a fraction of the sunset-to-sunset day)
    "evening": ("T18:00", "EVENING", 0.00),
    "when evening came": ("T18:00", "EVENING", 0.00),
    "when evening had come": ("T18:00", "EVENING", 0.00),
    "in the evening": ("T18:00", "EVENING", 0.00),
    "nightfall": ("T18:00", "EVENING", 0.00),
    "sunset": ("T18:00", "SUNSET", 0.00),
    "night": ("T21:00", "NIGHT", 0.13),
    "tonight": ("T21:00", "NIGHT", 0.13),
    "midnight": ("T00:00", "MIDNIGHT", 0.25),
    "cockcrow": ("T03:00", "COCKCROW", 0.38),
    "before the rooster crows": ("T03:00", "COCKCROW", 0.38),
    "dawn": ("T06:00", "DAWN", 0.50),
    "daybreak": ("T06:00", "DAWN", 0.50),
    "at dawn": ("T06:00", "DAWN", 0.50),
    "very early in the morning": ("T06:00", "DAWN", 0.50),
    "early in the morning": ("T06:00", "DAWN", 0.51),
    "very early": ("T06:00", "DAWN", 0.50),
    "while it was still dark": ("T05:00", "DAWN", 0.48),
    "at daybreak": ("T06:00", "DAWN", 0.50),
    "early": ("T06:30", "MORNING", 0.52),
    "morning": ("T08:00", "MORNING", 0.58),
    "in the morning": ("T08:00", "MORNING", 0.58),
    "noon": ("T11:00", "NOON", 0.71),
    "midday": ("T11:00", "NOON", 0.71),
    "afternoon": ("T14:00", "AFTERNOON", 0.83),
    "late": ("T17:00", "LATE", 0.95),
    "already late": ("T17:00", "LATE", 0.95),
}

# --- Table tab:prof-watches: Roman four-watch division --------------------
NIGHT_WATCHES: Dict[str, Tuple[str, str]] = {
    "first watch": ("T19:30", "EVENING"),
    "second watch": ("T22:30", "MIDNIGHT"),
    "third watch": ("T01:30", "COCKCROW"),
    "fourth watch": ("T04:30", "MORNING_WATCH"),
}

# --- Table tab:prof-feasts: feast lexicon --------------------------------
@dataclass(frozen=True)
class Feast:
    pred: str
    position: str            # position in the Hebrew liturgical calendar
    timex_type: TimexType
    value_relative: str
    relational: bool = False  # defined relative to another feast/Sabbath


FEASTS: Dict[str, Feast] = {
    "passover": Feast("PASSOVER", "14 Nisan", TimexType.DATE, "XXXX-XX-XX"),
    "the passover": Feast("PASSOVER", "14 Nisan", TimexType.DATE, "XXXX-XX-XX"),
    "feast of unleavened bread": Feast("UNLEAVENED_BREAD", "15-21 Nisan",
                                       TimexType.DATE, "XXXX-XX-XX"),
    "unleavened bread": Feast("UNLEAVENED_BREAD", "15-21 Nisan",
                              TimexType.DATE, "XXXX-XX-XX"),
    "day of preparation": Feast("PREPARATION", "day before a Sabbath or feast",
                                TimexType.DATE, "XXXX-XX-XX", relational=True),
    "preparation day": Feast("PREPARATION", "day before a Sabbath or feast",
                             TimexType.DATE, "XXXX-XX-XX", relational=True),
    "sabbath": Feast("SABBATH", "seventh day of the week", TimexType.DATE,
                     "XXXX-WXX-6"),
    "the sabbath": Feast("SABBATH", "seventh day of the week", TimexType.DATE,
                         "XXXX-WXX-6"),
    "first day of the week": Feast("FIRST_DAY", "day after a Sabbath",
                                   TimexType.DATE, "XXXX-WXX-7"),
    "pentecost": Feast("PENTECOST", "50 days after Passover", TimexType.DATE,
                       "XXXX-XX-XX"),
    "the feast": Feast("PASSOVER", "14 Nisan", TimexType.DATE, "XXXX-XX-XX"),
}

# Weekday lexicon (the chronology's day labels, for the anchor scaffold only;
# the corpus itself names weekdays only through 'Sabbath' and
# 'the first day of the week').
WEEKDAY_ORDER = ["Saturday", "Palm Sunday", "Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Dark Saturday", "Sunday"]

# --- Duration lexicon ----------------------------------------------------
NUMBER_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "a": 1, "an": 1, "half": 0, "several": 0,
}

DURATION_UNITS: Dict[str, str] = {
    "second": "S", "seconds": "S", "minute": "M", "minutes": "M",
    "hour": "H", "hours": "H", "day": "D", "days": "D",
    "week": "W", "weeks": "W", "month": "M", "months": "M",
    "year": "Y", "years": "Y",
}

SUB_DAY_UNITS = {"S", "M", "H"}


def duration_value(count: Optional[int], unit_code: str) -> str:
    """ISO 8601 duration. Sub-day units take the PT designator.

    The standard's own examples write 'P3H' for three hours, which is malformed
    under ISO 8601 (a normative reference of the standard); the profile emits
    the correct form (Appendix A, Section A.7).
    """
    n = "X" if count is None else str(count)
    if unit_code in SUB_DAY_UNITS:
        return f"PT{n}{unit_code}"
    if unit_code == "M":            # ambiguous: minutes handled above
        return f"P{n}M"
    return f"P{n}{unit_code}"


def hour_value(ordinal: str) -> Optional[Tuple[str, Mod]]:
    """Value for an ordinal temporal hour, always marked APPROX."""
    t = ORDINAL_HOURS.get(ordinal.lower())
    if t is None:
        return None
    return f"XXXX-XX-XX{t}", Mod.APPROX


def day_part_value(surface: str) -> Optional[Tuple[str, str, float]]:
    return DAY_PARTS.get(surface.lower())


def feast_of(surface: str) -> Optional[Feast]:
    s = surface.lower().strip()
    if s in FEASTS:
        return FEASTS[s]
    if s.startswith("the ") and s[4:] in FEASTS:
        return FEASTS[s[4:]]
    return None


# --- Absolute mode -------------------------------------------------------
#: Candidates discussed in the chronological literature (Appendix A, A.8).
ABSOLUTE_YEARS = {
    30: {"crucifixion": "0030-04-07", "nisan14": "0030-04-06"},
    33: {"crucifixion": "0033-04-03", "nisan14": "0033-04-02"},
}


def project_absolute(value: str, year: int) -> str:
    """Minimal projection used only when absolute mode is enabled."""
    anchor = ABSOLUTE_YEARS.get(year)
    if not anchor or not value.startswith("XXXX"):
        return value
    return anchor["crucifixion"][:4] + value[4:]
