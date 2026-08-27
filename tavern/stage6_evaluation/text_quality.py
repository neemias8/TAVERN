"""
Stage 6 - a regression guard against glued-word decoding artefacts.

gemma3:4b served through Ollama's `/api/generate` with a mistuned
`repeat_penalty` (see `OLLAMA_REPEAT_PENALTY` in
`tavern/stage5_generation/backbones.py`) drops whitespace under repetition
pressure: "came toBethphegeon theMountofOlves,Jesussenttwodisciples...". ROUGE
and METEOR both score the corrupted text without complaint -- word-level
n-gram overlap does not care whether the words are separated -- so a decoding
regression here would not otherwise be caught downstream. This module counts
how many generated events show the artefact and fails loud past a low
threshold, measured against the repeat_penalty=1.5 baseline (3/249 = 1.2%).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

#: two lowercase letters directly followed by a capital and a lowercase, e.g.
#: the boundary inside "toBethphegeon" -- a dropped word gap.
_GLUE_CASE = re.compile(r"[a-z]{2}[A-Z][a-z]")
#: a run of 18+ letters with no separator -- several words fused into one.
_GLUE_LONG = re.compile(r"[A-Za-z]{18,}")


@dataclass
class GluedWordReport:
    corrupted: int
    total: int
    fraction: float
    examples: List[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {"corrupted": self.corrupted, "total": self.total,
                "fraction": round(self.fraction, 4),
                "examples": self.examples[:5]}


def is_glued(text: str) -> bool:
    return bool(_GLUE_CASE.search(text) or _GLUE_LONG.search(text))


def _snippet(text: str) -> str:
    m = _GLUE_CASE.search(text) or _GLUE_LONG.search(text)
    if not m:
        return ""
    lo, hi = max(0, m.start() - 15), min(len(text), m.end() + 15)
    return text[lo:hi]


def scan(texts: Sequence[str]) -> GluedWordReport:
    """Count how many of `texts` (one per generated event) show the artefact."""
    bad = [t for t in texts if is_glued(t)]
    n = len(texts)
    return GluedWordReport(
        corrupted=len(bad), total=n,
        fraction=(len(bad) / n) if n else 0.0,
        examples=[_snippet(t) for t in bad])


def assert_below_threshold(texts: Sequence[str], max_fraction: float = 0.01
                           ) -> GluedWordReport:
    """Regression guard: fail if more than `max_fraction` of events are glued.

    Default threshold is below the measured repeat_penalty=1.5 baseline
    (1.2%), so that specific regression -- or an equivalent one -- trips it,
    while leaving room for the rare artefact any decoding produces.
    """
    report = scan(texts)
    if report.fraction > max_fraction:
        raise AssertionError(
            f"{report.corrupted}/{report.total} events "
            f"({report.fraction:.1%}) show glued-word decoding artefacts, "
            f"above the {max_fraction:.1%} threshold. Examples: "
            f"{report.examples[:3]}")
    return report
