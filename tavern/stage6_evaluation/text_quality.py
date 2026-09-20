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
from typing import Dict, List, Sequence

# The faithfulness measurements live in Stage 5 (`stage5_generation.fidelity`)
# because they compare a fusion with its OWN inputs -- no chronology, no
# harmony, no reference -- which is what lets the generation loop use them to
# repair itself. Re-exported here so the reporting path has one import.
from ..stage5_generation.fidelity import (  # noqa: F401
    MAX_INTERNAL_REDUNDANCY, MAX_NOVEL_CONTENT, MIN_DETAIL_RECALL, check,
    detail_recall, internal_redundancy, novel_content)

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


# ---------------------------------------------------------------------------
@dataclass
class FidelityReport:
    """Both sides of the premise, over a whole run."""
    total: int
    dropped: int              # events below MIN_DETAIL_RECALL
    invented: int             # events above MAX_NOVEL_CONTENT
    repeated: int             # events restating their own wording
    failing: int              # events failing any of the three
    median_recall: float
    median_novel: float
    median_redundancy: float
    by_fusion: Dict[str, int] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.failing / self.total if self.total else 0.0

    def as_row(self) -> dict:
        return {"total": self.total, "dropped": self.dropped,
                "invented": self.invented, "repeated": self.repeated,
                "failing": self.failing,
                "fraction": round(self.fraction, 4),
                "median_detail_recall": round(self.median_recall, 4),
                "median_novel_content": round(self.median_novel, 4),
                "median_internal_redundancy": round(self.median_redundancy, 4),
                "by_fusion": dict(self.by_fusion),
                "examples": self.examples[:5]}


def scan_fidelity(events: Sequence[dict],
                  min_recall: float = MIN_DETAIL_RECALL,
                  max_novel: float = MAX_NOVEL_CONTENT) -> FidelityReport:
    """Measure a run's curation records.

    Each record is one event as `stage5_generation.consolidate` writes it:
    `consolidated` plus `sources[*]["text"]`. Recomputed here rather than read
    from the record's own `detail_recall`/`novel_content` fields, so the guard
    is independent of the run that produced them -- it can be pointed at an
    older `curation.json` written before those fields existed.
    """
    from statistics import median
    rs, ns, ds, bad = [], [], [], []
    drop = inv = rep = 0
    by: Dict[str, int] = {}
    for e in events:
        fused = e.get("consolidated", "")
        srcs = [s.get("text", "") for s in e.get("sources", [])]
        if not fused or not srcs:
            continue
        v = check(fused, srcs, min_recall, max_novel)
        rs.append(v.recall)
        ns.append(v.novel)
        ds.append(v.redundancy)
        by[e.get("fusion", "unknown")] = by.get(e.get("fusion", "unknown"),
                                                0) + 1
        drop += v.recall < min_recall
        inv += v.novel > max_novel
        rep += "twice" in v.reason
        if not v.ok:
            bad.append(f"{e.get('marker', '?')} recall={v.recall:.2f} "
                       f"novel={v.novel:.2f} repeated={v.redundancy:.2f}")
    return FidelityReport(
        total=len(rs), dropped=drop, invented=inv, repeated=rep,
        failing=len(bad),
        median_recall=median(rs) if rs else 0.0,
        median_novel=median(ns) if ns else 0.0,
        median_redundancy=median(ds) if ds else 0.0,
        by_fusion=by, examples=bad)


def assert_fidelity_below(events: Sequence[dict], max_fraction: float = 0.02,
                          ignore_fallback: bool = True) -> FidelityReport:
    """Regression guard: fail if too many events lose or invent detail.

    Measured against the pre-fix `ancoragem` artifacts, which had no repair
    path at all: 104/289 = 36.0%.

    `ignore_fallback` excludes the events `RepairingFuser` handed to
    `UnionFuser` after the backbone failed twice. That is not an exemption
    for convenience: union is detail-preserving by construction and cannot
    fail the recall axis, but it does not deduplicate either, so it fails the
    repetition axis on clusters whose accounts say the same thing in nearly
    the same words -- 11 of the 30 fallbacks in the gemma3:4b run. Counting
    those as failures of the fusion path would make this guard permanently
    red for a reason the fusion path does not control and already reports.
    The fallbacks stay visible in `by_fusion`, and their own fidelity is in
    the report; assert on that separately if what you want to bound is how
    often the backbone gives up.
    """
    scored = ([e for e in events if e.get("fusion") != "union"]
              if ignore_fallback else list(events))
    report = scan_fidelity(scored)
    if report.fraction > max_fraction:
        raise AssertionError(
            f"{report.failing}/{report.total} events ({report.fraction:.1%}) "
            f"fail the fusion premise ({report.dropped} lose detail, "
            f"{report.invented} invent it, {report.repeated} repeat "
            f"themselves), above the {max_fraction:.1%} threshold. "
            f"Examples: {report.examples[:3]}")
    return report
