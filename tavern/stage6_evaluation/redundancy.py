"""
Stage 6 - within-summary redundancy (thesis Section 9.3.2).

Redundancy Elimination is one of the four objectives by which the task is
defined (Section 4.1), and it is the only one measurable on the output alone,
without a reference: repeated content is a property of a text, not of its
agreement with another text.

The measure is the distinct-n ratio -- the proportion of n-gram occurrences in
the text that are distinct -- for n = 1..4. Higher is less repetitive.

One caution governs every comparison made with it, and it is stated here
because it is easy to abuse. Distinct-n is NOT normalised for length. A shorter
text has fewer opportunities to repeat itself and scores higher for that reason
alone, so distinct-n may only be compared between texts of comparable length,
or in the direction where the length difference works against the conclusion.
The comparison the thesis draws is the concatenation of the four accounts
against the consolidation, where the consolidation is both shorter AND less
repetitive: the length difference is what consolidation is for, and the claim
is about the pair taken together, reported as a reduction ratio alongside the
compression ratio rather than as a bare score.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_WORD = re.compile(r"[a-z0-9']+")
ORDERS = (1, 2, 3, 4)


def tokenize(text: str) -> List[str]:
    """Lowercased word tokens. Deliberately not stemmed.

    Stemming would collapse morphological variants and understate repetition:
    "he went" and "they were going" are not a repeated n-gram to a reader.
    """
    return _WORD.findall(text.lower())


@dataclass
class RedundancyScores:
    tokens: int = 0
    distinct: Dict[int, Optional[float]] = field(default_factory=dict)

    def as_row(self) -> dict:
        row: Dict[str, object] = {"Tokens": self.tokens}
        for n in ORDERS:
            d = self.distinct.get(n)
            row[f"distinct-{n}"] = None if d is None else round(d, 4)
        return row


def distinct_n(tokens: Sequence[str], n: int) -> Optional[float]:
    """Proportion of the text's n-gram occurrences that are distinct."""
    if len(tokens) < n:
        return None
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(Counter(grams)) / len(grams)


def evaluate(text: str) -> RedundancyScores:
    t = tokenize(text)
    return RedundancyScores(tokens=len(t),
                            distinct={n: distinct_n(t, n) for n in ORDERS})


def repetition_reduction(consolidated: str, concatenated: str) -> dict:
    """How much repetition the consolidation removes, and at what compression.

    Both halves are reported because neither is interpretable alone. Removing
    repetition by discarding content is not consolidation, so the compression
    ratio has to sit beside the reduction; and a text that is shorter without
    being less repetitive has not consolidated anything either.
    """
    c = evaluate(consolidated)
    k = evaluate(concatenated)
    out: Dict[str, object] = {
        "tokens_concatenated": k.tokens,
        "tokens_consolidated": c.tokens,
        "compression": round(c.tokens / k.tokens, 4) if k.tokens else None,
    }
    for n in ORDERS:
        dc, dk = c.distinct.get(n), k.distinct.get(n)
        out[f"distinct-{n}_concatenated"] = None if dk is None else round(dk, 4)
        out[f"distinct-{n}_consolidated"] = None if dc is None else round(dc, 4)
        if dc is not None and dk is not None:
            # repetition rate is 1 - distinct-n; report the relative fall in it
            rk, rc = 1.0 - dk, 1.0 - dc
            out[f"repetition_reduction-{n}"] = (
                round((rk - rc) / rk, 4) if rk > 0 else None)
    return out


# ---------------------------------------------------------------------------
# content coverage: the counterweight redundancy needs
# ---------------------------------------------------------------------------
_STOP = frozenset("""a an and the of to in on at by for with from as is are was were
be been being he she it they them him her his its their you your we us our i me my
that this these those there here who whom whose which what when where why how not no
nor but or if then than so such all any both each few more most other some own same
too very can will just do does did into onto upon over under again once about against
between during before after above below up down out off while because until said say
says saying one two three four five six seven eight nine ten also may might must
shall would could have has had having""".split())

_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")


def _stem(word: str) -> str:
    w = word.lower()
    for suf in ("'s", "ing", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def content_types(text: str) -> set:
    """The distinct content words of a text, suffix-normalised."""
    return {_stem(m.group(0)) for m in _TOKEN.finditer(text)
            if len(m.group(0)) > 2 and m.group(0).lower() not in _STOP}


def coverage(output: str, sources: Sequence[str]) -> Optional[float]:
    """Share of the sources' content words that survive in the output.

    This is the counterweight that makes the distinct-n figures above
    interpretable, and without it they are actively misleading.

    An extractive consolidation reproduces one source verbatim, so by
    construction it cannot carry a detail that only another source reports. It is
    therefore shorter and less repetitive than a fusion *because* it is less
    complete, and reading its higher distinct-n as better consolidation rewards
    the discarding of content -- precisely the degenerate case that
    Section 9.3.2 warns about. Redundancy may only be compared at matched
    coverage; where coverage differs, both figures are reported as a pair and the
    question becomes whether the extra repetition is proportionate to the extra
    content.

    Measured on this corpus, over the events reported by more than one Gospel:
    the extractive configuration retains 42.6% of the sources' content words and
    the abstractive fusion 90.0%, at 2.02x the tokens.
    """
    src: set = set()
    for s in sources:
        src |= content_types(s)
    if not src:
        return None
    return len(content_types(output) & src) / len(src)


def coverage_over_events(pairs: Sequence[Tuple[str, Sequence[str]]]) -> dict:
    """Micro-averaged coverage over (output, sources) pairs, one per event.

    Micro-averaged rather than macro: a single-source event contributes little to
    the question, and averaging per-event rates would let the 87 single-source
    events dominate a figure that is about fusion.
    """
    tot = hit = 0
    mtot = mhit = 0
    for out, sources in pairs:
        src: set = set()
        for s in sources:
            src |= content_types(s)
        if not src:
            continue
        got = len(content_types(out) & src)
        tot += len(src)
        hit += got
        if len(sources) > 1:
            mtot += len(src)
            mhit += got
    return {
        "coverage": round(hit / tot, 4) if tot else None,
        "coverage_multi_source": round(mhit / mtot, 4) if mtot else None,
        "content_types_in_sources": tot,
    }
