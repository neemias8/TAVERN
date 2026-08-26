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
from typing import Dict, List, Optional, Sequence

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
