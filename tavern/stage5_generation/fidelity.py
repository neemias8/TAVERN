"""
Stage 5 - does a fusion keep what the accounts say, and only that?

The task's premise (thesis Section 8.1) is two-sided: every detail of every
account must survive into the consolidation, and a detail shared by several
accounts must be stated once. Until now nothing measured either side.
`stage6_evaluation/text_quality.py` catches glued words, and ROUGE against the
reference measures agreement with a *selection* that is itself incomplete
(Chapter 10's second threat to validity) -- so a fusion that silently drops a
whole witness, or that invents a paragraph out of nothing, scores no worse for
it. Measured on the `ancoragem` artifacts, 39/289 events carried at least 30%
content words present in no account at all, one of them an aviation accident
grafted onto Matthew 24:20; that passed every instrument the pipeline had.

Both measurements here compare a fused paragraph ONLY with the accounts it was
given. Neither reads the chronology, the harmony or the reference
consolidation, which is why they belong to Stage 5 and can run inside the
generation loop: they are a property of the fusion, not of the evaluation.
`stage6_evaluation.text_quality` re-exports them for the reporting path.

Terms are crudely stemmed before comparison. Without it `calls` -> `called` is
counted as a lost detail and any faithful paraphrase is punished as an
omission, which would make the guard an argument for copying -- the opposite
of what it is for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Set

from .backbones import _STOP

_WORD = re.compile(r"[A-Za-z']+")

#: the fusion keeps this fraction of the accounts' content terms, or it has
#: dropped something. Calibrated on the `ancoragem` artifacts so that the
#: events which lost a whole witness trip it -- E002 drops Matthew's donkey,
#: prophecy and cloaks and scores 0.57 -- while a genuine merge keeps its
#: margin: E074 fuses three accounts at 0.90, so the threshold sits below it
#: rather than on it. Median over `ancoragem` is 1.00.
MIN_DETAIL_RECALL = 0.88
#: content terms in the fusion that appear in no account. Anything above this
#: is invention, not paraphrase. Median over `ancoragem` is 0.02.
MAX_NOVEL_CONTENT = 0.20
#: token mass of the fusion sitting inside a 5-gram it states more than once.
#: Without this the guard has only one direction and its gradient points the
#: wrong way: told it had dropped a detail, gemma3:4b answered by pasting all
#: three accounts of E002 end to end -- 309 tokens, recall 0.91, which passes
#: a recall-only check while being the exact failure the fusion exists to
#: avoid. Compared against the sources' own worst repetition, never in the
#: absolute: this corpus narrates a command and then its execution in the
#: same words, and a fusion cannot be asked to undo that.
MAX_INTERNAL_REDUNDANCY = 0.25


def _stem(w: str) -> str:
    """Enough morphology to stop a paraphrase reading as a lost detail."""
    w = w.strip("'")
    for suf, repl in (("'s", ""), ("ies", "y"), ("ing", ""), ("ed", ""),
                      ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)] + repl
    return w


def content_terms(text: str) -> Set[str]:
    """Stemmed content words -- the unit a 'detail' is counted in."""
    return {_stem(w) for w in
            (m.lower() for m in _WORD.findall(text))
            if w not in _STOP and len(w) > 2}


def detail_recall(fused: str, sources: Sequence[str]) -> float:
    """Fraction of the accounts' content terms that reach the fusion."""
    need: Set[str] = set()
    for s in sources:
        need |= content_terms(s)
    if not need:
        return 1.0
    return len(content_terms(fused) & need) / len(need)


def novel_content(fused: str, sources: Sequence[str]) -> float:
    """Fraction of the fusion's content terms present in no account."""
    have = content_terms(fused)
    if not have:
        return 0.0
    known: Set[str] = set()
    for s in sources:
        known |= content_terms(s)
    return len(have - known) / len(have)


def _counted_terms(text: str) -> "Counter":
    """Content terms WITH multiplicity; `content_terms` throws it away."""
    from collections import Counter
    return Counter(_stem(w) for w in (m.lower() for m in _WORD.findall(text))
                   if w not in _STOP and len(w) > 2)


def excess_repetition(fused: str, sources: Sequence[str]) -> float:
    """How often the fusion restates a content term beyond any single account.

    `internal_redundancy` compares 5-grams and therefore only sees repetition
    that is verbatim. The fusion's characteristic failure is not verbatim: in
    E060 the same prophecy arrives three times, once from each Evangelist --
    "not one stone here will be left on another; every one will be thrown
    down" beside "not one stone will be left on another; every one of them
    will be thrown down". Those two share every content term (Jaccard 1.00)
    and almost no 5-gram, so the event passed at 0.24 against a 0.25
    threshold while being, to a reader, the exact thing the task forbids.

    The ceiling is per term and taken from the accounts, not fixed: Mark
    says "buildings" twice on his own, so a fusion may too. What it may not
    do is say it a third time because Matthew also said it once.
    """
    cf = _counted_terms(fused)
    total = sum(cf.values())
    if not total:
        return 0.0
    ceiling: dict = {}
    for s in sources:
        for t, c in _counted_terms(s).items():
            ceiling[t] = max(ceiling.get(t, 0), c)
    return sum(max(0, c - ceiling.get(t, 0))
               for t, c in cf.items()) / total


def internal_redundancy(text: str, n: int = 5) -> float:
    """Token mass of `text` inside an n-gram `text` states more than once."""
    t = [w.lower() for w in _WORD.findall(text)]
    if len(t) < n:
        return 0.0
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    seen: dict = {}
    for g in grams:
        seen[g] = seen.get(g, 0) + 1
    marked = [False] * len(t)
    for i, g in enumerate(grams):
        if seen[g] > 1:
            for j in range(i, i + n):
                marked[j] = True
    return sum(marked) / len(t)


@dataclass
class Verdict:
    ok: bool
    recall: float
    novel: float
    redundancy: float = 0.0
    reason: str = ""

    def as_row(self) -> dict:
        return {"ok": self.ok, "detail_recall": round(self.recall, 4),
                "novel_content": round(self.novel, 4),
                "internal_redundancy": round(self.redundancy, 4),
                "reason": self.reason}


def check(fused: str, sources: Sequence[str],
          min_recall: float = MIN_DETAIL_RECALL,
          max_novel: float = MAX_NOVEL_CONTENT,
          max_redundancy: float = MAX_INTERNAL_REDUNDANCY) -> Verdict:
    """All three sides of the premise, as one verdict.

    Keep every detail, invent none, state each once. A check on any two of
    them has a gradient the third one has to hold: recall alone rewards
    concatenation, brevity alone rewards dropping a witness.
    """
    if not fused.strip():
        return Verdict(False, 0.0, 0.0, 0.0, "empty")
    r = detail_recall(fused, sources)
    n = novel_content(fused, sources)
    d = internal_redundancy(fused)
    # the accounts' own repetition is the floor: a command narrated and then
    # carried out in the same words is the corpus, not the fusion's doing
    floor = max([max_redundancy] + [internal_redundancy(s) for s in sources])
    why = []
    if r < min_recall:
        why.append(f"it dropped detail the accounts carry "
                   f"(recall {r:.2f} < {min_recall:.2f})")
    if n > max_novel:
        why.append(f"it contains material in no account "
                   f"(novel {n:.2f} > {max_novel:.2f})")
    if d > floor:
        why.append(f"it states the same wording twice "
                   f"(repeated {d:.2f} > {floor:.2f})")
    return Verdict(not why, r, n, d, "; ".join(why))
