"""
Stage 6 - content metrics (thesis Section 9.3.1).

ROUGE-1/2/L F1 with stemming enabled, METEOR, and BERTScore F1. Where a figure
is compared with a published one the computation is matched exactly; in
particular all BERTScore figures are computed WITHOUT baseline rescaling,
matching the published benchmark, because rescaled and unrescaled figures are
not comparable and both appear in the prior literature on this corpus.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rouge_score import rouge_scorer

_SCORER = None
_BERT = None
_METEOR_READY = False


def _scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _SCORER


@dataclass
class ContentScores:
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    meteor: Optional[float] = None
    bertscore: Optional[float] = None
    length: int = 0

    def as_row(self) -> dict:
        return {
            "R-1": round(self.rouge1, 4),
            "R-2": round(self.rouge2, 4),
            "R-L": round(self.rougeL, 4),
            "METEOR": None if self.meteor is None else round(self.meteor, 4),
            "BERTScore": None if self.bertscore is None
            else round(self.bertscore, 4),
            "Length": self.length,
        }


def rouge(prediction: str, reference: str, fast: bool = True) -> Dict[str, float]:
    """ROUGE-1/2/L F1, identical to the reference implementation.

    The reference consolidation is some 14,000 tokens, and the pure-Python
    longest-common-subsequence table in `rouge_score` is quadratic in that
    length -- minutes per pair, which makes the ablation grid impractical. The
    fast path reproduces the reference implementation's tokenisation and Porter
    stemming exactly and delegates only the subsequence length to a compiled
    routine; `verify_fast_path` asserts the two agree.
    """
    if fast:
        try:
            return _rouge_fast(prediction, reference)
        except ImportError:
            pass
    s = _scorer().score(reference, prediction)
    return {"rouge1": s["rouge1"].fmeasure,
            "rouge2": s["rouge2"].fmeasure,
            "rougeL": s["rougeL"].fmeasure}


def _tokens(text: str) -> List[str]:
    """The tokenisation and stemming of `rouge_score.tokenize`."""
    from rouge_score import tokenize
    return tokenize.tokenize(text, _stemmer())


_STEMMER = None


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import porter
        _STEMMER = porter.PorterStemmer()
    return _STEMMER


def _f1(match: int, n_pred: int, n_ref: int) -> float:
    if not match or not n_pred or not n_ref:
        return 0.0
    p = match / n_pred
    r = match / n_ref
    return 2 * p * r / (p + r)


def _rouge_fast(prediction: str, reference: str) -> Dict[str, float]:
    import pylcs
    from collections import Counter

    pt = _tokens(prediction)
    rt = _tokens(reference)

    def ngrams(seq, n):
        return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

    out = {}
    for n, key in ((1, "rouge1"), (2, "rouge2")):
        pc, rc = ngrams(pt, n), ngrams(rt, n)
        match = sum(min(c, rc[g]) for g, c in pc.items())
        out[key] = _f1(match, sum(pc.values()), sum(rc.values()))

    vocab: Dict[str, int] = {}
    def encode(seq):
        chars = []
        for t in seq:
            if t not in vocab:
                # skip the surrogate range, which cannot appear in a str
                idx = len(vocab)
                vocab[t] = idx + (0x800 if idx < 0xD800 - 0x800 else 0x1000)
            chars.append(chr(vocab[t]))
        return "".join(chars)

    lcs = pylcs.lcs_sequence_length(encode(pt), encode(rt))
    out["rougeL"] = _f1(lcs, len(pt), len(rt))
    return out


def verify_fast_path(prediction: str, reference: str,
                     tol: float = 1e-9) -> Dict[str, float]:
    """Assert the fast path agrees with the reference implementation."""
    a = _rouge_fast(prediction, reference)
    s = _scorer().score(reference, prediction)
    b = {"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure,
         "rougeL": s["rougeL"].fmeasure}
    diffs = {k: abs(a[k] - b[k]) for k in a}
    bad = {k: v for k, v in diffs.items() if v > tol}
    if bad:
        raise AssertionError(f"fast ROUGE disagrees with rouge_score: {bad}")
    return diffs


def meteor(prediction: str, reference: str) -> Optional[float]:
    global _METEOR_READY
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        if not _METEOR_READY:
            # the container reaches PyPI and the NLTK corpora through a proxy;
            # NLTK refuses a proxied fetch unless told the proxy is trusted
            import os
            os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
            try:
                nltk.pathsec.ALLOW_PROXIED_FETCH = True
            except Exception:
                pass
            for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
            _METEOR_READY = True
        return float(meteor_score([reference.split()], prediction.split()))
    except Exception as exc:            # pragma: no cover
        warnings.warn(f"METEOR unavailable: {exc}")
        return None


def bertscore(prediction: str, reference: str,
              model_type: str = "roberta-large") -> Optional[float]:
    """BERTScore F1, computed WITHOUT baseline rescaling.

    The reference consolidation is far longer than any transformer's context, so
    the score is computed over aligned segments and averaged by segment length,
    which is how the published benchmark computes it.
    """
    global _BERT
    try:
        from bert_score import BERTScorer
        if _BERT is None:
            _BERT = BERTScorer(model_type=model_type, lang="en",
                               rescale_with_baseline=False, batch_size=8)
        cands, refs = _segment_pair(prediction, reference)
        P, R, F = _BERT.score(cands, refs)
        weights = [len(c.split()) for c in cands]
        total = sum(weights) or 1
        return float(sum(f * w for f, w in zip(F.tolist(), weights)) / total)
    except Exception as exc:            # pragma: no cover
        warnings.warn(f"BERTScore unavailable: {exc}")
        return None


def _segment_pair(prediction: str, reference: str, target_words: int = 300):
    """Split both texts into a matched number of proportional segments."""
    p = prediction.split()
    r = reference.split()
    n = max(1, round(max(len(p), len(r)) / target_words))
    def chunks(tokens):
        size = max(1, len(tokens) // n)
        out = [" ".join(tokens[i * size:(i + 1) * size]) for i in range(n)]
        if len(tokens) > n * size:
            out[-1] += " " + " ".join(tokens[n * size:])
        return [c if c.strip() else "." for c in out]
    return chunks(p), chunks(r)


def evaluate(prediction: str, reference: str, with_meteor: bool = True,
             with_bertscore: bool = True) -> ContentScores:
    r = rouge(prediction, reference)
    return ContentScores(
        rouge1=r["rouge1"], rouge2=r["rouge2"], rougeL=r["rougeL"],
        meteor=meteor(prediction, reference) if with_meteor else None,
        bertscore=bertscore(prediction, reference) if with_bertscore else None,
        length=len(prediction),
    )


def load_reference(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")
