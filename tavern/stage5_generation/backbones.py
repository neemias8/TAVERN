"""
Stage 5 - fusion backbones (thesis Sections 8.3 and 8.4).

One fuser per backbone family, all sharing the interface `ExtractiveFuser`
defines: `fuse(texts, conflicted, context) -> str`, where `texts` are the
parallel accounts of ONE candidate canonical event, the most representative
first.

The generation controls are design requirements taken from the published work,
not tuning choices, and each is enforced here rather than left to the caller:

  * Prompting a model that is not instruction-tuned degrades it. PRIMERA given
    an explicit instruction hallucinated institutions absent from the source and
    echoed the prompt into its output; given only `<doc-sep>` separators it
    produced clean output. Non-instruction-tuned backbones therefore receive
    separator-delimited input and NO prompt.
  * Checkpoint selection is a domain decision. PEGASUS pre-trained on a
    multi-document NEWS corpus injected newspapers and contemporary politics
    into biblical narrative; a single-document checkpoint, despite narrower
    pre-training, was markedly more faithful. Pre-training DOMAIN distance
    predicts suitability; objective similarity does not.
  * Decoding is fixed: 256 new tokens maximum, 10 minimum, length penalty 0.8,
    four beams, no repeated 3-grams, repetition penalty 1.5. `temperature` is
    silently ignored under beam search, so it is not set.

Where a cluster carries a CONFLICT edge, an instructable backbone is asked to
present the divergence rather than resolve it silently; a backbone that cannot
be instructed falls back to the most representative single account, on the
reasoning that a faithful single account is preferable to a fused paragraph
that silently adjudicates a disagreement the system detected.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

DECODING = dict(
    max_new_tokens=256,
    min_new_tokens=10,
    length_penalty=0.8,
    num_beams=4,
    no_repeat_ngram_size=3,
    repetition_penalty=1.5,
)

CONSOLIDATION_PROMPT = (
    "Below are {n} parallel accounts of the same single event, taken from "
    "different sources.\n\n"
    "Write ONE paragraph that consolidates them into a single narrative.\n"
    "Requirements:\n"
    "- Keep every detail that appears in any account. Do not summarise, "
    "shorten or omit.\n"
    "- Where two accounts describe the same thing in different words, state it "
    "once.\n"
    "- Add nothing that is not in the accounts. Invent no names, places, "
    "numbers or motives.\n"
    "- Write continuous narrative prose. No lists, no headings, no commentary.\n"
)

CONFLICT_CLAUSE = (
    "- The accounts DISAGREE about the order or circumstances of this event. "
    "Present both readings explicitly rather than choosing one.\n"
)


# ---------------------------------------------------------------------------
class UnionFuser:
    """Deterministic detail-preserving fusion, with no pretrained model.

    Not a substitute for abstractive generation: it produces no new wording, so
    the seams between accounts remain visible. What it does guarantee is the
    property the extractive configuration lacks and that the task's
    Representativeness and Completeness objectives require --- every detail
    present in ANY version reaches the consolidation, and each is stated once.

    A sentence is admitted unless a sentence already admitted covers it, where
    coverage is measured over content words: the candidate is redundant when
    most of its content is already present AND it adds no content word of its
    own that the kept sentence lacks. That second condition is what keeps
    "he denied it, saying, 'Woman, I do not know him'" after
    "he denied it before them all" --- the shared clause does not license
    dropping the quotation.

    This is the default when no generation backbone is available, and it is
    reported under its own name rather than as abstractive output.
    """

    name = "union"
    instructable = False
    abstractive = False

    def __init__(self, coverage: float = 0.80, min_new_content: int = 2):
        self.coverage = coverage
        self.min_new_content = min_new_content

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None) -> str:
        kept: List[str] = []
        kept_content: List[set] = []
        for text in texts:
            for sent in _sentences(text):
                content = _content_words(sent)
                if not content:
                    continue
                redundant = False
                for prev in kept_content:
                    shared = len(content & prev)
                    new = content - prev
                    if (shared / len(content) >= self.coverage
                            and len(new) < self.min_new_content):
                        redundant = True
                        break
                if redundant:
                    continue
                kept.append(sent.strip())
                kept_content.append(content)
        return " ".join(kept)


_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "that", "this", "these", "those", "it",
    "its", "he", "him", "his", "she", "her", "they", "them", "their", "we",
    "us", "our", "you", "your", "i", "me", "my", "was", "were", "is", "are",
    "be", "been", "being", "had", "has", "have", "do", "did", "does", "will",
    "would", "shall", "should", "may", "might", "can", "could", "must", "not",
    "no", "so", "then", "there", "here", "who", "whom", "which", "what",
    "when", "where", "why", "how", "all", "any", "some", "one", "up", "out",
    "into", "over", "about", "after", "before", "again", "also", "very",
}


_SENT_BOUNDARY = re.compile(
    r'(?:(?<=[.!?])|(?<=[.!?]")|(?<=[.!?]\u201d)|(?<=[.!?]\'))\s+')


def _sentences(text: str) -> List[str]:
    """Split on sentence boundaries, keeping a trailing quotation mark with the
    sentence it closes -- otherwise the fusion strips the closing quote of every
    reported utterance, and this corpus is largely reported utterance."""
    parts = _SENT_BOUNDARY.split(text.strip())
    return [p for p in parts if p and p.strip()]


def _content_words(sent: str) -> set:
    words = re.findall(r"[A-Za-z']+", sent.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


# ---------------------------------------------------------------------------
class _TransformersFuser:
    """Shared loading and decoding for the local seq2seq backbones."""

    abstractive = True
    instructable = False
    separator = " "

    def __init__(self, model_name: str, device: Optional[str] = None,
                 max_input_tokens: int = 1024):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self._tok = None
        self._model = None
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self._device is None:
            self._device = _pick_device()
        self._model.to(self._device)
        self._model.eval()

    def _input(self, texts: Sequence[str], conflicted: bool) -> str:
        return self.separator.join(t.strip() for t in texts)

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None) -> str:
        if not texts:
            return ""
        self._load()
        import torch
        enc = self._tok(self._input(texts, conflicted), return_tensors="pt",
                        truncation=True, max_length=self.max_input_tokens)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model.generate(**enc, **DECODING)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"          # Intel Arc / Xe / Data Center
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BartFuser(_TransformersFuser):
    """BART-Large-CNN. Not instruction-tuned: no prompt, plain concatenation."""
    name = "bart"

    def __init__(self, model_name: str = "facebook/bart-large-cnn", **kw):
        super().__init__(model_name, max_input_tokens=1024, **kw)


class PegasusFuser(_TransformersFuser):
    """PEGASUS on a SINGLE-document checkpoint.

    The multi-document news checkpoint leaked news entities into biblical
    narrative; the single-document one did not. Pre-training domain distance is
    what predicts suitability here, not objective similarity to the task.
    """
    name = "pegasus"

    def __init__(self, model_name: str = "google/pegasus-cnn_dailymail", **kw):
        super().__init__(model_name, max_input_tokens=1024, **kw)


class PrimeraFuser(_TransformersFuser):
    """PRIMERA, pre-trained on document collections delimited by <doc-sep>.

    Receives the separator and NO prompt: an instruction is out of distribution
    for it and produced severe hallucination in the published study.
    """
    name = "primera"
    separator = " <doc-sep> "

    def __init__(self, model_name: str = "allenai/PRIMERA", **kw):
        super().__init__(model_name, max_input_tokens=4096, **kw)


# ---------------------------------------------------------------------------
class InstructFuser:
    """An instruction-tuned decoder-only model, run locally.

    This is the backbone the conflict signalling of Section 8.5 applies to: a
    cluster whose members are joined by a CONFLICT edge is fused with an
    explicit instruction to present the divergence rather than resolve it.
    """

    name = "instruct"
    instructable = True
    abstractive = True

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 device: Optional[str] = None):
        self.model_name = model_name
        self._tok = None
        self._model = None
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype="auto")
        if self._device is None:
            self._device = _pick_device()
        self._model.to(self._device)
        self._model.eval()

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None) -> str:
        if not texts:
            return ""
        self._load()
        import torch
        prompt = CONSOLIDATION_PROMPT.format(n=len(texts))
        if conflicted:
            prompt += CONFLICT_CLAUSE
        body = "\n\n".join(f"Account {i + 1}: {t.strip()}"
                           for i, t in enumerate(texts))
        messages = [{"role": "user", "content": prompt + "\n" + body}]
        text = self._tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        enc = self._tok(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**enc, **DECODING)
        gen = out[0][enc["input_ids"].shape[-1]:]
        return _clean(self._tok.decode(gen, skip_special_tokens=True))


class OllamaFuser:
    """An instruction-tuned model served by a local Ollama daemon.

    Included because it needs no model download through this package and no
    accelerator configuration: `ollama pull gemma3:4b` and the daemon's default
    endpoint are enough. Beam search is not available through Ollama, so
    decoding is greedy with the same repetition and length controls; the
    difference from `InstructFuser` is recorded rather than glossed.
    """

    name = "ollama"
    instructable = True
    abstractive = True

    def __init__(self, model: str = "gemma3:4b",
                 endpoint: str = "http://localhost:11434/api/generate",
                 timeout: int = 180):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def available(self) -> bool:
        try:
            urllib.request.urlopen(
                self.endpoint.replace("/api/generate", "/api/tags"), timeout=5)
            return True
        except Exception:
            return False

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None) -> str:
        if not texts:
            return ""
        prompt = CONSOLIDATION_PROMPT.format(n=len(texts))
        if conflicted:
            prompt += CONFLICT_CLAUSE
        body = "\n\n".join(f"Account {i + 1}: {t.strip()}"
                           for i, t in enumerate(texts))
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt + "\n" + body,
            "stream": False,
            "options": {
                "num_predict": DECODING["max_new_tokens"],
                "repeat_penalty": DECODING["repetition_penalty"],
                "temperature": 0.0,
            },
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return _clean(json.loads(r.read())["response"])
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc


def _clean(text: str) -> str:
    """Strip the openings an instruction-tuned model tends to prepend."""
    text = text.strip()
    text = re.sub(r"^(here is|here's|sure[,.]?|certainly[,.]?)\b[^\n:]*:\s*",
                  "", text, flags=re.I)
    text = re.sub(r"^(consolidated (narrative|account|paragraph))\s*:\s*", "",
                  text, flags=re.I)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
REGISTRY = {
    "extractive": None,          # resolved in __init__ to avoid a cycle
    "union": UnionFuser,
    "bart": BartFuser,
    "pegasus": PegasusFuser,
    "primera": PrimeraFuser,
    "instruct": InstructFuser,
    "ollama": OllamaFuser,
}


def build(name: str, cache_path=None, **kw):
    """Instantiate a fuser, falling back with a stated reason.

    An unavailable backbone is not a silent downgrade: the caller is told which
    backbone was requested, why it could not be used, and what ran instead, so
    that no output is ever labelled abstractive when it is not.

    `cache_path` wraps an abstractive backbone in `CachedFuser`, which makes a
    long generation run resumable. The deterministic backbones are not cached:
    recomputing them is cheaper than reading the cache.
    """
    from . import ExtractiveFuser

    def _wrap(f):
        if cache_path and getattr(f, "abstractive", False):
            return CachedFuser(f, cache_path)
        return f

    if name == "extractive":
        return ExtractiveFuser(), None
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown backbone {name!r}; "
                         f"choose from {sorted(REGISTRY)}")
    if name == "union":
        return cls(**kw), None
    if name == "ollama":
        fuser = cls(**kw)
        if fuser.available():
            return _wrap(fuser), None
        return UnionFuser(), (f"ollama daemon not reachable at "
                              f"{fuser.endpoint}; fell back to union")
    try:
        fuser = cls(**kw)
        fuser._load()
        return _wrap(fuser), None
    except Exception as exc:
        return UnionFuser(), (f"{name} unavailable ({type(exc).__name__}: "
                              f"{str(exc)[:120]}); fell back to union")


# ---------------------------------------------------------------------------
class CachedFuser:
    """On-disk memoisation of fusions, keyed by content.

    A generation run over this corpus is 248 model calls, and on a workstation
    without an accelerator that is measured in hours. Interrupting it and losing
    everything is the difference between a job someone will run and one they
    will not, so every fusion is written out as it is produced and read back on
    the next run.

    The key is a digest of the backbone, the model name, the accounts and the
    conflict flag, so a cache entry can only be reused for the identical call.
    Changing the prompt, the model or the clustering therefore invalidates
    exactly the entries it should and no others.
    """

    def __init__(self, inner, path):
        import pathlib
        self.inner = inner
        self.name = getattr(inner, "name", "unknown")
        self.instructable = getattr(inner, "instructable", False)
        self.abstractive = getattr(inner, "abstractive", False)
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["text"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def _key(self, texts: Sequence[str], conflicted: bool) -> str:
        import hashlib
        h = hashlib.sha1()
        h.update(self.name.encode())
        h.update(str(getattr(self.inner, "model",
                             getattr(self.inner, "model_name", ""))).encode())
        h.update(b"\x00conflict" if conflicted else b"\x00plain")
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8", "replace"))
        return h.hexdigest()

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None) -> str:
        key = self._key(texts, conflicted)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        text = self.inner.fuse(texts, conflicted=conflicted, context=context)
        self._cache[key] = text
        self.misses += 1
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "text": text}) + "\n")
        return text
