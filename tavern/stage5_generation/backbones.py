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

#: `repetition_penalty` above is a HuggingFace `generate()` parameter: a flat
#: multiplicative penalty applied once per token already seen anywhere in the
#: sequence. llama.cpp's `repeat_penalty` (what Ollama's `/api/generate`
#: exposes) shares the name but not the semantics -- it penalises only tokens
#: within a trailing `repeat_last_n` window (default 64) and its documented
#: normal range is 1.0-1.2. `OllamaFuser` used to pass 1.5 straight through,
#: which is far outside that range and was suppressing common tokens,
#: including the whitespace token, producing glued-together words ("came
#: toBethphegeon theMountofOlves..."). This constant is Ollama's own, deliber-
#: ately NOT reused from `DECODING`: the HuggingFace backbones keep
#: repetition_penalty=1.5 unchanged, since that is the published work's fixed
#: decoding control (thesis Chapter 8) and changing it there would break
#: comparability. This is a backend-specific correctness fix, not a relaxation
#: of that requirement -- see `tavern/stage6_evaluation/text_quality.py` for
#: the regression guard.
OLLAMA_REPEAT_PENALTY = 1.1

#: Bumped whenever the prompt text below changes. `CachedFuser._key` digests
#: it: the cache is keyed by what determines the output, and the prompt
#: determines the output. Before this existed the key covered the backbone,
#: the model, the repeat penalty and the accounts but NOT the prompt, so
#: editing the prompt silently reused the old generations -- the docstring
#: claimed otherwise, which is how it went unnoticed.
PROMPT_VERSION = 3

#: Version 1 stated the two objectives and left the model to reconcile them:
#: "Keep every detail ... Do not summarise, shorten or omit" against "state it
#: once". gemma3:4b under greedy decoding reconciles them the cheapest way
#: available, by copying each account in turn and splicing the results --
#: measured over the `ancoragem` artifacts, 63% of multi-source events came
#: out at 80% or more of the summed length of their sources with 60% or more
#: of their tokens inside verbatim 8-grams, and E118 is three accounts joined
#: by semicolons with a capitalised "Now" mid-sentence. Version 2 states the
#: METHOD instead of the goal, shows one worked merge, and closes the two
#: failure modes the audit found (fabricated continuations, echoed "Account N"
#: labels) with explicit prohibitions rather than hoping the goal implies them.
#: Version 3 only reworded REPAIR_PREFIX, once the guard grew its third axis.
CONSOLIDATION_PROMPT = (
    "You will merge {n} accounts of the same event into a single paragraph.\n"
    "\n"
    "Method:\n"
    "1. Take the longest account as the spine and keep its wording.\n"
    "2. Read each other account in turn. Wherever it states something the "
    "spine does not, insert that detail into the spine at the point where it "
    "belongs.\n"
    "3. Where two accounts state the same thing in different words, keep the "
    "spine's wording and do not write it a second time.\n"
    "\n"
    "Rules:\n"
    "- Every detail from every account must appear in your paragraph, once.\n"
    "- Use only the facts and the wording of the accounts. Invent nothing: no "
    "names, places, numbers, motives or events.\n"
    "- Do not continue the story beyond what the accounts say.\n"
    "- Never write the word \"Account\", never number or attribute the "
    "accounts, never comment on them.\n"
    "{conflict}"
    "- Output the paragraph and nothing else.\n"
    "\n"
    "Example\n"
    "Account 1: He entered the house and sat down.\n"
    "Account 2: He went into the house, which belonged to Simon, and sat at "
    "the table.\n"
    "Paragraph: He entered the house, which belonged to Simon, and sat down "
    "at the table.\n"
    "\n"
    "Now do the same with these accounts.\n"
)

#: Inserted into the Rules block, not appended after the example: version 1
#: appended it, which under the current layout would put it after the worked
#: merge and the hand-off line.
CONFLICT_CLAUSE = (
    "- These accounts DISAGREE about the order or circumstances of the event. "
    "Keep both readings -- do not choose one and do not drop either -- and "
    "join them in one sentence with \"while\" or \"although\" rather than "
    "writing them as two separate accounts.\n"
)

#: The strict re-ask of the repair pass (`RepairingFuser`). Not a different
#: task: the same instruction with the failure named, which is the cheapest
#: intervention that can work and the only one that leaves the first pass's
#: measurement intact.
REPAIR_PREFIX = (
    "Your previous attempt was rejected because {reason}. Follow the method "
    "exactly: one spine, insert only what the other accounts add, never write "
    "the same thing twice, invent nothing.\n"
    "\n"
)


def build_prompt(n: int, conflicted: bool = False, strict: bool = False,
                 reason: str = "") -> str:
    """The instruction an instructable backbone receives."""
    p = CONSOLIDATION_PROMPT.format(
        n=n, conflict=CONFLICT_CLAUSE if conflicted else "")
    if strict:
        p = REPAIR_PREFIX.format(reason=reason or "it was not faithful to the "
                                 "accounts") + p
    return p


#: Stop sequences for the Ollama call. gemma3:4b given a short single account
#: continued past it and invented further "Account 2:" / "Account 3:" blocks
#: (E165 of the `ancoragem` artifacts invents two witnesses that do not
#: exist); the single-source short-circuit in `consolidate` removes most of
#: the opportunity and these close the rest, including the commentary the
#: model likes to add after a blank line.
OLLAMA_STOP = ["\nAccount", "\nParagraph:", "\n\n"]

#: Hard ceiling on a fusion, whatever the cluster: 4 accounts of the longest
#: verse span in the corpus come to ~470 tokens, so this cannot bind on a
#: legitimate merge and only stops a runaway.
OLLAMA_MAX_PREDICT = 512


def _budget(texts: Sequence[str]) -> int:
    """How many tokens a faithful fusion of `texts` can need.

    `DECODING["max_new_tokens"]` is a fixed 256 for the HuggingFace backbones
    and was passed straight through to Ollama's `num_predict`. A fixed cap is
    wrong in both directions here: it truncated 9 of the `ancoragem` events
    mid-word ("...and all on account of my name. This will result"), and on a
    one-line account it left 250 tokens of room that gemma3:4b filled with
    invention. A fusion is bounded by its own sources -- it may not add and
    should not need much more than the union of them -- so the budget is
    derived from them, with room for connective tissue and the mild
    word-to-token expansion of this corpus's proper nouns.
    """
    words = sum(len(t.split()) for t in texts)
    return max(64, min(OLLAMA_MAX_PREDICT, int(1.45 * words) + 48))


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
             context=None, strict: bool = False, reason: str = "") -> str:
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
             context=None, strict: bool = False, reason: str = "") -> str:
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
             context=None, strict: bool = False, reason: str = "") -> str:
        if not texts:
            return ""
        self._load()
        import torch
        prompt = build_prompt(len(texts), conflicted, strict, reason)
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
    decoding is greedy with the same length controls; the difference from
    `InstructFuser` is recorded rather than glossed. Repetition control is
    NOT shared with `InstructFuser`: see `OLLAMA_REPEAT_PENALTY` above for why
    `DECODING["repetition_penalty"]` (a HuggingFace parameter) cannot be
    reused for llama.cpp's `repeat_penalty` of the same name.
    """

    name = "ollama"
    instructable = True
    abstractive = True

    def __init__(self, model: str = "gemma3:4b",
                 endpoint: str = "http://localhost:11434/api/generate",
                 timeout: int = 180,
                 repeat_penalty: float = OLLAMA_REPEAT_PENALTY):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.repeat_penalty = repeat_penalty

    def available(self) -> bool:
        try:
            urllib.request.urlopen(
                self.endpoint.replace("/api/generate", "/api/tags"), timeout=5)
            return True
        except Exception:
            return False

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None, strict: bool = False, reason: str = "") -> str:
        if not texts:
            return ""
        prompt = build_prompt(len(texts), conflicted, strict, reason)
        body = "\n\n".join(f"Account {i + 1}: {t.strip()}"
                           for i, t in enumerate(texts))
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt + "\n" + body,
            "stream": False,
            # A reasoning model served through /api/generate spends the whole
            # `num_predict` budget in a `thinking` field that this endpoint
            # discards, and returns `response: ""` with `done_reason:
            # "length"` and no error. gemma4:26b does exactly that: every
            # event would come back empty, fail the guard twice and land in
            # the union fallback, and the run would look merely mediocre
            # rather than broken. Ignored by models that do not reason.
            "think": False,
            "options": {
                "num_predict": _budget(texts),
                "repeat_penalty": self.repeat_penalty,
                "temperature": 0.0,
                "seed": 0,
                "stop": OLLAMA_STOP,
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
class RepairingFuser:
    """A faithfulness floor under an abstractive backbone.

    The task's premise is checkable against the backbone's own inputs
    (`fidelity.check`): a fusion that has dropped a detail or invented one can
    be recognised without the chronology and without the reference. This
    wrapper does the recognising, and gives a failed fusion two further
    chances before it gives up on the model for that event:

      1. the same instruction with the failure named (`strict=True`);
      2. failing that, `UnionFuser`'s output for the same accounts, which is
         detail-preserving by construction.

    The fallback is not a silent downgrade -- the count is reported, and
    `consolidate` records per event which of the three produced the paragraph,
    so the abstractive fraction is always visible. The alternative, letting a
    fabricated paragraph through because ROUGE does not punish it, is what
    produced an aviation accident in the middle of the Olivet discourse.

    Single-account clusters never reach here: `consolidate` short-circuits
    them, since there is nothing to fuse and the account itself is the only
    faithful answer.
    """

    def __init__(self, inner, min_recall: float = None,
                 max_novel: float = None):
        from . import fidelity
        self._f = fidelity
        self.inner = inner
        self.name = getattr(inner, "name", "unknown")
        self.instructable = getattr(inner, "instructable", False)
        self.abstractive = getattr(inner, "abstractive", False)
        self.min_recall = (fidelity.MIN_DETAIL_RECALL if min_recall is None
                           else min_recall)
        self.max_novel = (fidelity.MAX_NOVEL_CONTENT if max_novel is None
                          else max_novel)
        self._union = UnionFuser()
        self.clean = 0
        self.repaired = 0
        self.fell_back = 0
        #: what produced the most recent fusion: "first", "repair" or "union"
        self.last_action = "first"

    #: the cache lives inside this wrapper, so its counters are read through
    #: it; `pipeline.run` reports them for the whole Stage 5 pass.
    @property
    def hits(self) -> int:
        return getattr(self.inner, "hits", 0)

    @property
    def misses(self) -> int:
        return getattr(self.inner, "misses", 0)

    def _ok(self, out: str, texts: Sequence[str]):
        return self._f.check(out, texts, self.min_recall, self.max_novel)

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None, strict: bool = False, reason: str = "") -> str:
        out = self.inner.fuse(texts, conflicted=conflicted, context=context)
        v = self._ok(out, texts)
        if v.ok:
            self.clean += 1
            self.last_action = "first"
            return out
        retry = self.inner.fuse(texts, conflicted=conflicted, context=context,
                                strict=True, reason=v.reason)
        if self._ok(retry, texts).ok:
            self.repaired += 1
            self.last_action = "repair"
            return retry
        self.fell_back += 1
        self.last_action = "union"
        return self._union.fuse(texts, conflicted=conflicted)

    def report(self) -> dict:
        total = self.clean + self.repaired + self.fell_back
        return {"clean": self.clean, "repaired": self.repaired,
                "union_fallback": self.fell_back, "total": total,
                "abstractive_fraction":
                    round((self.clean + self.repaired) / total, 4)
                    if total else 0.0}


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

    An abstractive backbone is then wrapped in `RepairingFuser`, outside the
    cache rather than inside it: both the first attempt and the strict re-ask
    are cached on their own keys, so a resumed run replays the same decisions
    instead of re-paying for them.
    """
    from . import ExtractiveFuser

    def _wrap(f):
        if not getattr(f, "abstractive", False):
            return f
        if cache_path:
            f = CachedFuser(f, cache_path)
        return RepairingFuser(f)

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

    The key is a digest of the backbone, the model name, the decoding
    `repeat_penalty` (where the inner fuser has one), `PROMPT_VERSION` and the
    repair pass (where the backbone is instructable), the accounts and the
    conflict flag, so a cache entry can only be reused for the identical call.
    Changing the prompt, the model, the clustering or the repetition penalty
    therefore invalidates exactly the entries it should and no others.

    `PROMPT_VERSION` was not in the key until the prompt was first revised:
    the docstring asserted the property, nothing tested it, and an edited
    prompt would have silently replayed the previous generations.
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

    def _key(self, texts: Sequence[str], conflicted: bool,
             strict: bool = False) -> str:
        import hashlib
        h = hashlib.sha1()
        h.update(self.name.encode())
        h.update(str(getattr(self.inner, "model",
                             getattr(self.inner, "model_name", ""))).encode())
        h.update(b"\x00")
        h.update(str(getattr(self.inner, "repeat_penalty", "")).encode())
        # the prompt determines the output, so it belongs in the key. Only
        # the instructable backbones see a prompt at all, hence the guard:
        # adding it unconditionally would needlessly invalidate every cached
        # BART/PEGASUS/PRIMERA fusion, which are prompt-free by design.
        if getattr(self.inner, "instructable", False):
            h.update(f"\x00p{PROMPT_VERSION}".encode())
            h.update(b"\x00strict" if strict else b"\x00first")
        h.update(b"\x00conflict" if conflicted else b"\x00plain")
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8", "replace"))
        return h.hexdigest()

    def fuse(self, texts: Sequence[str], conflicted: bool = False,
             context=None, strict: bool = False, reason: str = "") -> str:
        key = self._key(texts, conflicted, strict)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        text = self.inner.fuse(texts, conflicted=conflicted, context=context,
                               strict=strict, reason=reason)
        self._cache[key] = text
        self.misses += 1
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "text": text}) + "\n")
        return text
