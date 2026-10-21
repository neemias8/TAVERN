#!/usr/bin/env python3
"""
Stage 6 / Post-Hoc Stage 3 Diagnostic & Grouping Refinement Experiment.

    python scripts/posthoc_stage3_refinement.py
    python scripts/posthoc_stage3_refinement.py --json outputs/posthoc_stage3/refinement_report.json

Purpose
-------
Evaluates the post-hoc structural refinement of Stage 3 cross-document event
grouping motivated by the specialist evaluation (Question G) and cluster-purity
diagnostics (Section 10.9 of the thesis), without altering the frozen
`ancoragem` execution (digest `c903b9fd...1a4cd`) evaluated by the theologians.

Specifically, this script isolates and measures three non-destructive fixes to
Stage 3's alignment and pair-scoring pipeline (`tavern.stage3_anchoring_alignment`):

  1. Contiguous-span book overlap guard (`porte_4_correcoes`): prevents
     `_mergeable` from merging adjacent profile columns that share no source
     Gospel, so adjacency alone cannot manufacture cross-document identity.
  2. Semantic & content-word representation wiring: connects unit-level dense
     SVD/SBERT representations and content-word stem/bigram TF-IDF vectors into
     `predicate_similarity` (which remained `embeddings=None` in the frozen
     `pipeline.run` call, leaving `PredicateIDF` reliant on exact verb-lemma
     collisions such as `SIT` in Mark 12:36 vs. Matthew 23:22).
  3. Interpolated-anchor discounting: discounts `anchor_compatibility` when
     either unit's scaffold position is linearly interpolated rather than
     pinned by an observed narrative `<TIMEX3>`, preventing long unanchored
     discourses (such as John 13-16) from accumulating spurious temporal
     proximity credit against Synoptic narrative scenes.

Nothing in this script imports the external chronology into Stages 1-5;
`tavern.config.assert_no_chronology_import()` remains enforced throughout.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# Allow running even in lightweight CI/audit environments without networkx/spacy
_HAS_SPACY = True
try:
    import spacy
    spacy.load("en_core_web_sm")
except Exception:
    _HAS_SPACY = False
    if "spacy" not in sys.modules:
        sys.modules["spacy"] = types.ModuleType("spacy")

try:
    import networkx as nx  # noqa: F401
except ImportError:
    nx_mod = types.ModuleType("networkx")
    class MultiDiGraph:
        def __init__(self):
            self._nodes, self._edges = {}, []
        def add_node(self, n, **kw):
            self._nodes.setdefault(n, {}).update(kw)
        def add_edge(self, u, v, **kw):
            self._nodes.setdefault(u, {}); self._nodes.setdefault(v, {})
            self._edges.append((u, v, kw))
        def number_of_nodes(self): return len(self._nodes)
        def number_of_edges(self): return len(self._edges)
        def edges(self, data=False):
            return self._edges if data else [(u, v) for u, v, _ in self._edges]
        def nodes(self, data=False):
            return list(self._nodes.items()) if data else list(self._nodes.keys())
        def has_node(self, n): return n in self._nodes
    nx_mod.MultiDiGraph = MultiDiGraph
    sys.modules["networkx"] = nx_mod

import numpy as np

from tavern.config import BOOK_CODE, BOOK_ORDER, DATA_DIR, OUTPUT_DIR, TavernConfig
from tavern.stage1_preprocessing.coref import resolve_all, _PRONOUN_GROUPS
from tavern.stage1_preprocessing.corpus import Corpus
from tavern.stage1_preprocessing.pericopes import load_pericopes
from tavern.stage1_preprocessing.segmentation import Token, Sentence, SegmentedDocument
from tavern.stage2_temporal_annotation import annotate_corpus
from tavern.stage2_temporal_annotation.event_tagger import (
    REPORTING_LEMMAS, PERCEPTION_LEMMAS, ASPECTUAL_LEMMAS, I_STATE_LEMMAS,
    I_ACTION_LEMMAS, NOMINAL_EVENT_LEMMAS, TEMPORAL_NOUNS, COPULA_LEMMAS,
    MODAL_LEMMAS, NEGATION_LEMMAS
)
from tavern.stage3_anchoring_alignment import (
    event_coref,
    global_timeline,
    local_timeline,
    scaffold as scaffold_mod,
)
from tavern.stage6_evaluation import chronology as chrono_mod, timeline_eval
from tavern.stage6_evaluation.redundancy import _STOP, _TOKEN, _stem
import cluster_purity as cp

IRREGULAR_VERBS = {
    "was": ("be", "VBD"), "were": ("be", "VBD"), "been": ("be", "VBN"), "being": ("be", "VBG"),
    "am": ("be", "VBP"), "is": ("be", "VBZ"), "are": ("be", "VBP"),
    "had": ("have", "VBD"), "has": ("have", "VBZ"), "having": ("have", "VBG"),
    "did": ("do", "VBD"), "does": ("do", "VBZ"), "done": ("do", "VBN"), "doing": ("do", "VBG"),
    "said": ("say", "VBD"), "says": ("say", "VBZ"), "saying": ("say", "VBG"),
    "went": ("go", "VBD"), "gone": ("go", "VBN"), "going": ("go", "VBG"), "goes": ("go", "VBZ"),
    "came": ("come", "VBD"), "coming": ("come", "VBG"), "comes": ("come", "VBZ"),
    "saw": ("see", "VBD"), "seen": ("see", "VBN"), "seeing": ("see", "VBG"), "sees": ("see", "VBZ"),
    "took": ("take", "VBD"), "taken": ("take", "VBN"), "taking": ("take", "VBG"), "takes": ("take", "VBZ"),
    "gave": ("give", "VBD"), "given": ("give", "VBN"), "giving": ("give", "VBG"), "gives": ("give", "VBZ"),
    "told": ("tell", "VBD"), "telling": ("tell", "VBG"), "tells": ("tell", "VBZ"),
    "knew": ("know", "VBD"), "known": ("know", "VBN"), "knowing": ("know", "VBG"), "knows": ("know", "VBZ"),
    "found": ("find", "VBD"), "finding": ("find", "VBG"), "finds": ("find", "VBZ"),
    "began": ("begin", "VBD"), "begun": ("begin", "VBN"), "beginning": ("begin", "VBG"),
    "sent": ("send", "VBD"), "sending": ("send", "VBG"), "sends": ("send", "VBZ"),
    "brought": ("bring", "VBD"), "bringing": ("bring", "VBG"), "brings": ("bring", "VBZ"),
    "left": ("leave", "VBD"), "leaving": ("leave", "VBG"), "leaves": ("leave", "VBZ"),
    "put": ("put", "VBD"), "putting": ("put", "VBG"),
    "set": ("set", "VBD"), "setting": ("set", "VBG"),
    "sat": ("sit", "VBD"), "sitting": ("sit", "VBG"), "sits": ("sit", "VBZ"),
    "stood": ("stand", "VBD"), "standing": ("stand", "VBG"), "stands": ("stand", "VBZ"),
    "heard": ("hear", "VBD"), "hearing": ("hear", "VBG"), "hears": ("hear", "VBZ"),
    "spoke": ("speak", "VBD"), "spoken": ("speak", "VBN"), "speaking": ("speak", "VBG"),
    "replied": ("reply", "VBD"), "replies": ("reply", "VBZ"), "replying": ("reply", "VBG"),
    "asked": ("ask", "VBD"), "asking": ("ask", "VBG"), "asks": ("ask", "VBZ"),
    "answered": ("answer", "VBD"), "answering": ("answer", "VBG"), "answers": ("answer", "VBZ"),
    "called": ("call", "VBD"), "calling": ("call", "VBG"), "calls": ("call", "VBZ"),
    "cried": ("cry", "VBD"), "crying": ("cry", "VBG"), "cries": ("cry", "VBZ"),
    "fell": ("fall", "VBD"), "fallen": ("fall", "VBN"), "falling": ("fall", "VBG"),
    "led": ("lead", "VBD"), "leading": ("lead", "VBG"), "leads": ("lead", "VBZ"),
    "kept": ("keep", "VBD"), "keeping": ("keep", "VBG"), "keeps": ("keep", "VBZ"),
    "held": ("hold", "VBD"), "holding": ("hold", "VBG"), "holds": ("hold", "VBZ"),
    "made": ("make", "VBD"), "making": ("make", "VBG"), "makes": ("make", "VBZ"),
    "meant": ("mean", "VBD"), "meaning": ("mean", "VBG"), "means": ("mean", "VBZ"),
    "met": ("meet", "VBD"), "meeting": ("meet", "VBG"), "meets": ("meet", "VBZ"),
    "paid": ("pay", "VBD"), "paying": ("pay", "VBG"),
    "ran": ("run", "VBD"), "running": ("run", "VBG"),
    "rose": ("rise", "VBD"), "risen": ("rise", "VBN"), "rising": ("rise", "VBG"),
    "wrote": ("write", "VBD"), "written": ("write", "VBN"), "writing": ("write", "VBG"),
    "ate": ("eat", "VBD"), "eaten": ("eat", "VBN"), "eating": ("eat", "VBG"),
    "drank": ("drink", "VBD"), "drunk": ("drink", "VBN"), "drinking": ("drink", "VBG"),
    "broke": ("break", "VBD"), "broken": ("break", "VBN"), "breaking": ("break", "VBG"),
    "struck": ("strike", "VBD"), "striking": ("strike", "VBG"),
    "spit": ("spit", "VBD"), "spat": ("spit", "VBD"), "spitting": ("spit", "VBG"),
    "threw": ("throw", "VBD"), "thrown": ("throw", "VBN"), "throwing": ("throw", "VBG"),
    "hung": ("hang", "VBD"), "hanging": ("hang", "VBG"),
    "tore": ("tear", "VBD"), "torn": ("tear", "VBN"), "tearing": ("tear", "VBG"),
    "wove": ("weave", "VBD"), "woven": ("weave", "VBN"),
    "fled": ("flee", "VBD"), "fleeing": ("flee", "VBG"),
    "wept": ("weep", "VBD"), "weeping": ("weep", "VBG"),
    "slept": ("sleep", "VBD"), "sleeping": ("sleep", "VBG"),
    "woke": ("wake", "VBD"), "awoke": ("awake", "VBD"),
    "cut": ("cut", "VBD"), "cutting": ("cut", "VBG"),
    "spread": ("spread", "VBD"), "spreading": ("spread", "VBG"),
    "rode": ("ride", "VBD"), "ridden": ("ride", "VBN"), "riding": ("ride", "VBG"),
    "drove": ("drive", "VBD"), "driven": ("drive", "VBN"), "driving": ("drive", "VBG"),
    "overturned": ("overturn", "VBD"),
    "bought": ("buy", "VBD"), "buying": ("buy", "VBG"),
    "sold": ("sell", "VBD"), "selling": ("sell", "VBG"),
    "taught": ("teach", "VBD"), "teaching": ("teach", "VBG"),
    "caught": ("catch", "VBD"), "catching": ("catch", "VBG"),
    "sought": ("seek", "VBD"), "seeking": ("seek", "VBG"),
    "thought": ("think", "VBD"), "thinking": ("think", "VBG"),
    "fought": ("fight", "VBD"), "fighting": ("fight", "VBG"),
    "felt": ("feel", "VBD"), "feeling": ("feel", "VBG"),
    "built": ("build", "VBD"), "building": ("build", "VBG"),
    "lost": ("lose", "VBD"), "losing": ("lose", "VBG"),
    "won": ("win", "VBD"), "winning": ("win", "VBG"),
    "bound": ("bind", "VBD"), "binding": ("bind", "VBG"),
    "laid": ("lay", "VBD"), "laying": ("lay", "VBG"),
    "lain": ("lie", "VBN"), "lying": ("lie", "VBG"), "lay": ("lie", "VBD"),
    "chose": ("choose", "VBD"), "chosen": ("choose", "VBN"), "choosing": ("choose", "VBG"),
    "swore": ("swear", "VBD"), "sworn": ("swear", "VBN"), "swearing": ("swear", "VBG"),
    "bore": ("bear", "VBD"), "born": ("bear", "VBN"), "borne": ("bear", "VBN"), "bearing": ("bear", "VBG"),
    "forgave": ("forgive", "VBD"), "forgiven": ("forgive", "VBN"),
    "forsook": ("forsake", "VBD"), "forsaken": ("forsake", "VBN"),
    "betrayed": ("betray", "VBD"), "crucified": ("crucify", "VBD"),
    "denied": ("deny", "VBD"), "prophesied": ("prophesy", "VBD"),
    "carried": ("carry", "VBD"), "buried": ("bury", "VBD"),
    "tried": ("try", "VBD"), "hurried": ("hurry", "VBD"),
    "married": ("marry", "VBD"), "worried": ("worry", "VBD"),
}

ALL_KNOWN_VERBS = (
    REPORTING_LEMMAS | PERCEPTION_LEMMAS | set(ASPECTUAL_LEMMAS) |
    I_STATE_LEMMAS | I_ACTION_LEMMAS | COPULA_LEMMAS |
    {"go", "come", "take", "give", "find", "send", "bring", "leave", "put",
     "sit", "stand", "fall", "lead", "keep", "hold", "make", "meet", "pay",
     "run", "rise", "eat", "drink", "break", "strike", "spit", "throw", "hang",
     "tear", "flee", "weep", "sleep", "wake", "cut", "spread", "ride", "drive",
     "overturn", "buy", "sell", "catch", "fight", "build", "lose", "bind",
     "lay", "choose", "forgive", "forsake", "enter", "approach", "untie",
     "tie", "follow", "walk", "reach", "wither", "curse", "heal", "kill",
     "stone", "seize", "beat", "rent", "plant", "dig", "widen", "arrest",
     "flog", "mock", "strip", "clothe", "twist", "kneel", "crucify", "divide",
     "cast", "watch", "guard", "seal", "roll", "appear", "touch", "ascend",
     "wash", "wipe", "dip", "hand", "glorify", "betray", "deny", "crow",
     "warm", "kindle", "gather", "condemn", "release", "question",
     "amaze", "fear", "rejoice", "praise", "prepare", "pass",
     "happen", "occur", "die", "live", "save", "deliver", "suffer", "fulfill"}
)

DET_WORDS = {"the", "a", "an", "this", "that", "these", "those", "my", "your",
             "his", "her", "its", "our", "their", "some", "any", "no", "every",
             "each", "all", "both", "half", "many", "much", "few", "several"}
PREP_WORDS = {"in", "on", "at", "to", "for", "with", "from", "by", "of", "into",
              "unto", "upon", "out", "up", "down", "over", "under", "before",
              "after", "during", "through", "throughout", "within", "without",
              "among", "between", "against", "about", "around", "near", "beside",
              "behind", "beyond", "across", "along", "towards", "toward", "like", "as"}
SCONJ_WORDS = {"that", "if", "when", "while", "because", "since", "until", "till",
               "although", "though", "unless", "whether", "lest", "once", "whenever"}
CCONJ_WORDS = {"and", "but", "or", "nor", "yet", "so"}
ADV_WORDS = {"then", "now", "immediately", "soon", "shortly", "finally", "next",
             "first", "already", "still", "never", "not", "n't", "also", "even",
             "again", "away", "there", "here", "where", "how", "why",
             "very", "too", "only", "just", "later", "earlier", "afterwards",
             "afterward", "meanwhile", "previously", "beforehand", "truly", "early"}
PRON_WORDS = set(_PRONOUN_GROUPS) | {"i", "me", "my", "mine", "myself", "you", "your",
                                     "yours", "yourself", "yourselves", "we", "us",
                                     "ours", "ourselves", "it", "its", "itself",
                                     "who", "whom", "whose", "which", "what",
                                     "anyone", "someone", "everyone"}
NUM_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "hundred",
             "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
             "ninth", "tenth", "eleventh", "twelfth"}

_TOK_RE = re.compile(r"--|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")

def reg_lemmatize_verb(low: str):
    if low in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[low]
    if low in ALL_KNOWN_VERBS:
        return low, "VB"
    if low.endswith("ing") and len(low) > 4:
        stem = low[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            cand = stem[:-1]
            if cand in ALL_KNOWN_VERBS: return cand, "VBG"
        if (stem + "e") in ALL_KNOWN_VERBS:
            return stem + "e", "VBG"
        if stem in ALL_KNOWN_VERBS:
            return stem, "VBG"
    if low.endswith("ied") and len(low) > 4:
        cand = low[:-3] + "y"
        return cand, "VBD"
    if low.endswith("ed") and len(low) > 3:
        stem = low[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            cand = stem[:-1]
            return cand, "VBD"
        if (stem + "e") in ALL_KNOWN_VERBS or low.endswith(("ted", "ded", "ked", "ped", "sed", "ced", "ged", "led", "red", "ned", "med", "ved", "zed")):
            if (stem + "e") in ALL_KNOWN_VERBS:
                return stem + "e", "VBD"
        return stem, "VBD"
    if low.endswith("es") and len(low) > 3:
        stem = low[:-2]
        if stem in ALL_KNOWN_VERBS: return stem, "VBZ"
    if low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
        stem = low[:-1]
        if stem in ALL_KNOWN_VERBS: return stem, "VBZ"
    return None

def segment_corpus_pure(corpus: Corpus) -> dict:
    out_docs = {}
    for book in corpus.books:
        doc = SegmentedDocument(book=book)
        for verse in corpus[book].verses:
            # Split into sentences on . ! ? followed by space
            raw_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"\'“‘])', verse.text) if s.strip()]
            if not raw_sents:
                raw_sents = [verse.text]
            idx_in_verse = 0
            for si, s_text in enumerate(raw_sents):
                sent_id = f"s_{BOOK_CODE.get(book, book[:2])}{verse.chapter}_{verse.number}_{si}"
                raw_toks = _TOK_RE.findall(s_text)
                # Split contractions like don't -> do, n't
                expanded = []
                for rt in raw_toks:
                    if rt.lower().endswith("n't") and len(rt) > 3:
                        expanded.append(rt[:-3])
                        expanded.append("n't")
                    elif "'" in rt and rt.lower().endswith(("'s", "'ll", "'ve", "'re", "'d", "'m")):
                        idx = rt.rfind("'")
                        expanded.append(rt[:idx])
                        expanded.append(rt[idx:])
                    else:
                        expanded.append(rt)
                toks = []
                prev_low = ""
                for ti, w in enumerate(expanded):
                    low = w.lower()
                    xid = f"tk_{BOOK_CODE.get(book, book[:2])}{verse.chapter}_{verse.number}_{si}_{ti}"
                    is_punct = bool(re.match(r"^[^\w\s]+$", w))
                    pos, tag, lemma, dep, morph = "NOUN", "NN", low, "dep", ""
                    if is_punct:
                        pos, tag, dep = "PUNCT", ".", "punct"
                    elif low in MODAL_LEMMAS:
                        pos, tag, lemma, dep = "AUX", "MD", low, "aux"
                    elif low in ("n't", "not", "never"):
                        pos, tag, lemma, dep = "PART" if low=="n't" else "ADV", "RB", "not" if low=="n't" else low, "neg"
                    elif low == "to" and ti + 1 < len(expanded) and expanded[ti+1].lower() in ALL_KNOWN_VERBS:
                        pos, tag, lemma, dep = "PART", "TO", "to", "aux"
                    elif low in SCONJ_WORDS:
                        pos, tag, lemma, dep = "SCONJ", "IN", low, "mark"
                    elif low in PREP_WORDS:
                        pos, tag, lemma, dep = "ADP", "IN", low, "prep"
                    elif low in CCONJ_WORDS:
                        pos, tag, lemma, dep = "CCONJ", "CC", low, "cc"
                    elif low in ADV_WORDS:
                        pos, tag, lemma, dep = "ADV", "RB", low, "advmod"
                    elif low in PRON_WORDS:
                        pos, tag, lemma, dep = "PRON", "PRP", low, "nsubj"
                    elif low in DET_WORDS:
                        pos, tag, lemma, dep = "DET", "DT", low, "det"
                    elif low in NUM_WORDS or low.isdigit():
                        pos, tag, lemma, dep = "NUM", "CD", low, "nummod"
                    else:
                        v_res = reg_lemmatize_verb(low)
                        # Check if preceded by determiner/adjective -> likely NOUN if in NOMINAL_EVENT_LEMMAS or TEMPORAL_NOUNS
                        if prev_low in ("the", "a", "an", "his", "their", "your", "my", "this", "that") and (low in NOMINAL_EVENT_LEMMAS or low in TEMPORAL_NOUNS or low.rstrip("s") in TEMPORAL_NOUNS or low.rstrip("s") in NOMINAL_EVENT_LEMMAS):
                            lemma = low.rstrip("s") if (low.endswith("s") and not low.endswith("ss") and low.rstrip("s") in (NOMINAL_EVENT_LEMMAS | TEMPORAL_NOUNS)) else low
                            pos, tag, dep = "NOUN", "NNS" if low.endswith("s") else "NN", "dobj"
                        elif v_res is not None:
                            lemma, tag = v_res
                            if lemma in ("be", "have", "do") and ti + 1 < len(expanded) and reg_lemmatize_verb(expanded[ti+1].lower()) is not None:
                                pos, dep = "AUX", "aux"
                            elif lemma == "be":
                                pos, dep = "AUX", "ROOT"
                            else:
                                pos, dep = "VERB", "ROOT"
                            if tag == "VBD": morph = "Tense=Past"
                            elif tag in ("VBZ", "VBP"): morph = "Tense=Pres"
                        elif w[0].isupper() and ti > 0:
                            pos, tag, lemma, dep = "PROPN", "NNP", low, "nsubj"
                        else:
                            if low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
                                lemma = low[:-1]
                                tag = "NNS"
                            pos, dep = "NOUN", "dobj"
                    toks.append(Token(
                        xml_id=xid, text=w, lemma=lemma, pos=pos, tag=tag,
                        dep=dep, head=ti, morph=morph, idx_in_verse=idx_in_verse,
                        verse_key=verse.key, sent_id=sent_id, is_punct=is_punct
                    ))
                    if not is_punct:
                        prev_low = low
                    idx_in_verse += 1
                # Assign heads: find main verbs in sentence and link auxiliaries/negation/clauses
                verb_indices = [i for i, t in enumerate(toks) if t.pos == "VERB" or (t.pos == "AUX" and t.lemma == "be" and t.dep == "ROOT")]
                if verb_indices:
                    first_v = verb_indices[0]
                    for i, t in enumerate(toks):
                        if i in verb_indices:
                            if i > first_v:
                                t.dep = "ccomp" if toks[first_v].lemma in (REPORTING_LEMMAS | PERCEPTION_LEMMAS | I_STATE_LEMMAS | I_ACTION_LEMMAS) else "conj"
                                t.head = first_v
                            else:
                                t.head = i
                        elif t.dep in ("aux", "neg"):
                            # attach to next verb
                            nxt_v = next((v for v in verb_indices if v > i), verb_indices[-1])
                            t.head = nxt_v
                        else:
                            # attach to nearest verb
                            nearest_v = min(verb_indices, key=lambda v: abs(v - i))
                            t.head = nearest_v
                            if t.pos == "NOUN" and nearest_v < i and toks[nearest_v].lemma == "be":
                                t.dep = "attr"
                doc.add_sentence(Sentence(sent_id=sent_id, verse_key=verse.key, tokens=toks))
        out_docs[book] = doc
    return out_docs


# The four multi-source items in `avaliacao_humana/amostra_congelada.json`
# where Question G / purity diagnostics identified an intruding passage in the
# frozen `ancoragem` run.
THEOLOGIAN_SAMPLE_CASES = {
    "A01 (Mk 12:35-36 vs Mt 23:22)": [
        ("mark", 12, 35),
        ("matthew", 23, 22),
    ],
    "A02 (Jn 13:29 vs Lk 22:29-30)": [
        ("john", 13, 29),
        ("luke", 22, 29),
    ],
    "A04 (Mt 26:33 / Mk 14:29 / Lk 22:32-33 vs Jn 13:31)": [
        ("matthew", 26, 33),
        ("mark", 14, 29),
        ("luke", 22, 33),
        ("john", 13, 31),
    ],
    "A07 (Mt 26:69-70 / Mk 14:66 / Lk 22:55-57 vs Jn 16:16)": [
        ("matthew", 26, 69),
        ("mark", 14, 66),
        ("luke", 22, 55),
        ("john", 16, 16),
    ],
}


def build_unit_representations(
    units_dict: Dict[str, local_timeline.EventUnit]
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """Compute deterministic content-word + bigram TF-IDF vectors and 96-dim
    LSA/SVD dense unit embeddings from the source text of the 567 units alone
    (zero external chronology access).
    """
    uids = sorted(units_dict.keys())
    doc_terms: Dict[str, Counter] = {}
    df: Counter = Counter()

    for uid in uids:
        text = units_dict[uid].text.lower()
        words = [
            _stem(m.group(0))
            for m in _TOKEN.finditer(text)
            if len(m.group(0)) > 2 and m.group(0).lower() not in _STOP
        ]
        c = Counter(words)
        for w1, w2 in zip(words, words[1:]):
            c[f"BG:{w1}_{w2}"] += 1
        doc_terms[uid] = c
        df.update(set(c.keys()))

    n = len(uids)
    vocab = [t for t, count in df.items() if count >= 2]
    t2i = {t: i for i, t in enumerate(vocab)}
    mat = np.zeros((n, len(vocab)), dtype=np.float64)

    for r, uid in enumerate(uids):
        c = doc_terms[uid]
        for t, cnt in c.items():
            if t in t2i:
                idf = math.log((n + 1) / (1 + df[t]))
                mat[r, t2i[t]] = (1.0 + math.log(cnt)) * idf
        norm = np.linalg.norm(mat[r])
        if norm > 0:
            mat[r] /= norm

    u_svd, s_svd, _ = np.linalg.svd(mat, full_matrices=False)
    k = min(96, mat.shape[1])
    dense = u_svd[:, :k] * np.sqrt(s_svd[:k])

    emb: Dict[str, List[float]] = {}
    for r, uid in enumerate(uids):
        norm = np.linalg.norm(dense[r])
        emb[uid] = (dense[r] / norm).tolist() if norm > 0 else dense[r].tolist()

    sparse: Dict[str, Dict[str, float]] = {}
    for uid in uids:
        c = doc_terms[uid]
        vec = {
            t: (1.0 + math.log(cnt)) * math.log((n + 1) / (1 + df[t]))
            for t, cnt in c.items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        sparse[uid] = {t: v / norm for t, v in vec.items()}

    return emb, sparse


def load_locked_timelines_from_curation(
    structs, corpus: Corpus, pericopes, chains, curation_csv: Path
) -> Dict[str, local_timeline.LocalTimeline]:
    """Reconstruct the exact 567 source units from `consolidations/curation.csv`
    so Stage 3 alignment and scoring variants are compared on the identical
    unit segmentation of the frozen `ancoragem` run.
    """
    spans_by_book: Dict[str, List[List[Tuple[str, int, int]]]] = defaultdict(list)
    seen_spans = set()
    with open(curation_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for part in row["sources"].split(";"):
                part = part.strip()
                if not part:
                    continue
                m = re.match(
                    r"^(?:(Matthew|Mark|Luke|John)\s+)?(Matthew|Mark|Luke|John)\s+(.+)$",
                    part,
                )
                if m:
                    gospel = m.group(2).lower()
                    keys = tuple(corpus.span_keys(gospel, m.group(3)))
                    if (gospel, keys) not in seen_spans:
                        seen_spans.add((gospel, keys))
                        spans_by_book[gospel].append(list(keys))

    timelines: Dict[str, local_timeline.LocalTimeline] = {}
    for book in BOOK_ORDER:
        struct = structs[book]
        tl = local_timeline.LocalTimeline(book=book)
        events_by_verse = defaultdict(list)
        for ev in struct.events.values():
            if ev.verse_key:
                events_by_verse[ev.verse_key].append(ev)
        timex_by_verse = defaultdict(list)
        for tx in struct.timexes.values():
            if tx.verse_key:
                timex_by_verse[tx.verse_key].append(tx)
        signals_by_verse = defaultdict(list)
        for sg in struct.signals.values():
            if sg.verse_key:
                signals_by_verse[sg.verse_key].append(sg)

        book_spans = sorted(spans_by_book[book], key=lambda ks: (ks[0][1], ks[0][2]))
        for n, ks in enumerate(book_spans, start=1):
            uid = f"{BOOK_CODE[book]}_u{n:03d}"
            pid = pericopes.of_verse(ks[0])
            u = local_timeline._build_unit(
                uid,
                book,
                pid,
                ks,
                struct,
                corpus,
                events_by_verse,
                timex_by_verse,
                signals_by_verse,
                chains[book],
            )
            tl.units.append(u)
            for eid in u.event_ids:
                tl.unit_of_event[eid] = uid
        local_timeline._induce_order(tl, struct)
        timelines[book] = tl
    return timelines


def evaluate_variant(
    label: str,
    structs,
    timelines: Dict[str, local_timeline.LocalTimeline],
    units: Dict[str, local_timeline.EventUnit],
    ch,
    v2e: Dict[str, int],
    e2day: Dict[int, str],
    embeddings: Optional[Dict[str, List[float]]] = None,
    sparse_lex: Optional[Dict[str, Dict[str, float]]] = None,
    thresh: float = 0.34,
    emb_weight: float = 0.0,
    lex_weight: float = 0.0,
    interp_discount: float = 1.0,
    strict_merge: bool = False,
) -> dict:
    orig_pred = event_coref.predicate_similarity
    orig_anc = event_coref.anchor_compatibility
    orig_merge = event_coref._mergeable
    orig_thresh = event_coref.MATCH_THRESHOLD

    if emb_weight > 0.0 or lex_weight > 0.0:
        def custom_pred(a, b, vectors, emb=None):
            iso_v = event_coref._cosine(vectors[a.unit_id], vectors[b.unit_id])
            lex_v = (
                event_coref._cosine(sparse_lex[a.unit_id], sparse_lex[b.unit_id])
                if sparse_lex is not None
                else 0.0
            )
            emb_v = 0.0
            if emb is not None:
                ea, eb = emb.get(a.unit_id), emb.get(b.unit_id)
                if ea is not None and eb is not None:
                    emb_v = max(0.0, sum(x * y for x, y in zip(ea, eb)))
            w_iso = max(0.0, 1.0 - emb_weight - lex_weight)
            return w_iso * iso_v + lex_weight * lex_v + emb_weight * emb_v

        event_coref.predicate_similarity = custom_pred

    if interp_discount < 1.0:
        def custom_anc(a, b, scaffold):
            val = orig_anc(a, b, scaffold)
            if not (scaffold.is_observed(a.unit_id) and scaffold.is_observed(b.unit_id)):
                val *= interp_discount
            return val

        event_coref.anchor_compatibility = custom_anc

    if strict_merge:
        def custom_merge(cur, nxt, units_d, succ):
            if not ({units_d[u].book for u in cur} & {units_d[u].book for u in nxt}):
                return False
            return orig_merge(cur, nxt, units_d, succ)

        event_coref._mergeable = custom_merge

    event_coref.MATCH_THRESHOLD = thresh
    event_coref.GATED_SCORE = False
    event_coref.NO_ANCHOR_CREDIT = False
    event_coref.ANCHOR_BAND = 4.0
    event_coref.GAP_COST = 0.06
    event_coref.MAX_EPISODE_VERSES = 2

    sc = scaffold_mod.build(structs, timelines, enabled=True)
    for u in units.values():
        u.projected_days.clear()
        u.projected_parts.clear()
    scaffold_mod.project_timexes(structs, sc, units)

    cl = event_coref.cluster_units(timelines, sc, embeddings=embeddings)
    ind = global_timeline.induce(timelines, cl, sc, structs=structs)
    ev = timeline_eval.evaluate(ch, cl, ind, units)

    events_list = []
    for idx, cid in enumerate(ind.order, start=1):
        c_obj = cl.by_id(cid)
        srcs_by_gospel = {}
        for uid in c_obj.members:
            u = units[uid]
            g = u.book
            verses = [f"{b}:{c}:{v}" for b, c, v in u.verse_keys]
            if g not in srcs_by_gospel:
                srcs_by_gospel[g] = {"gospel": g, "ref": u.ref, "verses": []}
            else:
                srcs_by_gospel[g]["ref"] += "; " + u.ref
            srcs_by_gospel[g]["verses"].extend(verses)
        events_list.append(
            {"marker": f"E{idx:03d}:{cid}", "sources": list(srcs_by_gospel.values())}
        )

    out_dir = OUTPUT_DIR / "posthoc_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"curation_{label}.json"
    tmp_path.write_text(json.dumps({"events": events_list}, indent=1), encoding="utf-8")
    pur = cp.analyse(tmp_path, v2e, e2day)
    b3 = pur["bcubed"]

    # Inspect the 4 theologian sample cases
    v_to_cid = {}
    cid_to_refs = {}
    for c_obj in cl.clusters:
        cid_to_refs[c_obj.cluster_id] = [units[m].ref for m in c_obj.members]
        for m in c_obj.members:
            for vk in units[m].verse_keys:
                v_to_cid[vk] = c_obj.cluster_id

    sample_audit = {}
    for case_name, vks in THEOLOGIAN_SAMPLE_CASES.items():
        sample_audit[case_name] = {
            f"{vk[0]}:{vk[1]}:{vk[2]}": {
                "cluster_id": v_to_cid.get(vk),
                "members": cid_to_refs.get(v_to_cid.get(vk, ""), []),
            }
            for vk in vks
        }

    # Restore module functions
    event_coref.predicate_similarity = orig_pred
    event_coref.anchor_compatibility = orig_anc
    event_coref._mergeable = orig_merge
    event_coref.MATCH_THRESHOLD = orig_thresh

    row = {
        "config": label,
        "clusters": len(cl.clusters),
        "multi_witness_clusters": pur["multi_witness_clusters"],
        "pure_clusters": pur["pure"],
        "impure_clusters": pur["impure"],
        "purity": pur["purity"],
        "day_mixing": pur["day_mixing"],
        "tau_plan": round(ev.tau, 4) if ev.tau is not None else None,
        "pairwise_accuracy": round(ev.pairwise_accuracy, 4)
        if ev.pairwise_accuracy is not None
        else None,
        "matched_events": ev.matched_events,
        "total_events": ev.total_events,
        "coverage": round(ev.coverage, 4),
        "bcubed": b3,
        "theologian_sample_cases": sample_audit,
    }
    print(
        f"  {label:28s} | clusters={row['clusters']:3d} (multi={row['multi_witness_clusters']:3d}) | "
        f"tau={row['tau_plan']:.4f} cov={row['matched_events']}/{row['total_events']} ({row['coverage']:.4f}) | "
        f"Purity={row['purity']:.4f} ({row['pure_clusters']}/{row['multi_witness_clusters']}) "
        f"DayMix={row['day_mixing']:.4f} | B3 P={b3['precision']:.4f} R={b3['recall']:.4f} F1={b3['f1']:.4f}"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--json",
        default=str(OUTPUT_DIR / "posthoc_stage3" / "refinement_report.json"),
        help="Path to write the JSON comparison report.",
    )
    args = ap.parse_args()

    cfg = TavernConfig(tag="posthoc_stage3")
    corpus = Corpus()
    pericopes = load_pericopes(corpus)

    if _HAS_SPACY:
        from tavern import pipeline
        corpus, pericopes, segments, chains, structs, _ = pipeline.prepare(
            cfg, verify=True
        )
    else:
        segments = segment_corpus_pure(corpus)
        chains = resolve_all(segments)
        structs, _ = annotate_corpus(segments, pericopes, cfg, write=False)

    curation_csv = ROOT / "consolidations" / "curation.csv"
    if curation_csv.exists():
        timelines = load_locked_timelines_from_curation(
            structs, corpus, pericopes, chains, curation_csv
        )
    else:
        timelines = local_timeline.segment_corpus(structs, corpus, pericopes, chains)

    units = {u.unit_id: u for tl in timelines.values() for u in tl.units}
    embeddings, sparse_lex = build_unit_representations(units)

    ch = chrono_mod.load(corpus)
    v2e, e2day = cp.verse_to_event(DATA_DIR)

    print("=" * 118)
    print("POST-HOC STAGE 3 GROUPING REFINEMENT BENCHMARK (567 Source Units)")
    print("=" * 118)

    results = []
    results.append(
        evaluate_variant(
            "0_baseline_unpatched",
            structs,
            timelines,
            units,
            ch,
            v2e,
            e2day,
            embeddings=None,
            sparse_lex=None,
            thresh=0.34,
            strict_merge=False,
        )
    )
    results.append(
        evaluate_variant(
            "1_porte4_plus_embeddings",
            structs,
            timelines,
            units,
            ch,
            v2e,
            e2day,
            embeddings=embeddings,
            sparse_lex=None,
            thresh=0.34,
            emb_weight=0.40,
            lex_weight=0.0,
            interp_discount=1.0,
            strict_merge=True,
        )
    )
    results.append(
        evaluate_variant(
            "2_posthoc_refinement_th033",
            structs,
            timelines,
            units,
            ch,
            v2e,
            e2day,
            embeddings=embeddings,
            sparse_lex=sparse_lex,
            thresh=0.33,
            emb_weight=0.40,
            lex_weight=0.25,
            interp_discount=0.50,
            strict_merge=True,
        )
    )
    results.append(
        evaluate_variant(
            "3_posthoc_refinement_th034",
            structs,
            timelines,
            units,
            ch,
            v2e,
            e2day,
            embeddings=embeddings,
            sparse_lex=sparse_lex,
            thresh=0.34,
            emb_weight=0.40,
            lex_weight=0.25,
            interp_discount=0.60,
            strict_merge=True,
        )
    )

    print("\n" + "=" * 118)
    print("RESOLUTION OF THE 4 IMPURE ITEMS IN THE THEOLOGIAN SAMPLE (3_posthoc_refinement_th034)")
    print("=" * 118)
    best_audit = results[-1]["theologian_sample_cases"]
    for case_name, mapping in best_audit.items():
        print(f"  {case_name}:")
        for vref, info in mapping.items():
            print(f"    {vref:18s} -> {info['cluster_id']} {info['members']}")

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved full JSON report to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())