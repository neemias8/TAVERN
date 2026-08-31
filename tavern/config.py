"""
Global configuration for the TAVERN framework.

Paths, corpus identity (content digests), and the one architectural rule the
thesis enforces in code: the chronology file is reachable only from Stage 6.
See Section 9.1 of the thesis (sec:setup-corpus).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SCHEMA_DIR = ROOT / "schema"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Corpus identity (thesis Table tab:task-md5)
# ---------------------------------------------------------------------------

GOSPEL_FILES = {
    "matthew": "EnglishNIVMatthew40_PW.xml",
    "mark": "EnglishNIVMark41_PW.xml",
    "luke": "EnglishNIVLuke42_PW.xml",
    "john": "EnglishNIVJohn43_PW.xml",
}

PERICOPE_FILES = {
    "matthew": "NIV_Matthew_PW_with_pericopes.xml",
    "mark": "NIV_Mark_PW_with_pericopes.xml",
    "luke": "NIV_Luke_PW_with_pericopes.xml",
    "john": "NIV_John_PW_with_pericopes.xml",
}

CHRONOLOGY_FILE = "ChronologyOfTheFourGospels_PW.xml"
GOLDEN_SAMPLE_FILE = "Golden_Sample.txt"

EXPECTED_MD5 = {
    "ChronologyOfTheFourGospels_PW.xml": "23cf52ee7c597b2aa190dc243d13f969",
    "EnglishNIVMatthew40_PW.xml": "8dea0ab56468c26dcb45c3160148c837",
    "EnglishNIVMark41_PW.xml": "003ba2b7cf7ca1b1af959f5352056ca8",
    "EnglishNIVLuke42_PW.xml": "0afe52402eb56d591406209c343ea4fb",
    "EnglishNIVJohn43_PW.xml": "2c7f59099ba90ef34666d23c6c480e6a",
    "Golden_Sample.txt": "dca94bbd4697c381c5d0f6859a658142",
}

# Canonical scope: Passion Week (thesis Section 4.2)
GOSPEL_SCOPE = {
    "matthew": (21, 28),
    "mark": (11, 16),
    "luke": (19, 24),
    "john": (12, 20),
}

BOOK_ORDER = ["matthew", "mark", "luke", "john"]

#: Short codes for xml:id construction. `book[:2]` would collide for
#: matthew/mark, which silently merges their annotation.
BOOK_CODE = {"matthew": "mt", "mark": "mk", "luke": "lk", "john": "jn"}

EXPECTED_VERSE_COUNTS = {"matthew": 389, "mark": 253, "luke": 285, "john": 318}
EXPECTED_TOTAL_VERSES = 1245
EXPECTED_CANONICAL_EVENTS = 169
EXPECTED_EVENT_VERSIONS = 363
EXPECTED_PERICOPES = 91

# ---------------------------------------------------------------------------
# Non-narrative discourse blocks (thesis Table tab:ann-discourse)
# Used by internal consistency check 5 (partition soundness).
# (book, start_chapter, start_verse, end_chapter, end_verse, label)
# ---------------------------------------------------------------------------

DISCOURSE_BLOCKS = [
    ("matthew", 23, 1, 23, 39, "Woes and lament"),
    ("matthew", 24, 1, 25, 46, "Olivet discourse"),
    ("matthew", 21, 28, 21, 44, "Parables of the two sons and the tenants"),
    ("matthew", 22, 1, 22, 14, "Parable of the banquet"),
    ("mark", 13, 1, 13, 37, "Olivet discourse"),
    ("mark", 12, 1, 12, 11, "Parable of the vineyard"),
    ("luke", 21, 5, 21, 36, "Olivet discourse"),
    ("luke", 20, 9, 20, 18, "Parable of the tenants"),
    ("john", 14, 1, 16, 33, "Farewell discourse"),
    ("john", 17, 1, 17, 26, "High-priestly prayer"),
    ("john", 12, 44, 12, 50, "Summary of the message"),
]

# ---------------------------------------------------------------------------
# Known-conflict regression suite (thesis Section 9.5, check 6)
# ---------------------------------------------------------------------------

KNOWN_CONFLICTS = {
    "fig_tree": {
        "label": "Timing of the fig tree (Matthew vs Mark)",
        "spans": [("matthew", 21, 18, 21, 22), ("mark", 11, 12, 11, 25)],
    },
    "passover_day": {
        "label": "Relation of the crucifixion to the Passover (John vs Synoptics)",
        "spans": [("john", 18, 28, 19, 16), ("mark", 14, 12, 14, 26),
                  ("luke", 22, 7, 22, 20), ("matthew", 26, 17, 26, 30)],
    },
    "cockcrow": {
        "label": "Sequence of the cockcrow (all four)",
        "spans": [("matthew", 26, 69, 26, 75), ("mark", 14, 66, 14, 72),
                  ("luke", 22, 54, 22, 62), ("john", 18, 15, 18, 27)],
    },
}


@dataclass
class TavernConfig:
    """Runtime configuration for a pipeline execution."""

    # Stage 2
    projection_mode: str = "relative"      # 'relative' | 'absolute'
    absolute_year: int = 30
    spacy_model: str = "en_core_web_sm"

    # Ablation switches (thesis Section 9.6)
    use_veridicality: bool = True
    use_closure: bool = True
    use_anchor_scaffold: bool = True
    use_graph_propagation: bool = True
    # Stage 3 granularity (locked by the sweep of Section 10.4)
    max_unit_verses: int = 6
    entity_turnover: float = 0.60
    max_episode_verses: Optional[int] = 2   # None = pericope boundary only
    anchor_band: float = 4.0
    gap_cost: float = 0.06
    match_threshold: float = 0.34
    # H-A / H-B, cluster-purity hypotheses (see event_coref.GATED_SCORE /
    # NO_ANCHOR_CREDIT). H-C is anchor_band above; test 1.0 there.
    gated_score: bool = False
    no_anchor_credit: bool = False
    cascade_levels: tuple = (1, 2, 3, 4)
    # Addendum 11 -- isolating which of Addendum 9's three changes produced
    # the ancoragem gain. Ablations over an already-adopted configuration;
    # none of them is ever the default and none is meant to be adopted.
    disable_projection: bool = False          # R1: scaffold.project_timexes never runs
    disable_projection_indexing: bool = False  # R2: projection runs, but D:/P: terms are not indexed
    legacy_participants: bool = False          # R3: pre-Addendum-9 _UBIQUITOUS_ENTITIES + bare Jaccard

    # Stage 4
    gnn_hidden: int = 128
    gnn_layers: int = 2
    gnn_heads: int = 4
    gnn_epochs: int = 200
    gnn_lr: float = 5e-3
    gnn_lambda_rel: float = 1.0
    seeds: tuple = (13, 42, 1337)

    # Stage 5
    # extractive | union | bart | pegasus | primera | instruct | ollama
    backbone: str = "union"
    backbone_model: str = ""               # override the checkpoint
    max_new_tokens: int = 256
    min_new_tokens: int = 10
    length_penalty: float = 0.8
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.5

    # Embeddings
    sbert_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    use_sbert: bool = True

    tag: str = "default"
    extra: dict = field(default_factory=dict)

    def run_dir(self) -> Path:
        d = OUTPUT_DIR / self.tag
        d.mkdir(parents=True, exist_ok=True)
        return d


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_corpus(strict: bool = True) -> dict:
    """Verify the content digests of Table tab:task-md5."""
    report = {}
    for name, expected in EXPECTED_MD5.items():
        path = DATA_DIR / name
        if not path.exists():
            report[name] = ("MISSING", expected, None)
            continue
        actual = md5_of(path)
        report[name] = ("OK" if actual == expected else "MISMATCH", expected, actual)
    if strict:
        bad = {k: v for k, v in report.items() if v[0] != "OK"}
        if bad:
            raise RuntimeError(f"Corpus digest verification failed: {bad}")
    return report


def assert_no_chronology_import() -> None:
    """Guard enforcing the separation of Section 9.1.

    Stages 1-5 must never read the chronology file. This is called by the
    Stage 1-5 entry points; it inspects the call stack of the loader instead of
    relying on convention.
    """
    import inspect

    forbidden = ("stage1_", "stage2_", "stage3_", "stage4_", "stage5_")
    for frame in inspect.stack():
        mod = frame.filename.replace(os.sep, "/")
        if any(f"/tavern/{p}" in mod for p in forbidden):
            raise RuntimeError(
                "The chronology file is reachable only from Stage 6 "
                f"(attempted load from {mod}). See thesis Section 9.1."
            )
