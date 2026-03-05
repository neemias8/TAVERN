"""
TAVERN — Utility helpers
========================
Provides XML parsing, embedding utilities, graph creation, verse‐reference
parsing, and optional Sentence‐BERT support used across all pipeline stages.
"""

import xml.etree.ElementTree as ET
import re
import numpy as np
import networkx as nx
import hashlib
from typing import List, Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_gospel_xml(file_path: str) -> List[Dict[str, Any]]:
    """Parse a Gospel XML file and return a list of verse dicts with chapter,
    verse number, and text."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    verses_with_metadata: List[Dict[str, Any]] = []

    for chapter in root.iter('chapter'):
        chapter_num = int(chapter.get('number', 0))
        for verse in chapter.iter('verse'):
            verse_num = int(verse.get('number', 0))
            if verse.text:
                verses_with_metadata.append({
                    'text': verse.text.strip(),
                    'chapter': chapter_num,
                    'verse': verse_num
                })

    return verses_with_metadata


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def simple_word_embedding(text: str, dim: int = 300) -> np.ndarray:
    """Compute a deterministic, lightweight hashed bag-of-words embedding.

    Uses a signed hashing trick into a fixed-size vector and L2‐normalises it.
    No external models required — works fully offline.
    """
    if not text:
        return np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 1) & 1) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# --- Optional Sentence‐BERT wrapper ----------------------------------------

_sbert_model = None

def _ensure_sbert():
    """Lazy-load a Sentence-BERT model.  Returns the model or ``None``."""
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _sbert_model
    except Exception:
        return None


def sbert_embedding(text: str) -> Optional[np.ndarray]:
    """Return a 384-d Sentence‐BERT embedding, or ``None`` on failure."""
    model = _ensure_sbert()
    if model is None:
        return None
    try:
        vec = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.astype(np.float32)
    except Exception:
        return None


def get_embedding(text: str, dim: int = 300) -> np.ndarray:
    """Get an embedding for *text*, preferring Sentence-BERT if available,
    falling back to the deterministic hash embedding."""
    vec = sbert_embedding(text)
    if vec is not None:
        return vec
    return simple_word_embedding(text, dim=dim)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two vectors (safe for zero vectors)."""
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    A = set(text_a.lower().split())
    B = set(text_b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def create_graph(events: List[Dict[str, Any]]) -> nx.DiGraph:
    """Create a directed graph with one node per event."""
    G = nx.DiGraph()
    for event in events:
        G.add_node(event['id'], **event)
    return G


# ---------------------------------------------------------------------------
# Verse‐reference parsing
# ---------------------------------------------------------------------------

def parse_verse_range(verse_ref: str) -> Optional[Tuple[int, int, int, int, Optional[str], Optional[str]]]:
    """Parse verse references such as ``'21:1-7'``, ``'15:18-16:4'``,
    ``'21:19a'``, ``'21:19b'``.

    Returns
    -------
    tuple
        ``(start_chapter, start_verse, end_chapter, end_verse, start_part, end_part)``
        where *parts* are ``'a'``, ``'b'``, or ``None``.  Returns ``None`` on
        parse failure.
    """
    if not verse_ref or not verse_ref.strip():
        return None
    try:
        verse_ref = verse_ref.strip()
        # Cross-chapter pattern e.g., 15:18-16:4
        m = re.match(r'(\d+):(\d+)([ab]?)-(\d+):(\d+)([ab]?)$', verse_ref)
        if m:
            sc = int(m.group(1)); sv = int(m.group(2)); sp = m.group(3) or None
            ec = int(m.group(4)); ev = int(m.group(5)); ep = m.group(6) or None
            return sc, sv, ec, ev, sp, ep
        # Single chapter
        parts = verse_ref.split(':')
        if len(parts) != 2:
            return None
        chapter = int(parts[0])
        vr = parts[1]
        # Single verse (possibly with suffix)
        m2 = re.match(r'^(\d+)([ab]?)$', vr)
        if m2:
            v = int(m2.group(1))
            p = m2.group(2) or None
            return chapter, v, chapter, v, p, p
        # Range within chapter, possibly with suffixes
        if '-' in vr:
            a, b = vr.split('-', 1)
            ma = re.match(r'^(\d+)([ab]?)$', a.strip())
            mb = re.match(r'^(\d+)([ab]?)$', b.strip())
            if not (ma and mb):
                return None
            sv = int(ma.group(1)); sp = ma.group(2) or None
            ev = int(mb.group(1)); ep = mb.group(2) or None
            return chapter, sv, chapter, ev, sp, ep
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ISO-TimeML constants (ISO 24617-1:2012)
# ---------------------------------------------------------------------------

# EVENT @class values (Table A.1 in ISO 24617-1:2012 Annex A)
ISO_EVENT_CLASSES = {
    'OCCURRENCE',   # Events describing an action/process
    'STATE',        # Stative events
    'REPORTING',    # Communication events (said, told, ...)
    'PERCEPTION',   # Perception events (saw, heard, ...)
    'ASPECTUAL',    # Aspectual predicates (began, stopped, ...)
    'I_ACTION',     # Intentional action (tried, attempted, ...)
    'I_STATE',      # Intentional state (wanted, believed, ...)
}

# TIMEX3 @type values (Section 7.3.4 / Table A.2)
ISO_TIMEX3_TYPES = {'DATE', 'TIME', 'DURATION', 'SET'}

# TLINK @relType values (Allen's 13 interval relations — Section 7.3.5)
ISO_TLINK_RELATIONS = {
    'BEFORE', 'AFTER',
    'INCLUDES', 'IS_INCLUDED',
    'DURING', 'DURING_INV',
    'SIMULTANEOUS', 'IAFTER', 'IBEFORE',
    'IDENTITY', 'BEGINS', 'ENDS',
    'BEGUN_BY', 'ENDED_BY',
    'OVERLAP', 'OVERLAPPED_BY',
}

# SLINK relation types (subordination)
ISO_SLINK_RELATIONS = {
    'MODAL', 'EVIDENTIAL', 'NEG_EVIDENTIAL',
    'FACTIVE', 'COUNTER_FACTIVE', 'CONDITIONAL',
}

# ALINK relation types (aspectual)
ISO_ALINK_RELATIONS = {
    'INITIATES', 'TERMINATES', 'CONTINUES', 'CULMINATES', 'REINITIATES',
}

# Temporal signal keywords mapped to likely TLINK relation
TEMPORAL_SIGNALS: Dict[str, str] = {
    'before': 'BEFORE',
    'after': 'AFTER',
    'during': 'DURING',
    'while': 'SIMULTANEOUS',
    'when': 'SIMULTANEOUS',
    'until': 'ENDED_BY',
    'since': 'BEGUN_BY',
    'then': 'AFTER',
    'later': 'AFTER',
    'earlier': 'BEFORE',
    'meanwhile': 'SIMULTANEOUS',
    'immediately': 'IAFTER',
    'following': 'AFTER',
    'prior to': 'BEFORE',
    'next': 'AFTER',
    'afterwards': 'AFTER',
    'previously': 'BEFORE',
}


# ---------------------------------------------------------------------------
# Biblical temporal normalisation helpers
# ---------------------------------------------------------------------------

# Maps biblical "hour" references to approximate modern clock times.
# "Third hour" ≈ 9 AM (counting from 6 AM sunrise).
BIBLICAL_HOURS = {
    'first': 'T06:00', 'second': 'T07:00', 'third': 'T09:00',
    'fourth': 'T10:00', 'fifth': 'T11:00', 'sixth': 'T12:00',
    'seventh': 'T13:00', 'eighth': 'T14:00', 'ninth': 'T15:00',
    'tenth': 'T16:00', 'eleventh': 'T17:00', 'twelfth': 'T18:00',
}

# Duration keywords to ISO 8601 approximate durations
DURATION_KEYWORDS = {
    'three days': 'P3D', 'two days': 'P2D', 'forty days': 'P40D',
    'six days': 'P6D', 'seven days': 'P7D', 'eight days': 'P8D',
    'three years': 'P3Y', 'a week': 'P7D', 'a year': 'P1Y',
}

def normalise_biblical_timex(text: str) -> Optional[str]:
    """Attempt to normalise a biblical temporal expression to ISO 8601."""
    t = text.lower().strip()
    # Check durations first
    for kw, val in DURATION_KEYWORDS.items():
        if kw in t:
            return val
    # Biblical hour references
    for ordinal, clock in BIBLICAL_HOURS.items():
        if ordinal in t and 'hour' in t:
            return clock
    # Relative day references
    if 'third day' in t:
        return '+P3D'
    if 'next day' in t or 'following day' in t:
        return '+P1D'
    if 'sabbath' in t.lower():
        return 'XXXX-WXX-6'   # Saturday
    return None
