"""
TAVERN — Stage 2: Cross-Document Event Alignment
==================================================
Implements cross-document event alignment using:

1. **Chronology-guided alignment** — uses the Aschmann chronology table to
   find parallel pericopes across gospels.
2. **Semantic similarity** — Sentence-BERT embeddings (with hash-embedding
   fallback) for measuring event textual similarity.
3. **Entity overlap** — shared named entities boost alignment scores.
4. **Temporal proximity** — aligned events are penalised if they are
   far apart in the chronological sequence.
5. **Conflict detection** — incompatible temporal relations across gospels
   are flagged for downstream resolution.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from utils.helpers import (
    get_embedding,
    cosine_similarity,
    jaccard_similarity,
    parse_verse_range,
    TEMPORAL_SIGNALS,
)

# ---------------------------------------------------------------------------
# Chronology helpers
# ---------------------------------------------------------------------------

def _extract_chapter_verse(verse_ref: str) -> Tuple[int, int]:
    """Extract (chapter, start_verse) from a verse reference string."""
    parsed = parse_verse_range(verse_ref)
    if parsed:
        return parsed[0], parsed[1]
    # Fallback: try simple chapter:verse pattern
    m = re.match(r'(\d+):(\d+)', verse_ref)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 0)


def find_parallel_pericopes(
        chronology: List[Dict[str, Any]],
        doc_ids: List[str],
) -> List[Dict[str, Any]]:
    """Identify pericopes that appear in multiple gospel documents.

    Parameters
    ----------
    chronology : list of dict
        Rows from the Aschmann harmony table, each with keys like
        ``'Event'``, ``'Matthew'``, ``'Mark'``, ``'Luke'``, ``'John'``.
    doc_ids : list of str
        Document identifiers present in the pipeline (e.g. ``['matthew', 'john']``).

    Returns
    -------
    list of dict
        Each entry has:
        * ``event_name`` — narrative description from the chronology
        * ``doc_refs``   — mapping ``{doc_id: verse_ref}`` for each gospel
        * ``chrono_idx`` — position in the chronology (for temporal proximity)
    """
    # Map doc_ids to column names (case-insensitive)
    col_map: Dict[str, str] = {}
    for doc_id in doc_ids:
        for col in ('Matthew', 'Mark', 'Luke', 'John'):
            if doc_id.lower() == col.lower():
                col_map[doc_id] = col
                break

    parallels = []
    for idx, row in enumerate(chronology):
        refs: Dict[str, str] = {}
        for doc_id, col in col_map.items():
            ref = row.get(col, '')
            if ref and str(ref).strip():
                refs[doc_id] = str(ref).strip()
        if len(refs) >= 2:  # At least two gospels share this pericope
            parallels.append({
                'event_name': row.get('Event', f'Pericope_{idx}'),
                'doc_refs': refs,
                'chrono_idx': idx,
            })
    return parallels


# ---------------------------------------------------------------------------
# Multi-factor alignment score
# ---------------------------------------------------------------------------

def _entity_overlap(text_a: str, text_b: str) -> float:
    """Simple named-entity overlap using capitalised word heuristic."""
    def _entities(t: str) -> set:
        return {w for w in re.findall(r'\b[A-Z][a-z]+\b', t)}
    A = _entities(text_a)
    B = _entities(text_b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def compute_alignment_score(
        event_a: Dict[str, Any],
        event_b: Dict[str, Any],
        chrono_distance: Optional[int] = None,
        *,
        w_sem: float = 0.45,
        w_ent: float = 0.30,
        w_prox: float = 0.25,
) -> float:
    """Compute a multi-factor alignment score for two events.

    The score is a weighted sum:

    .. code-block:: text

        score = w_sem  * semantic_similarity
              + w_ent  * entity_overlap
              + w_prox * temporal_proximity

    Default weights: SBERT 45 %, entity 30 %, proximity 25 %.

    Parameters
    ----------
    event_a, event_b : dict
        Event annotation dicts (must have ``'text'`` key).
    chrono_distance : int or None
        Absolute difference in ``chrono_idx`` between the two events'
        pericopes. ``None`` means ignore temporal proximity.
    w_sem, w_ent, w_prox : float
        Blending weights (must sum to ≤ 1.0).
    """
    text_a = event_a.get('text', '')
    text_b = event_b.get('text', '')

    # Semantic similarity
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)
    sem_score = cosine_similarity(emb_a, emb_b)

    # Entity overlap
    ent_score = _entity_overlap(text_a, text_b)

    # Temporal proximity (normalised 0-1, higher = closer)
    if chrono_distance is not None:
        prox_score = max(0.0, 1.0 - chrono_distance / 50.0)
    else:
        prox_score = 0.5  # Neutral when unknown

    return w_sem * sem_score + w_ent * ent_score + w_prox * prox_score


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

_CONTRADICTORY_PAIRS = {
    ('BEFORE', 'AFTER'), ('AFTER', 'BEFORE'),
    ('SIMULTANEOUS', 'BEFORE'), ('SIMULTANEOUS', 'AFTER'),
    ('BEGINS', 'ENDS'), ('ENDS', 'BEGINS'),
}


def detect_conflicts(
        alignments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect temporal relation conflicts across aligned event groups.

    Parameters
    ----------
    alignments : list of dict
        Each alignment has ``'events'`` (list of event dicts) and optionally
        ``'tlinks'`` linking pairs of them.

    Returns
    -------
    list of dict
        Each conflict dict has ``'event_a'``, ``'event_b'``, ``'rel_a'``,
        ``'rel_b'``, and ``'docs'``.
    """
    conflicts: List[Dict[str, Any]] = []

    for alignment in alignments:
        events = alignment.get('events', [])
        tlinks = alignment.get('tlinks', [])

        # Build a dict of (eventID_pair) → relType per document
        pair_rels: Dict[Tuple, Dict[str, str]] = {}
        for tlink in tlinks:
            eid = tlink.get('eventID', '')
            rid = tlink.get('relatedToEvent', '')
            rel = tlink.get('relType', '')
            doc = tlink.get('doc_id', 'unknown')
            key = (eid, rid)
            if key not in pair_rels:
                pair_rels[key] = {}
            pair_rels[key][doc] = rel

        # Check each pair for contradictions
        for (eid, rid), doc_rels in pair_rels.items():
            doc_list = list(doc_rels.items())
            for (doc1, rel1), (doc2, rel2) in combinations(doc_list, 2):
                if (rel1, rel2) in _CONTRADICTORY_PAIRS:
                    conflicts.append({
                        'event_a': eid,
                        'event_b': rid,
                        'rel_a': rel1,
                        'rel_b': rel2,
                        'docs': (doc1, doc2),
                    })
    return conflicts


# ---------------------------------------------------------------------------
# Main alignment function
# ---------------------------------------------------------------------------

def align_events(
        annotated_docs: Dict[str, List[Dict[str, Any]]],
        chronology: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """Align events across Gospel documents.

    Parameters
    ----------
    annotated_docs : dict
        ``{doc_id: [annotation_dict, ...]}``, output of Stage 1.
    chronology : list of dict or None
        Harmony / chronology table rows. If ``None``, falls back to
        pure semantic alignment.
    threshold : float
        Minimum alignment score to include a pair. Default 0.35.

    Returns
    -------
    list of dict
        Each alignment group has:
        * ``'group_id'``   — unique identifier
        * ``'events'``     — list of aligned event dicts
        * ``'scores'``     — pairwise alignment scores
        * ``'conflicts'``  — any detected temporal conflicts
    """
    doc_ids = list(annotated_docs.keys())

    # Build per-doc event lists
    doc_events: Dict[str, List[Dict]] = {
        doc_id: [a for a in anns if a['type'] == 'EVENT']
        for doc_id, anns in annotated_docs.items()
    }
    doc_tlinks: Dict[str, List[Dict]] = {
        doc_id: [a for a in anns if a['type'] == 'TLINK']
        for doc_id, anns in annotated_docs.items()
    }

    # Build chronology map (event_name → chrono_idx)
    chrono_map: Dict[str, Dict] = {}
    if chronology:
        parallels = find_parallel_pericopes(chronology, doc_ids)
        for p in parallels:
            chrono_map[p['event_name']] = p

    alignments: List[Dict[str, Any]] = []
    group_id = 0

    for doc_a, doc_b in combinations(doc_ids, 2):
        events_a = doc_events[doc_a]
        events_b = doc_events[doc_b]

        for ev_a in events_a:
            best_score = threshold
            best_ev_b = None

            for ev_b in events_b:
                # Temporal proximity from chronology
                chrono_dist = None
                for pericope in chrono_map.values():
                    refs = pericope.get('doc_refs', {})
                    if doc_a in refs and doc_b in refs:
                        chrono_dist = 0  # Same pericope
                        break

                score = compute_alignment_score(
                    ev_a, ev_b, chrono_distance=chrono_dist)

                if score > best_score:
                    best_score = score
                    best_ev_b = ev_b

            if best_ev_b is not None:
                # Gather tlinks involving these events
                combined_tlinks = [
                    {**tl, 'doc_id': doc_a}
                    for tl in doc_tlinks[doc_a]
                    if ev_a['id'] in (tl.get('eventID'), tl.get('relatedToEvent'))
                ] + [
                    {**tl, 'doc_id': doc_b}
                    for tl in doc_tlinks[doc_b]
                    if best_ev_b['id'] in (tl.get('eventID'), tl.get('relatedToEvent'))
                ]

                alignment_entry = {
                    'group_id': f'align_{group_id}',
                    'events': [ev_a, best_ev_b],
                    'scores': {f'{doc_a}-{doc_b}': best_score},
                    'tlinks': combined_tlinks,
                    'conflicts': [],
                }

                # Detect conflicts within this alignment
                conflicts = detect_conflicts([alignment_entry])
                alignment_entry['conflicts'] = conflicts

                alignments.append(alignment_entry)
                group_id += 1

    print(f"  Aligned {len(alignments)} event pairs across "
          f"{len(list(combinations(doc_ids, 2)))} document pairs")
    if any(a['conflicts'] for a in alignments):
        n_conflicts = sum(len(a['conflicts']) for a in alignments)
        print(f"  Detected {n_conflicts} temporal conflicts")

    return alignments
