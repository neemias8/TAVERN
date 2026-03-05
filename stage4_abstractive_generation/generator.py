"""
TAVERN — Stage 4: Temporal Narrative Generation
================================================
Generates coherent narrative summaries from the temporal knowledge graph,
using graph-guided sentence ordering and an LLM backend for fluent text.

Pipeline
--------
1. **Temporal segmentation** — cluster events into "day" segments using
   TIMEX3 anchors and TLINK BEFORE chains.
2. **Graph-guided ordering** — topological sort on the subgraph for each
   segment to establish a coherent narrative sequence.
3. **Narrative generation** — each segment is passed to an LLM (or a
   template-based fallback) to produce fluent prose.
4. **Output** — a clean, multi-paragraph narrative string.
"""

from __future__ import annotations

import os
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from utils.helpers import cosine_similarity

# ---------------------------------------------------------------------------
# LLM backend (optional)
# ---------------------------------------------------------------------------

_openai_client = None

def _get_openai_client():
    """Lazy-initialise OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        import openai
        api_key = os.environ.get('OPENAI_API_KEY', '')
        if api_key:
            _openai_client = openai.OpenAI(api_key=api_key)
            return _openai_client
    except Exception:
        pass
    return None


def _llm_generate(prompt: str, max_tokens: int = 512) -> str:
    """Generate text using OpenAI GPT, or return empty string on failure."""
    client = _get_openai_client()
    if client is None:
        return ''
    try:
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Temporal segmentation
# ---------------------------------------------------------------------------

_DAY_MARKERS = {
    'morning', 'dawn', 'daybreak', 'sunrise', 'evening', 'sunset',
    'night', 'midnight', 'noon', 'that day', 'the next day',
    'the following day', 'third day', 'sabbath', 'passover',
}


def _is_day_boundary(text: str) -> bool:
    """Return True if *text* contains a marker that signals a new day."""
    t = text.lower()
    return any(marker in t for marker in _DAY_MARKERS)


def segment_by_day(
        events: List[Dict[str, Any]],
        tlinks: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Partition events into day-level narrative segments.

    Strategy
    --------
    * Scan events in narrative order (by ``verse_idx``, then ``verse``).
    * A new segment starts whenever a TIMEX3 or event text contains a
      day-boundary marker.
    * Short segments (< 2 events) are merged with the previous one.

    Returns
    -------
    list of list of dict
        Each inner list is one day-segment of events.
    """
    if not events:
        return []

    # Sort by verse position
    sorted_events = sorted(
        events,
        key=lambda e: (e.get('verse_idx', 0), e.get('verse', 0))
    )

    segments: List[List[Dict]] = [[]]
    for ev in sorted_events:
        if _is_day_boundary(ev.get('text', '')) and segments[-1]:
            segments.append([])
        segments[-1].append(ev)

    # Merge tiny segments
    merged: List[List[Dict]] = []
    for seg in segments:
        if len(seg) < 2 and merged:
            merged[-1].extend(seg)
        else:
            merged.append(seg)

    return merged


# ---------------------------------------------------------------------------
# Graph-guided ordering
# ---------------------------------------------------------------------------

def order_segment(
        segment_events: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        tlinks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order events within a segment using a topological sort on TLINKs.

    Falls back to verse-order when the TLINK subgraph has cycles or no edges.
    """
    if len(segment_events) <= 1:
        return segment_events

    seg_ids = {ev['id'] for ev in segment_events}
    id_to_ev = {ev['id']: ev for ev in segment_events}

    G = nx.DiGraph()
    G.add_nodes_from(seg_ids)

    for tl in tlinks:
        src = tl.get('eventID', '')
        dst = tl.get('relatedToEvent', '')
        rel = tl.get('relType', '')
        if src in seg_ids and dst in seg_ids and rel in ('BEFORE', 'AFTER', 'SEQUENTIAL'):
            if rel == 'AFTER':
                G.add_edge(dst, src)
            else:
                G.add_edge(src, dst)

    try:
        ordered_ids = list(nx.topological_sort(G))
        return [id_to_ev[eid] for eid in ordered_ids if eid in id_to_ev]
    except nx.NetworkXUnfeasible:
        # Cycle — fall back to verse order
        return sorted(segment_events,
                      key=lambda e: (e.get('verse_idx', 0), e.get('verse', 0)))


# ---------------------------------------------------------------------------
# Template-based narrative generation
# ---------------------------------------------------------------------------

_REPORTING_INTROS = [
    '{subject} said,', '{subject} declared,', '{subject} told them,',
    'According to {subject},', '{subject} proclaimed,',
]
_TRANSITION_WORDS = [
    'Then', 'Afterwards', 'Following this', 'Next', 'Subsequently',
    'At that time', 'After this',
]


def _extract_subject(text: str) -> str:
    """Heuristic: return the first capitalised word as the subject."""
    tokens = text.split()
    for tok in tokens:
        if tok and tok[0].isupper() and tok.isalpha():
            return tok
    return 'He'


def _template_sentence(ev: Dict[str, Any], idx: int) -> str:
    """Generate a single template-based sentence for an event."""
    text = ev.get('text', '').strip().rstrip('.')
    ev_class = ev.get('event_class', 'OCCURRENCE')
    polarity = ev.get('polarity', 'POS')

    if polarity == 'NEG' and 'not' not in text.lower():
        text = 'it was not the case that ' + text[0].lower() + text[1:]

    if idx == 0:
        return text + '.'

    transition = _TRANSITION_WORDS[idx % len(_TRANSITION_WORDS)]
    if ev_class == 'REPORTING':
        subj = _extract_subject(text)
        intro_tmpl = _REPORTING_INTROS[idx % len(_REPORTING_INTROS)]
        intro = intro_tmpl.format(subject=subj)
        return f'{transition}, {intro} "{text}."'

    return f'{transition}, {text}.'


def _generate_segment_narrative(
        segment_events: List[Dict[str, Any]],
        segment_idx: int,
) -> str:
    """Generate a narrative paragraph for one day-segment."""
    if not segment_events:
        return ''

    # Try LLM first
    event_summaries = ' | '.join(
        f"{e.get('verse_ref', '?')}: {e.get('text', '')[:120]}"
        for e in segment_events[:12]  # Cap at 12 events
    )
    prompt = (
        f"You are narrating events from the Gospels in chronological order. "
        f"Write a single coherent paragraph (4-8 sentences) describing the "
        f"following events in sequence. Use past tense, third person. "
        f"Events: {event_summaries}"
    )
    llm_text = _llm_generate(prompt)
    if llm_text:
        return llm_text

    # Fallback: template-based
    sentences = [
        _template_sentence(ev, idx)
        for idx, ev in enumerate(segment_events[:10])
    ]
    return ' '.join(sentences)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_narrative(
        annotated_docs: Dict[str, List[Dict[str, Any]]],
        alignments: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
) -> str:
    """Generate a temporally-ordered narrative from multiple Gospel documents.

    Parameters
    ----------
    annotated_docs : dict
        ``{doc_id: [annotation_dict, ...]}``, output of Stage 1.
    alignments : list of dict
        Cross-document event alignment groups, output of Stage 2.
    embeddings : dict
        ``{node_id: embedding_array}``, output of Stage 3.

    Returns
    -------
    str
        Multi-paragraph narrative string.
    """
    # Merge events from all documents
    all_events: List[Dict] = []
    all_tlinks: List[Dict] = []

    for doc_id, annotations in annotated_docs.items():
        for ann in annotations:
            if ann['type'] == 'EVENT':
                all_events.append({**ann, 'doc_id': doc_id})
            elif ann['type'] == 'TLINK':
                all_tlinks.append({**ann, 'doc_id': doc_id})

    # De-duplicate events using alignments
    seen_group_ids: set = set()
    deduped_events: List[Dict] = []
    aligned_event_ids: set = set()

    for alignment in alignments:
        gid = alignment['group_id']
        if gid not in seen_group_ids:
            seen_group_ids.add(gid)
            evs = alignment.get('events', [])
            if evs:
                # Pick the event with the most text as representative
                rep = max(evs, key=lambda e: len(e.get('text', '')))
                deduped_events.append(rep)
                for ev in evs:
                    aligned_event_ids.add(ev['id'])

    # Add unaligned events
    for ev in all_events:
        if ev['id'] not in aligned_event_ids:
            deduped_events.append(ev)

    if not deduped_events:
        return 'No events to narrate.'

    # Segment by day
    segments = segment_by_day(deduped_events, all_tlinks)

    # Order within each segment and generate narrative
    paragraphs: List[str] = []
    for seg_idx, segment in enumerate(segments):
        ordered = order_segment(segment, embeddings, all_tlinks)
        para = _generate_segment_narrative(ordered, seg_idx)
        if para:
            paragraphs.append(para)

    return '\n\n'.join(paragraphs) if paragraphs else 'No narrative generated.'
