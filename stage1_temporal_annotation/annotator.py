"""
TAVERN — Stage 1: Temporal Annotation (ISO-TimeML)
====================================================
Annotates each Gospel document according to ISO 24617-1:2012 (ISO-TimeML),
producing structured annotation dicts for:

* **<EVENT>** — with ``class``, ``tense``, ``aspect``, ``polarity`` attributes
* **<TIMEX3>** — with ``type``, normalised ``value``
* **<SIGNAL>** — temporal connectives / prepositions
* **<TLINK>** — intra-document temporal ordering links
* **<SLINK>** — subordination links (modal, evidential, …)
"""

import re
from typing import Any, Dict, List, Optional

from utils.helpers import (
    parse_gospel_xml,
    ISO_EVENT_CLASSES,
    ISO_TLINK_RELATIONS,
    TEMPORAL_SIGNALS,
    normalise_biblical_timex,
)

# ---------------------------------------------------------------------------
# spaCy setup (optional but strongly recommended)
# ---------------------------------------------------------------------------
nlp = None
try:
    import spacy
    for model_name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            nlp = spacy.load(model_name)
            break
        except Exception:
            continue
except Exception:
    pass


# ---------------------------------------------------------------------------
# EVENT classification helpers (ISO 24617-1 Annex A.2.1)
# ---------------------------------------------------------------------------

_REPORTING_LEMMAS = {
    'say', 'tell', 'ask', 'answer', 'reply', 'declare', 'announce',
    'proclaim', 'preach', 'teach', 'warn', 'command', 'order', 'call',
    'cry', 'shout', 'write', 'testify', 'prophesy', 'explain',
}
_PERCEPTION_LEMMAS = {
    'see', 'hear', 'watch', 'notice', 'observe', 'feel', 'perceive',
    'look', 'behold', 'witness',
}
_ASPECTUAL_LEMMAS = {
    'begin', 'start', 'stop', 'finish', 'end', 'cease', 'continue',
    'resume', 'complete',
}
_I_ACTION_LEMMAS = {
    'try', 'attempt', 'seek', 'desire', 'want', 'plan', 'decide',
    'intend', 'hope', 'promise', 'agree', 'refuse', 'choose',
}
_I_STATE_LEMMAS = {
    'believe', 'know', 'think', 'understand', 'love', 'fear', 'hate',
    'trust', 'doubt', 'suppose', 'expect', 'remember', 'forget',
}
_STATE_PATTERNS = re.compile(
    r'\b(is|are|was|were|am|be|being|been|has|have|had|belong|contain|'
    r'consist|exist|remain|seem|appear|become)\b', re.I
)


def _classify_event(token, sentence_text: str) -> str:
    """Determine the ISO-TimeML event class for a verb token."""
    lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else token.text.lower()
    if lemma in _REPORTING_LEMMAS:
        return 'REPORTING'
    if lemma in _PERCEPTION_LEMMAS:
        return 'PERCEPTION'
    if lemma in _ASPECTUAL_LEMMAS:
        return 'ASPECTUAL'
    if lemma in _I_ACTION_LEMMAS:
        return 'I_ACTION'
    if lemma in _I_STATE_LEMMAS:
        return 'I_STATE'
    if _STATE_PATTERNS.search(token.text):
        return 'STATE'
    return 'OCCURRENCE'


def _extract_tense(token) -> str:
    """Map spaCy morphology to ISO-TimeML tense values."""
    morph = token.morph.to_dict() if hasattr(token.morph, 'to_dict') else {}
    tense = morph.get('Tense', '')
    if tense == 'Past':
        return 'PAST'
    if tense == 'Pres':
        return 'PRESENT'
    # Check for auxiliaries indicating future
    for child in token.children:
        if child.lemma_ in ('will', 'shall'):
            return 'FUTURE'
    return 'PAST'  # Default for narrative text


def _extract_aspect(token) -> str:
    """Map spaCy morphology to ISO-TimeML aspect values."""
    morph = token.morph.to_dict() if hasattr(token.morph, 'to_dict') else {}
    aspect = morph.get('Aspect', '')
    if aspect == 'Perf':
        return 'PERFECTIVE'
    if aspect == 'Prog':
        return 'PROGRESSIVE'
    # Heuristic: past tense without auxiliary → perfective
    tense = morph.get('Tense', '')
    if tense == 'Past':
        return 'PERFECTIVE'
    return 'NONE'


def _extract_polarity(sentence_text: str) -> str:
    """Detect negative polarity markers."""
    neg_pattern = re.compile(r'\b(not|never|no|neither|nor|cannot|n\'t)\b', re.I)
    return 'NEG' if neg_pattern.search(sentence_text) else 'POS'


# ---------------------------------------------------------------------------
# TIMEX3 extraction & normalisation
# ---------------------------------------------------------------------------

# Regex patterns for temporal expressions not caught by spaCy NER
_TIMEX_PATTERNS = [
    # Ordinal day references: "the third day", "the next day"
    (re.compile(r'\bthe\s+(first|second|third|fourth|fifth|sixth|seventh|'
                r'eighth|ninth|tenth|next|following|previous)\s+day\b', re.I),
     'DATE'),
    # Biblical hour: "the third hour", "about the sixth hour"
    (re.compile(r'\b(?:about\s+)?the\s+(first|second|third|fourth|fifth|sixth|'
                r'seventh|eighth|ninth|tenth|eleventh|twelfth)\s+hour\b', re.I),
     'TIME'),
    # Duration: "three days", "forty days and forty nights"
    (re.compile(r'\b(\d+|two|three|four|five|six|seven|eight|nine|ten|'
                r'forty|twelve)\s+(days?|nights?|hours?|years?|weeks?|months?)\b', re.I),
     'DURATION'),
    # Named periods: "Passover", "Sabbath", "Feast of Unleavened Bread"
    (re.compile(r'\b(Passover|Sabbath|Feast\s+of\s+\w+|Pentecost|'
                r'Day\s+of\s+Preparation|evening|morning|dawn|sunset|'
                r'midnight|noon)\b', re.I),
     'DATE'),
]

# Temporal signal patterns
_SIGNAL_PATTERN = re.compile(
    r'\b(before|after|during|while|when|until|since|then|later|earlier|'
    r'meanwhile|immediately|following|next|afterwards|previously|'
    r'prior\s+to|as\s+soon\s+as)\b', re.I
)


def _extract_timex3(text: str, verse_idx: int, doc_id: str,
                    counter: int) -> tuple:
    """Extract TIMEX3 annotations from text.

    Returns (list_of_timex3_annotations, updated_counter).
    """
    timex_annotations = []
    seen_spans = set()

    # 1. spaCy NER-based extraction
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ('DATE', 'TIME'):
                span_key = (ent.start_char, ent.end_char)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    timex_type = 'TIME' if ent.label_ == 'TIME' else 'DATE'
                    normalised = normalise_biblical_timex(ent.text)
                    timex_annotations.append({
                        'id': f"{doc_id}_t{counter}",
                        'text': ent.text,
                        'verse_idx': verse_idx,
                        'type': 'TIMEX3',
                        'timex3_type': timex_type,
                        'value': normalised,  # May be None
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                    })
                    counter += 1

    # 2. Regex-based extraction (catch biblical expressions spaCy misses)
    for pattern, timex_type in _TIMEX_PATTERNS:
        for m in pattern.finditer(text):
            span_key = (m.start(), m.end())
            # Avoid duplicates with NER-found spans
            overlaps = any(
                not (m.end() <= s[0] or m.start() >= s[1])
                for s in seen_spans
            )
            if not overlaps:
                seen_spans.add(span_key)
                normalised = normalise_biblical_timex(m.group())
                timex_annotations.append({
                    'id': f"{doc_id}_t{counter}",
                    'text': m.group(),
                    'verse_idx': verse_idx,
                    'type': 'TIMEX3',
                    'timex3_type': timex_type,
                    'value': normalised,
                    'start_char': m.start(),
                    'end_char': m.end(),
                })
                counter += 1

    return timex_annotations, counter


def _extract_signals(text: str, verse_idx: int, doc_id: str,
                     counter: int) -> tuple:
    """Extract temporal SIGNAL annotations."""
    signals = []
    for m in _SIGNAL_PATTERN.finditer(text):
        signals.append({
            'id': f"{doc_id}_s{counter}",
            'text': m.group(),
            'verse_idx': verse_idx,
            'type': 'SIGNAL',
            'start_char': m.start(),
            'end_char': m.end(),
        })
        counter += 1
    return signals, counter


# ---------------------------------------------------------------------------
# TLINK generation (intra-document)
# ---------------------------------------------------------------------------

def _generate_tlinks(events: List[Dict], signals: List[Dict],
                     doc_id: str) -> List[Dict]:
    """Generate intra-document TLINKs based on:
    1. Sequential narrative order (consecutive events → BEFORE)
    2. Signal-based relations (explicit temporal connectives)
    3. Event-to-TIMEX3 anchoring (IS_INCLUDED)
    """
    tlinks: List[Dict] = []
    tlink_id = 0

    # Filter only EVENT annotations
    event_annos = [e for e in events if e['type'] == 'EVENT']
    timex_annos = [t for t in events if t['type'] == 'TIMEX3']
    signal_annos = signals

    # 1. Sequential ordering: events in narrative order get BEFORE links
    for i in range(len(event_annos) - 1):
        e1 = event_annos[i]
        e2 = event_annos[i + 1]
        tlinks.append({
            'id': f"{doc_id}_tl{tlink_id}",
            'type': 'TLINK',
            'relType': 'BEFORE',
            'eventID': e1['id'],
            'relatedToEvent': e2['id'],
            'source': 'narrative_order',
        })
        tlink_id += 1

    # 2. Signal-based relations
    for sig in signal_annos:
        sig_text = sig['text'].lower().strip()
        rel_type = TEMPORAL_SIGNALS.get(sig_text)
        if rel_type:
            # Find the closest event before and after the signal in the same verse
            verse_events = [e for e in event_annos if e['verse_idx'] == sig['verse_idx']]
            if len(verse_events) >= 2:
                # Link the event containing the signal to the next event
                tlinks.append({
                    'id': f"{doc_id}_tl{tlink_id}",
                    'type': 'TLINK',
                    'relType': rel_type,
                    'eventID': verse_events[0]['id'],
                    'relatedToEvent': verse_events[-1]['id'],
                    'signalID': sig['id'],
                    'source': 'signal',
                })
                tlink_id += 1

    # 3. Event-to-TIMEX3 anchoring (IS_INCLUDED)
    for timex in timex_annos:
        # Find events in the same verse
        verse_events = [e for e in event_annos if e['verse_idx'] == timex['verse_idx']]
        for ev in verse_events:
            tlinks.append({
                'id': f"{doc_id}_tl{tlink_id}",
                'type': 'TLINK',
                'relType': 'IS_INCLUDED',
                'eventID': ev['id'],
                'relatedToTime': timex['id'],
                'source': 'temporal_anchor',
            })
            tlink_id += 1

    return tlinks


# ---------------------------------------------------------------------------
# SLINK generation (subordination)
# ---------------------------------------------------------------------------

def _generate_slinks(events: List[Dict], doc_id: str) -> List[Dict]:
    """Generate SLINKs for subordination relations between events.

    Detects patterns like reporting verbs introducing subordinate events,
    modal constructions, etc.
    """
    slinks: List[Dict] = []
    slink_id = 0
    event_annos = [e for e in events if e['type'] == 'EVENT']

    for i, ev in enumerate(event_annos):
        ev_class = ev.get('event_class', '')

        # REPORTING events introduce EVIDENTIAL SLINKs to the next event
        if ev_class == 'REPORTING' and i + 1 < len(event_annos):
            next_ev = event_annos[i + 1]
            if next_ev['verse_idx'] == ev['verse_idx']:  # Same verse
                slinks.append({
                    'id': f"{doc_id}_sl{slink_id}",
                    'type': 'SLINK',
                    'relType': 'EVIDENTIAL',
                    'eventID': ev['id'],
                    'subordinatedEvent': next_ev['id'],
                })
                slink_id += 1

        # I_ACTION events produce MODAL SLINKs
        if ev_class == 'I_ACTION' and i + 1 < len(event_annos):
            next_ev = event_annos[i + 1]
            if next_ev['verse_idx'] == ev['verse_idx']:
                slinks.append({
                    'id': f"{doc_id}_sl{slink_id}",
                    'type': 'SLINK',
                    'relType': 'MODAL',
                    'eventID': ev['id'],
                    'subordinatedEvent': next_ev['id'],
                })
                slink_id += 1

        # Negative polarity with a factive → COUNTER_FACTIVE
        if ev.get('polarity') == 'NEG' and ev_class in ('I_STATE', 'PERCEPTION'):
            if i + 1 < len(event_annos):
                next_ev = event_annos[i + 1]
                if next_ev['verse_idx'] == ev['verse_idx']:
                    slinks.append({
                        'id': f"{doc_id}_sl{slink_id}",
                        'type': 'SLINK',
                        'relType': 'COUNTER_FACTIVE',
                        'eventID': ev['id'],
                        'subordinatedEvent': next_ev['id'],
                    })
                    slink_id += 1

    return slinks


# ---------------------------------------------------------------------------
# Main annotation function
# ---------------------------------------------------------------------------

def annotate_document(doc_path: str, doc_id: str) -> List[Dict[str, Any]]:
    """Annotate a single Gospel document with ISO-TimeML elements.

    Returns a flat list of annotation dicts (EVENTs, TIMEX3s, SIGNALs, TLINKs, SLINKs).
    """
    verses_metadata = parse_gospel_xml(doc_path)
    annotations: List[Dict[str, Any]] = []
    event_id = 0
    timex_counter = 0
    signal_counter = 0

    for verse_idx, verse_data in enumerate(verses_metadata):
        verse_text = verse_data['text']
        chapter = verse_data['chapter']
        verse_num = verse_data['verse']

        # ---- TIMEX3 extraction ----
        timex_annos, timex_counter = _extract_timex3(
            verse_text, verse_idx, doc_id, timex_counter)
        annotations.extend(timex_annos)

        # ---- SIGNAL extraction ----
        signal_annos, signal_counter = _extract_signals(
            verse_text, verse_idx, doc_id, signal_counter)
        annotations.extend(signal_annos)

        # ---- EVENT extraction ----
        if nlp is None:
            # Fallback: one EVENT per verse (no NLP available)
            annotations.append({
                'id': f"{doc_id}_e{event_id}",
                'text': verse_text.strip(),
                'verse_idx': verse_idx,
                'chapter': chapter,
                'verse': verse_num,
                'verse_ref': f"{chapter}:{verse_num}",
                'type': 'EVENT',
                'event_class': 'OCCURRENCE',
                'tense': 'PAST',
                'aspect': 'PERFECTIVE',
                'polarity': _extract_polarity(verse_text),
                'relations': [],
            })
            event_id += 1
            continue

        # With spaCy: extract events anchored by verbs
        doc = nlp(verse_text)
        seen_sents = set()

        for token in doc:
            if token.pos_ == 'VERB':
                sentence_text = token.sent.text.strip()
                # Deduplicate within verse
                sent_key = (verse_idx, sentence_text)
                if sent_key in seen_sents:
                    continue
                seen_sents.add(sent_key)

                # ISO-TimeML EVENT attributes
                event_class = _classify_event(token, sentence_text)
                tense = _extract_tense(token)
                aspect = _extract_aspect(token)
                polarity = _extract_polarity(sentence_text)

                annotations.append({
                    'id': f"{doc_id}_e{event_id}",
                    'text': sentence_text,
                    'verb_lemma': token.lemma_,
                    'verse_idx': verse_idx,
                    'chapter': chapter,
                    'verse': verse_num,
                    'verse_ref': f"{chapter}:{verse_num}",
                    'type': 'EVENT',
                    'event_class': event_class,
                    'tense': tense,
                    'aspect': aspect,
                    'polarity': polarity,
                    'relations': [],
                })
                event_id += 1

    # ---- Generate TLINKs (intra-document temporal ordering) ----
    tlinks = _generate_tlinks(annotations, signal_annos, doc_id)
    annotations.extend(tlinks)

    # ---- Generate SLINKs (subordination) ----
    slinks = _generate_slinks(annotations, doc_id)
    annotations.extend(slinks)

    return annotations


def run_annotation(docs: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """Run ISO-TimeML annotation on all Gospel documents.

    Parameters
    ----------
    docs : dict
        Mapping of document ID → file path.

    Returns
    -------
    dict
        Mapping of document ID → list of annotation dicts.
    """
    annotated_docs = {}
    for doc_id, path in docs.items():
        annotated_docs[doc_id] = annotate_document(path, doc_id)
        n_events = sum(1 for a in annotated_docs[doc_id] if a['type'] == 'EVENT')
        n_timex = sum(1 for a in annotated_docs[doc_id] if a['type'] == 'TIMEX3')
        n_tlinks = sum(1 for a in annotated_docs[doc_id] if a['type'] == 'TLINK')
        n_slinks = sum(1 for a in annotated_docs[doc_id] if a['type'] == 'SLINK')
        n_signals = sum(1 for a in annotated_docs[doc_id] if a['type'] == 'SIGNAL')
        print(f"  {doc_id}: {n_events} EVENTs, {n_timex} TIMEX3s, "
              f"{n_signals} SIGNALs, {n_tlinks} TLINKs, {n_slinks} SLINKs")
    return annotated_docs
