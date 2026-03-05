"""
TAVERN — Stage 5: Evaluation Suite
====================================
Computes evaluation metrics for TAVERN pipeline outputs.

Metrics
-------
* **Temporal Ordering Accuracy (TOA)** — fraction of event pairs whose
  predicted TLINK relation matches the gold standard.
* **Temporal F1** — precision/recall/F1 over predicted vs gold TLINK types.
* **Entity Coverage** — fraction of named entities in the gold corpus
  covered by the generated narrative.
* **Narrative Coherence Score (NCS)** — average consecutive-sentence
  cosine similarity in the generated text (proxy for fluency).
* **Adjacent Pair Accuracy (APA)** — accuracy on adjacent (narrative-order)
  event pairs only.
* **Redundancy Ratio (RR)** — fraction of near-duplicate sentences in the
  generated narrative (lower is better).
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.helpers import get_embedding, cosine_similarity

# ---------------------------------------------------------------------------
# Temporal ordering accuracy
# ---------------------------------------------------------------------------

def temporal_ordering_accuracy(
        predicted_tlinks: List[Dict[str, Any]],
        gold_tlinks: List[Dict[str, Any]],
) -> float:
    """Fraction of predicted TLINKs that match the gold on (eventID, relatedToEvent).

    Only evaluates pairs that appear in both predicted and gold sets.
    """
    gold_map: Dict[Tuple, str] = {}
    for tl in gold_tlinks:
        key = (tl.get('eventID', ''), tl.get('relatedToEvent', ''))
        if key[0] and key[1]:
            gold_map[key] = tl.get('relType', '')

    if not gold_map:
        return 0.0

    correct = 0
    evaluated = 0
    for tl in predicted_tlinks:
        key = (tl.get('eventID', ''), tl.get('relatedToEvent', ''))
        if key in gold_map:
            evaluated += 1
            if tl.get('relType', '') == gold_map[key]:
                correct += 1

    return correct / evaluated if evaluated > 0 else 0.0


# ---------------------------------------------------------------------------
# Temporal F1
# ---------------------------------------------------------------------------

def temporal_f1(
        predicted_tlinks: List[Dict[str, Any]],
        gold_tlinks: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Precision, recall, and F1 over predicted vs gold TLINKs.

    A predicted TLINK is a true positive iff (eventID, relatedToEvent, relType)
    all match the gold.
    """
    gold_set = {
        (tl.get('eventID', ''), tl.get('relatedToEvent', ''), tl.get('relType', ''))
        for tl in gold_tlinks
        if tl.get('eventID') and tl.get('relatedToEvent')
    }
    pred_set = {
        (tl.get('eventID', ''), tl.get('relatedToEvent', ''), tl.get('relType', ''))
        for tl in predicted_tlinks
        if tl.get('eventID') and tl.get('relatedToEvent')
    }

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


# ---------------------------------------------------------------------------
# Entity coverage
# ---------------------------------------------------------------------------

def entity_coverage(
        gold_annotations: List[Dict[str, Any]],
        generated_narrative: str,
) -> float:
    """Fraction of named entities (from gold EVENTs) covered by the narrative.

    Uses a capitalised-word heuristic for entity detection.
    """
    def _entities(text: str) -> set:
        return {w.rstrip('.,;:') for w in re.findall(r'\b[A-Z][a-z]+\b', text)}

    gold_entities: set = set()
    for ann in gold_annotations:
        if ann.get('type') == 'EVENT':
            gold_entities |= _entities(ann.get('text', ''))

    if not gold_entities:
        return 1.0  # Nothing to cover

    narrative_entities = _entities(generated_narrative)
    covered = gold_entities & narrative_entities
    return len(covered) / len(gold_entities)


# ---------------------------------------------------------------------------
# Narrative coherence score
# ---------------------------------------------------------------------------

def narrative_coherence_score(narrative: str) -> float:
    """Average cosine similarity between consecutive sentences.

    Higher values indicate more coherent / topically consistent text.
    A score of 0.0 is returned for narratives with fewer than 2 sentences.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', narrative)
                 if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 0.0

    similarities = []
    for s1, s2 in zip(sentences, sentences[1:]):
        e1 = get_embedding(s1)
        e2 = get_embedding(s2)
        similarities.append(cosine_similarity(e1, e2))

    return float(np.mean(similarities))


# ---------------------------------------------------------------------------
# Adjacent Pair Accuracy
# ---------------------------------------------------------------------------

def adjacent_pair_accuracy(
        predicted_tlinks: List[Dict[str, Any]],
        gold_tlinks: List[Dict[str, Any]],
) -> float:
    """Accuracy restricted to adjacent (narrative-order) event pairs.

    Considers only TLINKs with ``source == 'narrative_order'`` in the
    predicted set, matched against gold TLINKs.
    """
    adjacent_pred = [
        tl for tl in predicted_tlinks
        if tl.get('source') == 'narrative_order'
    ]
    return temporal_ordering_accuracy(adjacent_pred, gold_tlinks)


# ---------------------------------------------------------------------------
# Redundancy Ratio
# ---------------------------------------------------------------------------

def redundancy_ratio(
        narrative: str,
        similarity_threshold: float = 0.85,
) -> float:
    """Fraction of sentences that are near-duplicates of an earlier sentence.

    Two sentences are considered near-duplicates if their cosine similarity
    exceeds *similarity_threshold*.

    Returns 0.0 for narratives with fewer than 2 sentences.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', narrative)
                 if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 0.0

    embeddings = [get_embedding(s) for s in sentences]
    redundant = 0

    for i in range(1, len(sentences)):
        for j in range(i):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                redundant += 1
                break  # Count each sentence at most once

    return redundant / len(sentences)


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------

def evaluate(
        annotated_docs: Dict[str, List[Dict[str, Any]]],
        alignments: List[Dict[str, Any]],
        generated_narrative: str,
        gold_tlinks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full TAVERN evaluation suite.

    Parameters
    ----------
    annotated_docs : dict
        ``{doc_id: [annotation_dict, ...]}``, output of Stage 1.
    alignments : list of dict
        Cross-document alignments, output of Stage 2.
    generated_narrative : str
        Output of Stage 4.
    gold_tlinks : list of dict or None
        Ground-truth TLINKs for supervised metrics. If ``None``, supervised
        metrics (TOA, F1, APA) are skipped.

    Returns
    -------
    dict
        Metric name → value.
    """
    # Gather predicted TLINKs
    predicted_tlinks: List[Dict] = []
    all_gold_annotations: List[Dict] = []

    for doc_id, annotations in annotated_docs.items():
        for ann in annotations:
            if ann['type'] == 'TLINK':
                predicted_tlinks.append(ann)
            all_gold_annotations.append(ann)

    results: Dict[str, Any] = {}

    # Supervised metrics
    if gold_tlinks:
        results['temporal_ordering_accuracy'] = temporal_ordering_accuracy(
            predicted_tlinks, gold_tlinks)
        tf1 = temporal_f1(predicted_tlinks, gold_tlinks)
        results['temporal_precision'] = tf1['precision']
        results['temporal_recall'] = tf1['recall']
        results['temporal_f1'] = tf1['f1']
        results['adjacent_pair_accuracy'] = adjacent_pair_accuracy(
            predicted_tlinks, gold_tlinks)

    # Unsupervised metrics
    results['entity_coverage'] = entity_coverage(
        all_gold_annotations, generated_narrative)
    results['narrative_coherence_score'] = narrative_coherence_score(
        generated_narrative)
    results['redundancy_ratio'] = redundancy_ratio(generated_narrative)

    # Summary stats
    total_events = sum(
        sum(1 for a in anns if a['type'] == 'EVENT')
        for anns in annotated_docs.values()
    )
    total_tlinks = len(predicted_tlinks)
    results['total_events'] = total_events
    results['total_tlinks'] = total_tlinks
    results['total_alignments'] = len(alignments)

    return results
