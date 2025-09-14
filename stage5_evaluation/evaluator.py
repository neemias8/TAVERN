from rouge_score import rouge_scorer
import json
import re
from typing import List, Optional

def evaluate_summary(generated_summary, reference_path='data/reference_summary.txt'):
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference = f.read().strip()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated_summary)
    return scores

def evaluate_bertscore(generated_summary, reference_path='data/Golden_Sample.txt', lang='en', rescale_with_baseline=True):
    """Compute BERTScore (P/R/F1) against a reference file.

    Returns a dict with precision, recall, f1 as floats.
    """
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference = f.read().strip()
    # Lazy import to avoid dependency at import-time if unused
    from bert_score import score as bertscore_score
    P, R, F1 = bertscore_score([generated_summary], [reference], lang=lang, rescale_with_baseline=rescale_with_baseline)
    return {
        'precision': float(P.mean().item()),
        'recall': float(R.mean().item()),
        'f1': float(F1.mean().item())
    }

def scores_to_dict(scores):
    """Convert rouge_score Scores to a plain dict of floats."""
    out = {}
    for k, v in scores.items():
        try:
            out[k] = {
                'precision': float(v.precision),
                'recall': float(v.recall),
                'f1': float(v.fmeasure),
            }
        except Exception:
            # Already a dict or unexpected type
            try:
                out[k] = {
                    'precision': float(v.get('precision', 0.0)),
                    'recall': float(v.get('recall', 0.0)),
                    'f1': float(v.get('f1', 0.0)),
                }
            except Exception:
                pass
    return out

def save_scores(scores, out_path='outputs/rouge.json'):
    """Save ROUGE scores to JSON (UTF-8)."""
    data = scores_to_dict(scores)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path

def save_bertscore(scores, out_path='outputs/bertscore.json'):
    """Save BERTScore metrics (P/R/F1) to JSON (UTF-8)."""
    # scores is already a plain dict of floats
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    return out_path


# -------------------- METEOR --------------------

def _ensure_wordnet(download: bool = True) -> bool:
    '''Ensure NLTK WordNet (and OMW) are available; optionally download.'''
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets('test')
            return True
        except LookupError:
            if download:
                try:
                    nltk.download('wordnet', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
                    wn.synsets('test')
                    return True
                except Exception:
                    return False
            return False
    except Exception:
        return False

def _simple_tokenize(text: str) -> List[str]:
    '''Lightweight tokenizer (letters/digits underscore), lowercased.'''
    if not text:
        return []
    return re.findall(r'\w+', text.lower())

def evaluate_meteor(generated_summary: str, reference_path: str = 'data/Golden_Sample.txt'):
    '''Compute METEOR between generated text and a reference file.

    Uses NLTK's implementation with simple regex tokenization to avoid
    requiring NLTK's punkt models. Returns a dict: { 'meteor': float }.'''
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference = f.read().strip()

    # Ensure WordNet corpora (optional) for synonym matches
    _ensure_wordnet(download=True)

    # Lazy import to avoid hard dependency if unused
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
    except Exception as e:
        raise RuntimeError(f'NLTK METEOR not available: {e}')

    ref_tokens = _simple_tokenize(reference)
    hyp_tokens = _simple_tokenize(generated_summary)
    score = float(nltk_meteor_score([ref_tokens], hyp_tokens)) if hyp_tokens else 0.0
    return { 'meteor': score }

def save_meteor(scores, out_path='outputs/meteor.json'):
    '''Save METEOR score to JSON (UTF-8).'''
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    return out_path
# ---------------- Kendall's Tau (ordering) ----------------

def _kendalls_tau_fallback(x: List[int], y: List[int]) -> float:
    '''Compute Kendall's Tau-b (without ties handling) as a fallback.

    Assumes x and y are permutations of the same set; returns tau in [-1, 1].
    '''
    n = len(x)
    if n < 2:
        return 0.0
    # Build order maps
    rank_y = { val: i for i, val in enumerate(y) }
    # Count concordant/discordant pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = x[i], x[j]
            yi, yj = rank_y[xi], rank_y[xj]
            if (xj - xi) * (yj - yi) > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom

def evaluate_kendalls_tau(consolidated_events: List[dict], G=None):
    '''Compute Kendall's Tau between predicted event order and chronology.

    - If a graph G is provided (NetworkX DiGraph), uses topological sort as the
      predicted order. Otherwise, uses the order of consolidated_events.
    - Each consolidated event must have a unique integer chronology_index.

    Returns dict: { 'tau': float, 'p_value': Optional[float], 'n': int }
    '''
    # Resolve predicted order of events
    predicted_ids: List[str] = []
    idx_by_id = {}
    if consolidated_events:
        idx_by_id = { ev['id']: ev.get('chronology_index') for ev in consolidated_events if 'id' in ev }

    if G is not None:
        try:
            # Lazy import to avoid hard dependency at import-time
            import networkx as nx  # noqa: F401
            predicted_ids = list(G.nodes())
            # If graph is directed and a DAG, try topological order
            try:
                predicted_ids = list(nx.topological_sort(G))
            except Exception:
                # Keep insertion order of nodes
                pass
        except Exception:
            # Fallback to consolidated_events order
            predicted_ids = [ev['id'] for ev in consolidated_events if 'id' in ev]
    else:
        predicted_ids = [ev['id'] for ev in consolidated_events if 'id' in ev]

    # Map to chronology indices
    pred_indices = []
    for eid in predicted_ids:
        ci = idx_by_id.get(eid)
        if isinstance(ci, int):
            pred_indices.append(ci)

    # Ensure we have a clean permutation 0..n-1 if possible
    if not pred_indices:
        return { 'tau': 0.0, 'p_value': None, 'n': 0 }

    gold_indices = sorted(pred_indices)

    # Try SciPy first, fallback otherwise
    tau = None
    p_value: Optional[float] = None
    try:
        from scipy.stats import kendalltau
        r = kendalltau(pred_indices, gold_indices)
        tau = float(getattr(r, 'correlation', getattr(r, 'statistic', 0.0)))
        p_value = float(getattr(r, 'pvalue', None)) if hasattr(r, 'pvalue') else None
    except Exception:
        tau = float(_kendalls_tau_fallback(pred_indices, gold_indices))
        p_value = None

    return { 'tau': tau, 'p_value': p_value, 'n': len(pred_indices) }

def save_kendalls_tau(scores, out_path='outputs/kendall_tau.json'):
    '''Save Kendall's Tau result to JSON (UTF-8).'''
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    return out_path


