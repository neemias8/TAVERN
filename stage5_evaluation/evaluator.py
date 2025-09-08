from rouge_score import rouge_scorer
import json

def evaluate_summary(generated_summary, reference_path='data/reference_summary.txt'):
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference = f.read().strip()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated_summary)
    return scores

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
