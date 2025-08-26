from rouge_score import rouge_scorer

def evaluate_summary(generated_summary, reference_path='data/reference_summary.txt'):
    with open(reference_path, 'r') as f:
        reference = f.read().strip()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated_summary)
    return scores