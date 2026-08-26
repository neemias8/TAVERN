import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage5_evaluation.evaluator import evaluate_summary, save_scores

def main():
    with io.open('outputs/summary.txt', 'r', encoding='utf-8') as f:
        summary = f.read()
    scores = evaluate_summary(summary)
    path = save_scores(scores, 'outputs/rouge.json')
    print('Saved', path)

if __name__ == '__main__':
    main()
