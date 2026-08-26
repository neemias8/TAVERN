"""
TAVERN — Main Pipeline
========================
Temporal Anchoring for Version Consolidation in
Abstractive Multi-Document Summarization

Orchestrates the full 6-stage pipeline:
  1. Temporal Annotation (ISO-TimeML)
  2. Cross-Document Event Alignment
  3. Graph Construction & GNN Processing
  4. Temporally-Guided Abstractive Summarization
  5. Evaluation (ROUGE, BERTScore, METEOR, Kendall's Tau,
     Adjacent Pair Accuracy, Redundancy)
"""

import os
import sys
import platform
import xml.etree.ElementTree as ET

from stage1_temporal_annotation.annotator import run_annotation
from stage2_event_alignment.aligner import align_events
from stage3_gnn_modeling.gnn_model import run_gnn
from stage4_abstractive_generation.generator import generate_summary
from stage5_evaluation.evaluator import (
    evaluate_summary, save_scores,
    evaluate_bertscore, save_bertscore,
    evaluate_meteor, save_meteor,
    evaluate_kendalls_tau, save_kendalls_tau,
    evaluate_adjacent_pair_accuracy, save_adjacent_pair_accuracy,
    evaluate_redundancy, save_redundancy,
)


def load_chronology_xml(xml_path: str) -> list:
    """Load the Aschmann chronology from XML into a list of dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    table = []
    for event_elem in root.findall('.//event'):
        event_dict = {
            'id': event_elem.get('id'),
            'day': (event_elem.find('day').text
                    if event_elem.find('day') is not None else None),
            'description': (event_elem.find('description').text
                            if event_elem.find('description') is not None else None),
            'when_where': (event_elem.find('when_where').text
                           if event_elem.find('when_where') is not None else None),
            'matthew': (event_elem.find('matthew').text
                        if event_elem.find('matthew') is not None else None),
            'mark': (event_elem.find('mark').text
                     if event_elem.find('mark') is not None else None),
            'luke': (event_elem.find('luke').text
                     if event_elem.find('luke') is not None else None),
            'john': (event_elem.find('john').text
                     if event_elem.find('john') is not None else None),
        }
        table.append(event_dict)
    return table


def main():
    # ---- Encoding setup ----
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass
    if platform.system() == 'Windows':
        try:
            os.system('chcp 65001 >NUL')
        except Exception:
            pass

    os.makedirs('outputs', exist_ok=True)

    # ---- Input documents ----
    docs = {
        'matthew': 'data/EnglishNIVMatthew40_PW.xml',
        'mark':    'data/EnglishNIVMark41_PW.xml',
        'luke':    'data/EnglishNIVLuke42_PW.xml',
        'john':    'data/EnglishNIVJohn43_PW.xml',
    }

    # ---- Load chronology ----
    chronology_table = load_chronology_xml('data/ChronologyOfTheFourGospels_PW.xml')
    print(f"Chronology loaded: {len(chronology_table)} events.\n")

    # ==================================================================
    # Stage 1: Temporal Annotation (ISO-TimeML)
    # ==================================================================
    print("=" * 60)
    print("Stage 1: Temporal Annotation (ISO 24617-1 / ISO-TimeML)")
    print("=" * 60)
    annotated_docs = run_annotation(docs)
    total_events = sum(
        sum(1 for a in annos if a['type'] == 'EVENT')
        for annos in annotated_docs.values()
    )
    total_timex = sum(
        sum(1 for a in annos if a['type'] == 'TIMEX3')
        for annos in annotated_docs.values()
    )
    total_tlinks = sum(
        sum(1 for a in annos if a['type'] == 'TLINK')
        for annos in annotated_docs.values()
    )
    print(f"\n  Total: {total_events} EVENTs, {total_timex} TIMEX3s, "
          f"{total_tlinks} TLINKs\n")

    # ==================================================================
    # Stage 2: Cross-Document Event Alignment
    # ==================================================================
    print("=" * 60)
    print("Stage 2: Cross-Document Event Alignment")
    print("=" * 60)
    all_events, alignments = align_events(annotated_docs, chronology_table)
    print(f"  Aligned {len(alignments)} event pairs.\n")

    # ==================================================================
    # Stage 3: Graph Construction & GNN Processing
    # ==================================================================
    print("=" * 60)
    print("Stage 3: Graph Construction & Relational GAT Processing")
    print("=" * 60)
    enriched_events, G, consolidated_events = run_gnn(
        all_events, chronology_table, docs=docs)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # ==================================================================
    # Stage 4: Temporally-Guided Abstractive Summarization
    # ==================================================================
    print("=" * 60)
    print("Stage 4: Temporally-Guided Abstractive Summarization")
    print("=" * 60)
    summary = generate_summary(
        consolidated_events, G, enriched_embeddings=enriched_events)
    print(f"\n  Summary length: {len(summary)} characters")
    print(f"  Preview: {summary[:300]}...\n")

    # ==================================================================
    # Stage 5: Evaluation
    # ==================================================================
    print("=" * 60)
    print("Stage 5: Evaluation")
    print("=" * 60)

    # Collect all metrics into a unified dict for the summary table
    all_metrics: dict = {}

    # --- ROUGE ---
    try:
        rouge_scores = evaluate_summary(
            summary, reference_path='data/Golden_Sample.txt')
        print(f"  ROUGE: {rouge_scores}")
        save_scores(rouge_scores, out_path='outputs/rouge.json')
        for key in ('rouge1', 'rouge2', 'rougeL'):
            try:
                score_obj = rouge_scores[key]
                all_metrics[f'{key}_precision'] = float(score_obj.precision)
                all_metrics[f'{key}_recall'] = float(score_obj.recall)
                all_metrics[f'{key}_f1'] = float(score_obj.fmeasure)
            except Exception:
                try:
                    all_metrics[f'{key}_f1'] = float(score_obj['f1'])
                    all_metrics[f'{key}_precision'] = float(score_obj['precision'])
                    all_metrics[f'{key}_recall'] = float(score_obj['recall'])
                except Exception:
                    pass
    except Exception as e:
        print(f"  Warning: ROUGE failed: {e}")

    # --- BERTScore ---
    try:
        bert = evaluate_bertscore(
            summary, reference_path='data/Golden_Sample.txt',
            lang='en', rescale_with_baseline=True)
        print(f"  BERTScore: {bert}")
        save_bertscore(bert, out_path='outputs/bertscore.json')
        all_metrics['bertscore_precision'] = bert.get('precision', 0.0)
        all_metrics['bertscore_recall'] = bert.get('recall', 0.0)
        all_metrics['bertscore_f1'] = bert.get('f1', 0.0)
    except Exception as e:
        print(f"  Warning: BERTScore failed: {e}")

    # --- METEOR ---
    try:
        meteor = evaluate_meteor(
            summary, reference_path='data/Golden_Sample.txt')
        print(f"  METEOR: {meteor}")
        save_meteor(meteor, out_path='outputs/meteor.json')
        all_metrics['meteor'] = meteor.get('meteor', 0.0)
    except Exception as e:
        print(f"  Warning: METEOR failed: {e}")

    # --- Kendall's Tau (temporal ordering) ---
    try:
        kt = evaluate_kendalls_tau(consolidated_events, G)
        print(f"  Kendall's Tau: {kt}")
        save_kendalls_tau(kt, out_path='outputs/kendall_tau.json')
        all_metrics['kendalls_tau'] = kt.get('tau', 0.0)
        all_metrics['kendalls_tau_p_value'] = kt.get('p_value')
        all_metrics['kendalls_tau_n'] = kt.get('n', 0)
    except Exception as e:
        print(f"  Warning: Kendall's Tau failed: {e}")

    # --- Adjacent Pair Accuracy (temporal coherence) ---
    try:
        apa = evaluate_adjacent_pair_accuracy(consolidated_events, G)
        print(f"  Adjacent Pair Accuracy: {apa}")
        save_adjacent_pair_accuracy(apa, out_path='outputs/adjacent_pair_accuracy.json')
        all_metrics['adjacent_pair_accuracy'] = apa.get('accuracy', 0.0)
        all_metrics['adjacent_pair_correct'] = apa.get('correct_pairs', 0)
        all_metrics['adjacent_pair_total'] = apa.get('total_pairs', 0)
    except Exception as e:
        print(f"  Warning: Adjacent Pair Accuracy failed: {e}")

    # --- Redundancy ---
    try:
        redund = evaluate_redundancy(summary)
        print(f"  Redundancy: {redund}")
        save_redundancy(redund, out_path='outputs/redundancy.json')
        all_metrics['ngram_redundancy'] = redund.get('ngram_redundancy_ratio', 0.0)
        all_metrics['sentence_redundancy'] = redund.get('sentence_redundancy_ratio', 0.0)
    except Exception as e:
        print(f"  Warning: Redundancy metric failed: {e}")

    # ==================================================================
    # Save unified metrics summary (CSV + JSON)
    # ==================================================================
    import json as _json
    import csv as _csv

    summary_rows = [
        ('ROUGE-1', 'Precision', all_metrics.get('rouge1_precision', '')),
        ('ROUGE-1', 'Recall', all_metrics.get('rouge1_recall', '')),
        ('ROUGE-1', 'F1', all_metrics.get('rouge1_f1', '')),
        ('ROUGE-2', 'Precision', all_metrics.get('rouge2_precision', '')),
        ('ROUGE-2', 'Recall', all_metrics.get('rouge2_recall', '')),
        ('ROUGE-2', 'F1', all_metrics.get('rouge2_f1', '')),
        ('ROUGE-L', 'Precision', all_metrics.get('rougeL_precision', '')),
        ('ROUGE-L', 'Recall', all_metrics.get('rougeL_recall', '')),
        ('ROUGE-L', 'F1', all_metrics.get('rougeL_f1', '')),
        ('BERTScore', 'Precision', all_metrics.get('bertscore_precision', '')),
        ('BERTScore', 'Recall', all_metrics.get('bertscore_recall', '')),
        ('BERTScore', 'F1', all_metrics.get('bertscore_f1', '')),
        ('METEOR', 'Score', all_metrics.get('meteor', '')),
        ("Kendall's Tau", 'Tau', all_metrics.get('kendalls_tau', '')),
        ("Kendall's Tau", 'p-value', all_metrics.get('kendalls_tau_p_value', '')),
        ("Kendall's Tau", 'N events', all_metrics.get('kendalls_tau_n', '')),
        ('Adj. Pair Accuracy', 'Accuracy', all_metrics.get('adjacent_pair_accuracy', '')),
        ('Adj. Pair Accuracy', 'Correct / Total',
         f"{all_metrics.get('adjacent_pair_correct', '')}/{all_metrics.get('adjacent_pair_total', '')}"),
        ('Redundancy (3-gram)', 'Ratio', all_metrics.get('ngram_redundancy', '')),
        ('Redundancy (sentence)', 'Ratio', all_metrics.get('sentence_redundancy', '')),
    ]
    csv_path = 'outputs/metrics_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = _csv.writer(cf)
        writer.writerow(['Metric', 'Measure', 'Value'])
        for row in summary_rows:
            val = row[2]
            if isinstance(val, float):
                val = f"{val:.4f}"
            writer.writerow([row[0], row[1], val])
    print(f"\n  Metrics summary saved: {csv_path}")

    json_path = 'outputs/metrics_summary.json'
    with open(json_path, 'w', encoding='utf-8') as jf:
        _json.dump(all_metrics, jf, ensure_ascii=False, indent=2, default=str)
    print(f"  Metrics JSON saved: {json_path}")

    # --- Print summary table to console ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Measure':<12} {'Value'}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for row in summary_rows:
        val = row[2]
        if isinstance(val, float):
            val = f"{val:.4f}"
        print(f"  {row[0]:<25} {row[1]:<12} {val}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs saved to ./outputs/")
    print("  - summary.txt                  (final narrative)")
    print("  - consolidation_details.json   (per-event consolidation log)")
    print("  - consolidation_details.csv    (per-event consolidation log)")
    print("  - metrics_summary.csv          (all metrics in one table)")
    print("  - metrics_summary.json         (all metrics as JSON)")
    print("  - rouge.json / bertscore.json / meteor.json / ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
