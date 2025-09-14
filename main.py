import os
import sys
import platform
import xml.etree.ElementTree as ET
from stage1_temporal_annotation.annotator import run_annotation
from stage2_event_alignment.aligner import align_events
from stage3_gnn_modeling.gnn_model import run_gnn
from stage4_abstractive_generation.generator import generate_summary
from stage5_evaluation.evaluator import evaluate_summary, save_scores, scores_to_dict, evaluate_bertscore, save_bertscore, evaluate_meteor, save_meteor, evaluate_kendalls_tau, save_kendalls_tau

def load_chronology_xml(xml_path):
    """Load the chronology from XML into list of dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    table = []
    for event_elem in root.findall('.//event'):  # Use .// to find events at any level
        event_dict = {
            'id': event_elem.get('id'),  # Capture the event ID if present
            'day': event_elem.find('day').text if event_elem.find('day') is not None else None,
            'description': event_elem.find('description').text if event_elem.find('description') is not None else None,
            'when_where': event_elem.find('when_where').text if event_elem.find('when_where') is not None else None,
            'matthew': event_elem.find('matthew').text if event_elem.find('matthew') is not None else None,
            'mark': event_elem.find('mark').text if event_elem.find('mark') is not None else None,
            'luke': event_elem.find('luke').text if event_elem.find('luke') is not None else None,
            'john': event_elem.find('john').text if event_elem.find('john') is not None else None
        }
        table.append(event_dict)
    return table

def main():
    # Ensure UTF-8 stdout/stderr to avoid UnicodeEncodeError on Windows consoles
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass
    try:
        if platform.system() == 'Windows':
            os.system('chcp 65001 >NUL')
    except Exception:
        pass
    os.makedirs('outputs', exist_ok=True)
    
    docs = {
        'matthew': 'data/EnglishNIVMatthew40_PW.xml',
        'mark': 'data/EnglishNIVMark41_PW.xml',
        'luke': 'data/EnglishNIVLuke42_PW.xml',
        'john': 'data/EnglishNIVJohn43_PW.xml'
    }
    
    # Load chronology from XML
    chronology_table = load_chronology_xml('data/ChronologyOfTheFourGospels_PW.xml')
    print("Chronology loaded from XML.")
    
    # Stage 1
    annotated_docs = run_annotation(docs)
    print("Stage 1: Annotations complete.")
    
    # Stage 2
    all_events, alignments = align_events(annotated_docs, chronology_table)
    print("Stage 2: Alignments complete.")
    
    # Stage 3
    enriched_events, G, consolidated_events = run_gnn(all_events, chronology_table, docs=docs)
    print("Stage 3: GNN modeling complete.")
    
    # Stage 4
    summary = generate_summary(consolidated_events, G)
    print("Stage 4: Summary generated.")
    print(summary)
    
    # Stage 5
    # ROUGE against Golden Sample
    rouge_scores = evaluate_summary(summary, reference_path='data/Golden_Sample.txt')
    print("Stage 5: ROUGE scores:", rouge_scores)
    
    # Persist metrics in outputs as JSON (UTF-8)
    try:
        out_metrics = save_scores(rouge_scores, out_path='outputs/rouge.json')
        print(f"Saved ROUGE metrics to {out_metrics}")
    except Exception as e:
        print(f"Warning: Failed to save ROUGE metrics: {e}")

    # BERTScore against Golden Sample
    try:
        bert = evaluate_bertscore(summary, reference_path='data/Golden_Sample.txt', lang='en', rescale_with_baseline=True)
        print("Stage 5: BERTScore:", bert)
        out_bs = save_bertscore(bert, out_path='outputs/bertscore.json')
        print(f"Saved BERTScore to {out_bs}")
    except Exception as e:
        print(f"Warning: Failed to compute/save BERTScore: {e}")


    # METEOR against Golden Sample
    try:
        meteor = evaluate_meteor(summary, reference_path='data/Golden_Sample.txt')
        print("Stage 5: METEOR:", meteor)
        out_meteor = save_meteor(meteor, out_path='outputs/meteor.json')
        print(f"Saved METEOR to {out_meteor}")
    except Exception as e:
        print(f"Warning: Failed to compute/save METEOR: {e}")

    # Kendall's Tau for event ordering
    try:
        kt = evaluate_kendalls_tau(consolidated_events, G)
        print("Stage 5: Kendall's Tau:", kt)
        out_kt = save_kendalls_tau(kt, out_path='outputs/kendall_tau.json')
        print(f"Saved Kendall's Tau to {out_kt}")
    except Exception as e:
        print(f"Warning: Failed to compute/save Kendall's Tau: {e}")
if __name__ == "__main__":
    main()
