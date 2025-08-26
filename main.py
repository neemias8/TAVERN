import os
from stage1_temporal_annotation.annotator import run_annotation
from stage2_event_alignment.aligner import align_events
from stage3_gnn_modeling.gnn_model import run_gnn
from stage4_abstractive_generation.generator import generate_summary
from stage5_evaluation.evaluator import evaluate_summary
from utils.helpers import parse_chronology_pdf

def main():
    os.makedirs('outputs', exist_ok=True)
    
    docs = {
        'matthew': 'data/EnglishNIVMatthew40_PW.xml',
        'mark': 'data/EnglishNIVMark41_PW.xml',
        'luke': 'data/EnglishNIVLuke42_PW.xml',
        'john': 'data/EnglishNIVJohn43_PW.xml'
    }
    
    # Parse chronology PDF
    chronology_table = parse_chronology_pdf('data/ChronologyOfTheFourGospels.pdf')
    print("Chronology parsed.")
    
    # Stage 1
    annotated_docs = run_annotation(docs)
    print("Stage 1: Annotations complete.")
    
    # Stage 2
    all_events, alignments = align_events(annotated_docs, chronology_table)
    print("Stage 2: Alignments complete.")
    
    # Stage 3
    enriched_events = run_gnn(all_events, chronology_table)
    print("Stage 3: GNN modeling complete.")
    
    # Stage 4
    summary = generate_summary(enriched_events, annotated_docs, chronology_table)
    print("Stage 4: Summary generated.")
    print(summary)
    
    # Stage 5
    scores = evaluate_summary(summary)
    print("Stage 5: ROUGE scores:", scores)

if __name__ == "__main__":
    main()