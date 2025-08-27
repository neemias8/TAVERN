import os
import xml.etree.ElementTree as ET
from stage1_temporal_annotation.annotator import run_annotation
from stage2_event_alignment.aligner import align_events
from stage3_gnn_modeling.gnn_model import run_gnn
from stage4_abstractive_generation.generator import generate_summary
from stage5_evaluation.evaluator import evaluate_summary

def load_chronology_xml(xml_path):
    """Load the chronology from XML into list of dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    table = []
    for event_elem in root.findall('.//event'):  # Use .// to find events at any level
        event_dict = {
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
    enriched_events, G, consolidated_events = run_gnn(all_events, chronology_table)
    print("Stage 3: GNN modeling complete.")
    
    # Stage 4
    summary = generate_summary(consolidated_events, G)
    print("Stage 4: Summary generated.")
    print(summary)
    
    # Stage 5
    scores = evaluate_summary(summary)
    print("Stage 5: ROUGE scores:", scores)

if __name__ == "__main__":
    main()