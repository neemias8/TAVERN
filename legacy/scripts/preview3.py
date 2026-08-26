import os
import sys
from pathlib import Path
# Ensure project root on sys.path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import xml.etree.ElementTree as ET

from stage1_temporal_annotation.annotator import run_annotation
from stage2_event_alignment.aligner import align_events
from stage3_gnn_modeling.gnn_model import run_gnn
from stage4_abstractive_generation.generator import generate_summary


def load_chronology_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    table = []
    for event_elem in root.findall('.//event'):
        event_dict = {
            'id': event_elem.get('id'),
            'day': event_elem.find('day').text if event_elem.find('day') is not None else None,
            'description': event_elem.find('description').text if event_elem.find('description') is not None else None,
            'when_where': event_elem.find('when_where').text if event_elem.find('when_where') is not None else None,
            'matthew': event_elem.find('matthew').text if event_elem.find('matthew') is not None else None,
            'mark': event_elem.find('mark').text if event_elem.find('mark') is not None else None,
            'luke': event_elem.find('luke').text if event_elem.find('luke') is not None else None,
            'john': event_elem.find('john').text if event_elem.find('john') is not None else None,
        }
        table.append(event_dict)
    return table


def main():
    os.makedirs('outputs', exist_ok=True)
    docs = {
        'matthew': 'data/EnglishNIVMatthew40_PW.xml',
        'mark': 'data/EnglishNIVMark41_PW.xml',
        'luke': 'data/EnglishNIVLuke42_PW.xml',
        'john': 'data/EnglishNIVJohn43_PW.xml',
    }
    chronology = load_chronology_xml('data/ChronologyOfTheFourGospels_PW.xml')
    # Limit to first 3 chronology rows
    chronology3 = chronology[:3]

    annotated_docs = run_annotation(docs)
    all_events, alignments = align_events(annotated_docs, chronology3)
    enriched_events, G, consolidated_events = run_gnn(all_events, chronology3)

    summary = generate_summary(consolidated_events, G)
    # Print only first three numbered sections
    import re
    parts = []
    for n in [1, 2, 3]:
        pattern = rf"\b{n}\s+(.+?)(?=\s+{n+1}\s+|$)"
        m = re.search(pattern, summary, flags=re.DOTALL)
        if m:
            parts.append(m.group(1).strip())
    for i, p in enumerate(parts, 1):
        print(f"EVENT {i}: {p}")


if __name__ == "__main__":
    main()
