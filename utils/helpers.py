import xml.etree.ElementTree as ET
import re
import json
import torch
import numpy as np
import networkx as nx
import PyPDF2
from transformers import BartTokenizer

def parse_gospel_xml(file_path):
    """Parse XML to extract verses as list of strings."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    verses = []
    for verse in root.iter('verse'):
        if verse.text:
            verses.append(verse.text.strip())
    return verses

def parse_chronology_pdf(pdf_path):
    """Extract table from PDF into list of dicts using PyPDF2."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        full_text = ''
        for page in reader.pages:
            full_text += page.extract_text() + '\n'
    
    # Simple parsing: Split lines, detect rows based on pattern (ID, description, when_where, verses)
    lines = full_text.split('\n')
    table = []
    current_row = None
    for line in lines:
        line = line.strip()
        if re.match(r'^\d{4}', line):  # Starts with ID like 1351
            if current_row:
                table.append(current_row)
            parts = re.split(r'\s{2,}', line)  # Split on multiple spaces
            current_row = {
                'id': parts[0] if len(parts) > 0 else '',
                'description': parts[1] if len(parts) > 1 else '',
                'when_where': parts[2] if len(parts) > 2 else '',
                'matthew': parts[3] if len(parts) > 3 else '',
                'mark': parts[4] if len(parts) > 4 else '',
                'luke': parts[5] if len(parts) > 5 else '',
                'john': parts[6] if len(parts) > 6 else ''
            }
        elif current_row and line:  # Append to description if continuation
            current_row['description'] += ' ' + line
    if current_row:
        table.append(current_row)
    
    with open('outputs/parsed_chronology.json', 'w') as f:
        json.dump(table, f, indent=4)
    return table

def simple_word_embedding(text, tokenizer):
    """Get embedding using BART tokenizer (average token embeds)."""
    inputs = tokenizer(text, return_tensors="pt")
    embeds = torch.mean(inputs.input_ids.float(), dim=1)  # Simplified
    return embeds.detach().numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def create_graph(events):
    G = nx.DiGraph()  # Directed for temporal relations
    for event in events:
        G.add_node(event['id'], **event)
    return G