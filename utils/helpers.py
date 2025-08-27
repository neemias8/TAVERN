import xml.etree.ElementTree as ET
import re
import torch
import numpy as np
import networkx as nx
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