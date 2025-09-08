import xml.etree.ElementTree as ET
import re
import numpy as np
import networkx as nx
import hashlib

def parse_gospel_xml(file_path):
    """Parse XML to extract verses with chapter and verse information."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    verses_with_metadata = []
    
    for chapter in root.iter('chapter'):
        chapter_num = int(chapter.get('number', 0))
        for verse in chapter.iter('verse'):
            verse_num = int(verse.get('number', 0))
            if verse.text:
                verses_with_metadata.append({
                    'text': verse.text.strip(),
                    'chapter': chapter_num,
                    'verse': verse_num
                })
    
    return verses_with_metadata

def simple_word_embedding(text: str, dim: int = 300) -> np.ndarray:
    """Compute a deterministic, lightweight hashed bag-of-words embedding.

    - No external models required; works offline.
    - Uses signed hashing trick into a fixed-size vector and L2 normalizes it.
    """
    if not text:
        return np.zeros(dim, dtype=np.float32)
    # Tokenize on word chars
    tokens = re.findall(r"\w+", text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 1) & 1) else -1.0
        vec[idx] += sign
    # L2 normalize (avoid div by zero)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def create_graph(events):
    G = nx.DiGraph()  # Directed for temporal relations
    for event in events:
        G.add_node(event['id'], **event)
    return G

def parse_verse_range(verse_ref: str):
    """Parse verse references like '21:1-7', '15:18-16:4', '21:19a', '21:19b'.

    Returns tuple: (start_chapter, start_verse, end_chapter, end_verse, start_part, end_part)
    where parts are 'a', 'b', or None. Returns None on parse failure.
    """
    if not verse_ref or not verse_ref.strip():
        return None
    try:
        verse_ref = verse_ref.strip()
        # Cross-chapter pattern e.g., 15:18-16:4
        m = re.match(r'(\d+):(\d+)([ab]?)-(\d+):(\d+)([ab]?)$', verse_ref)
        if m:
            sc = int(m.group(1)); sv = int(m.group(2)); sp = m.group(3) or None
            ec = int(m.group(4)); ev = int(m.group(5)); ep = m.group(6) or None
            return sc, sv, ec, ev, sp, ep
        # Single chapter
        parts = verse_ref.split(':')
        if len(parts) != 2:
            return None
        chapter = int(parts[0])
        vr = parts[1]
        # Extract suffix if any
        m2 = re.match(r'^(\d+)([ab]?)$', vr)
        if m2:
            v = int(m2.group(1)); p = (m2.group(2) or None)
            return chapter, v, chapter, v, p, p
        # Range within chapter, possibly suffixes
        if '-' in vr:
            a, b = vr.split('-', 1)
            ma = re.match(r'^(\d+)([ab]?)$', a.strip())
            mb = re.match(r'^(\d+)([ab]?)$', b.strip())
            if not (ma and mb):
                return None
            sv = int(ma.group(1)); sp = (ma.group(2) or None)
            ev = int(mb.group(1)); ep = (mb.group(2) or None)
            return chapter, sv, chapter, ev, sp, ep
        return None
    except Exception:
        return None
