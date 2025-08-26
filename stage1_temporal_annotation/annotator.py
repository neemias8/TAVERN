import spacy
from utils.helpers import parse_gospel_xml

nlp = spacy.load("en_core_web_sm")

def annotate_document(doc_path, doc_id):
    verses = parse_gospel_xml(doc_path)
    annotations = []
    event_id = 0
    for verse_idx, verse in enumerate(verses):
        doc = nlp(verse)
        for token in doc:
            if token.pos_ == 'VERB':  # Events as verbs (ISO-TimeML inspired)
                annotations.append({
                    'id': f"{doc_id}_e{event_id}",
                    'text': token.text,
                    'verse_idx': verse_idx,
                    'type': 'EVENT',
                    'relations': []
                })
                event_id += 1
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:  # Times
                annotations.append({
                    'id': f"{doc_id}_t{event_id}",
                    'text': ent.text,
                    'verse_idx': verse_idx,
                    'type': 'TIME'
                })
                event_id += 1
        # Simple relations: Look for temporal words
        if re.search(r'\bbefore\b', verse.lower()):
            # Add BEFORE relation placeholder (filled later)
            pass
    return annotations

def run_annotation(docs):
    annotated_docs = {}
    for doc_id, path in docs.items():
        annotated_docs[doc_id] = annotate_document(path, doc_id)
    return annotated_docs