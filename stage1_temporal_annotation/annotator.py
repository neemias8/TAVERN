import spacy
import re
from utils.helpers import parse_gospel_xml

nlp = spacy.load("en_core_web_sm")

def annotate_document(doc_path, doc_id):
    verses_metadata = parse_gospel_xml(doc_path)
    annotations = []
    event_id = 0
    for verse_idx, verse_data in enumerate(verses_metadata):
        verse_text = verse_data['text']
        chapter = verse_data['chapter']
        verse_num = verse_data['verse']
        
        doc = nlp(verse_text)
        for token in doc:
            if token.pos_ == 'VERB':  # Events as verbs (ISO-TimeML inspired)
                sentence_text = token.sent.text.strip()
                is_duplicate = any(
                    ann['verse_idx'] == verse_idx and ann['text'] == sentence_text
                    for ann in annotations if ann['type'] == 'EVENT'
                )
                if not is_duplicate:
                    annotations.append({
                        'id': f"{doc_id}_e{event_id}",
                        'text': sentence_text,  # Use the whole sentence
                        'verse_idx': verse_idx,
                        'chapter': chapter,
                        'verse': verse_num,
                        'verse_ref': f"{chapter}:{verse_num}",  # Real chapter:verse reference
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
                    'verse_ref': f"{verse_idx + 1}",
                    'type': 'TIME'
                })
                event_id += 1
        # Simple relations: Look for temporal words
        if re.search(r'\bbefore\b', verse_text.lower()):
            # Add BEFORE relation placeholder (filled later)
            pass
    return annotations

def run_annotation(docs):
    annotated_docs = {}
    for doc_id, path in docs.items():
        annotated_docs[doc_id] = annotate_document(path, doc_id)
    return annotated_docs