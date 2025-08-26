from utils.helpers import simple_word_embedding, cosine_similarity, parse_chronology_pdf
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def align_events(annotated_docs, chronology_table, threshold=0.7):
    all_events = []
    for doc_id, annos in annotated_docs.items():
        for anno in annos:
            if anno['type'] == 'EVENT':
                anno['doc_id'] = doc_id
                anno['embedding'] = simple_word_embedding(anno['text'], tokenizer)
                all_events.append(anno)
    
    alignments = {}
    # Use chronology to align parallels
    for row in chronology_table:
        parallels = {
            'matthew': row['matthew'],
            'mark': row['mark'],
            'luke': row['luke'],
            'john': row['john']
        }
        parallel_events = [e for e in all_events if any(p in e.get('verse_ref', '') for p in parallels.values() if p)]  # Match verse refs
        for i, e1 in enumerate(parallel_events):
            for e2 in parallel_events[i+1:]:
                sim = cosine_similarity(e1['embedding'], e2['embedding'])
                if sim > threshold:
                    relation = 'SIMULTANEOUS'  # Parallels in same row
                    e1['relations'].append({'to': e2['id'], 'type': relation})
                    e2['relations'].append({'to': e1['id'], 'type': relation})
                    alignments[(e1['id'], e2['id'])] = sim
    
    # Add cosine-based alignments for non-chronology matches
    for i, e1 in enumerate(all_events):
        for j, e2 in enumerate(all_events[i+1:]):
            if e1['doc_id'] != e2['doc_id']:
                sim = cosine_similarity(e1['embedding'], e2['embedding'])
                if sim > threshold:
                    relation = 'SIMULTANEOUS' if abs(e1['verse_idx'] - e2['verse_idx']) < 5 else 'BEFORE'
                    e1['relations'].append({'to': e2['id'], 'type': relation})
                    e2['relations'].append({'to': e1['id'], 'type': relation})
                    alignments[(e1['id'], e2['id'])] = sim
    return all_events, alignments