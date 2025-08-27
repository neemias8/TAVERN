from utils.helpers import simple_word_embedding, cosine_similarity
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
    # Use chronology table (from XML) to align parallels
    for row in chronology_table:
        parallels = {
            'matthew': row['matthew'],
            'mark': row['mark'],
            'luke': row['luke'],
            'john': row['john']
        }
        # Filter non-null parallels
        active_docs = {k: v for k, v in parallels.items() if v}
        if len(active_docs) > 1:  # At least two Gospels cover this event
            parallel_events = []
            for doc, ref in active_docs.items():
                # Find events in annotated_docs that match ref (e.g., chapter:verse)
                # Assume verse_ref is now a string like "21:1-7" - update annotator if needed
                matched = [e for e in all_events if e['doc_id'] == doc and ref == e.get('verse_ref', '')]
                parallel_events.extend(matched)
            
            for i, e1 in enumerate(parallel_events):
                for e2 in parallel_events[i+1:]:
                    sim = cosine_similarity(e1['embedding'], e2['embedding'])
                    if sim > threshold:
                        relation = 'SIMULTANEOUS'  # Parallels in same row
                        e1['relations'].append({'to': e2['id'], 'type': relation})
                        e2['relations'].append({'to': e1['id'], 'type': relation})
                        alignments[(e1['id'], e2['id'])] = sim
    
    # Additional cosine-based alignments as before
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