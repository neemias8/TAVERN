from utils.helpers import simple_word_embedding, cosine_similarity, parse_verse_range

def align_events(annotated_docs, chronology_table, threshold=0.7):
    all_events = []
    for doc_id, annos in annotated_docs.items():
        for anno in annos:
            if anno['type'] == 'EVENT':
                anno['doc_id'] = doc_id
                # Lightweight, deterministic embedding
                anno['embedding'] = simple_word_embedding(anno['text'])
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
                parsed = parse_verse_range(ref)
                if not parsed:
                    continue
                sc, sv, ec, ev, sp, ep = parsed
                def in_range(e):
                    if e['doc_id'] != doc:
                        return False
                    ch = e.get('chapter'); vs = e.get('verse')
                    if ch is None or vs is None:
                        return False
                    if sc == ec:
                        if ch != sc:
                            return False
                        return sv <= vs <= ev
                    # Cross-chapter
                    if ch < sc or ch > ec:
                        return False
                    if ch == sc and vs < sv:
                        return False
                    if ch == ec and vs > ev:
                        return False
                    return True
                matched = [e for e in all_events if in_range(e)]
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
