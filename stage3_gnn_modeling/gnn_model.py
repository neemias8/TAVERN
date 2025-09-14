import torch
import torch.nn as nn
import networkx as nx
from utils.helpers import create_graph, parse_verse_range
from utils.helpers import parse_gospel_xml

def split_text_intelligently(text, part):
    """Split text into two halves, preferring natural breakpoints like sentences."""
    if not text or len(text) < 20:
        return text
    
    # Try to find a good midpoint by looking for sentence boundaries
    mid_point = len(text) // 2
    
    # Look for the nearest sentence boundary within a reasonable window
    search_window = min(len(text) // 4, 100)  # Don't search too far
    
    # Search backwards from midpoint for sentence endings
    best_split = mid_point
    for i in range(max(0, mid_point - search_window), min(len(text), mid_point + search_window)):
        if text[i] in '.!?':
            # Found a sentence ending, check if it's closer to midpoint
            if abs(i - mid_point) < abs(best_split - mid_point):
                best_split = i + 1
    
    # If no sentence boundary found, look for other natural breaks
    if best_split == mid_point:
        for i in range(max(0, mid_point - search_window), min(len(text), mid_point + search_window)):
            if text[i] in ',;:':
                if abs(i - mid_point) < abs(best_split - mid_point):
                    best_split = i + 1
    
    # Return the requested part
    if part == 'first':
        return text[:best_split].strip()
    else:  # part == 'second'
        return text[best_split:].strip()

class SimpleGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        return torch.matmul(adj, self.linear(x))

def _token_set(s):
    return set(str(s).lower().split())

def _overlap(a, b):
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _dedup_and_cap(text_list, cap=8, threshold=0.6):
    """Deduplicate near-duplicate sentences while preserving order and cap length.

    - text_list: list[str] of short sentences (as produced by the annotator)
    - Removes a sentence if it highly overlaps with any kept sentence.
    - Keeps at most `cap` sentences.
    """
    result = []
    for t in text_list:
        t = (t or '').strip()
        if not t:
            continue
        dup = any(_overlap(t, r) >= threshold for r in result)
        if not dup:
            result.append(t)
        if len(result) >= cap:
            break
    return result

## parse_verse_range moved to utils.helpers for reuse

def _build_verse_text_maps(docs):
    """Build a mapping (gospel, chapter, verse) -> full verse text using the XML files.

    docs: dict like {'matthew': path, 'mark': path, 'luke': path, 'john': path}
    returns: dict[(str,int,int)] -> str
    """
    verse_text_map = {}
    if not docs:
        return verse_text_map
    for gospel, path in docs.items():
        try:
            verses = parse_gospel_xml(path)
            for v in verses:
                ch = v.get('chapter'); vs = v.get('verse'); tx = (v.get('text') or '').strip()
                if ch and vs and tx:
                    verse_text_map[(gospel, ch, vs)] = tx
        except Exception:
            # If parsing fails for any gospel, leave it empty; we fallback to annotations
            pass
    return verse_text_map

def _slice_text_by_part(text, part):
    """Return first ('a') or second ('b') half of a text, preferring sentence boundaries.

    This is a heuristic to approximate verse parts when references include a/b suffix.
    """
    if not text or part not in ('a', 'b'):
        return text
    first = split_text_intelligently(text, 'first')
    second = split_text_intelligently(text, 'second')
    return first if part == 'a' else second

def consolidate_and_build_graph(events, chronology_table, docs=None):
    """Consolidate events from multiple gospels into enriched macro-events using chronology table."""
    
    print(f"Total events received: {len(events)}")
    
    # Group events by document and create a lookup by chapter:verse
    gospel_events_by_verse = {}
    for event in events:
        doc_id = event['id'].split('_')[0].lower()
        
        # Use the real chapter and verse information from the event
        chapter = event.get('chapter')
        verse = event.get('verse')
        
        if chapter and verse:
            key = (doc_id, chapter, verse)
            if key not in gospel_events_by_verse:
                gospel_events_by_verse[key] = []
            gospel_events_by_verse[key].append(event)
    
    print(f"Created verse lookup for {len(gospel_events_by_verse)} verse keys")
    
    # Process chronology table to create consolidated events
    consolidated_events = []
    gospels = ['matthew', 'mark', 'luke', 'john']
    
    print(f"Chronology table has {len(chronology_table)} entries")
    if chronology_table:
        print(f"Sample chronology entry: {chronology_table[0]}")
    
    # Pre-build verse text maps if doc paths are provided
    verse_text_map = _build_verse_text_maps(docs)

    for i, row in enumerate(chronology_table):  # Process ALL events
        description = row.get('description', f'Event {i}')
        print(f"\nProcessing chronology row {i}: {description}")
        
        # Collect full verse text per gospel (integral text per account)
        gospel_full_texts = {g: '' for g in gospels}
        event_ids_in_macro = []
        
        for gospel in gospels:
            verse_ref = row.get(gospel)
            if verse_ref and verse_ref.strip():
                parsed = parse_verse_range(verse_ref)
                if parsed:
                    start_chapter, start_verse, end_chapter, end_verse, start_part, end_part = parsed
                    print(f"  Looking for {gospel} chapter {start_chapter}, verses {start_verse}-{end_verse}")
                    if start_chapter != end_chapter:
                        print(f"    Cross-chapter reference: {start_chapter}:{start_verse} to {end_chapter}:{end_verse}")
                    if start_part or end_part:
                        print(f"    With parts: start='{start_part}', end='{end_part}'")
                    
                    # Handle cross-chapter references
                    chapters_to_process = []
                    if start_chapter == end_chapter:
                        # Single chapter reference
                        chapters_to_process.append((start_chapter, start_verse, end_verse))
                    else:
                        # Cross-chapter reference - we need to process multiple chapters
                        # For now, let's handle the common case where we go from one chapter to the next
                        if end_chapter == start_chapter + 1:
                            # Process remaining verses in start chapter
                            chapters_to_process.append((start_chapter, start_verse, 999))  # Use 999 as "end of chapter"
                            # Process verses from beginning of end chapter
                            chapters_to_process.append((end_chapter, 1, end_verse))
                        else:
                            print(f"    Skipping complex cross-chapter reference spanning more than 2 chapters")
                            continue
                    
                    # Find events in the specified chapter(s) and verse range(s)
                    for chapter, verse_start, verse_end in chapters_to_process:
                        # Find the actual last verse in the chapter if we used 999
                        if verse_end == 999:
                            # Find the highest verse number in this chapter for this gospel
                            max_verse = 0
                            for key in gospel_events_by_verse.keys():
                                if key[0] == gospel and key[1] == chapter:
                                    max_verse = max(max_verse, key[2])
                            verse_end = max_verse if max_verse > 0 else verse_start
                        
                        # Build integral text for this gospel across the verses
                        collected_texts = []
                        for verse in range(verse_start, verse_end + 1):
                            key = (gospel, chapter, verse)
                            # Gather event IDs for traceability
                            events_in_verse = gospel_events_by_verse.get(key, [])
                            if events_in_verse:
                                # Decide if we need to filter this verse's events by a/b for original_events only
                                def filter_by_part(evts, part):
                                    if not evts or not part:
                                        return evts
                                    n = len(evts)
                                    if n == 1:
                                        return evts  # nothing to split
                                    mid = (n + 1) // 2
                                    return (evts[:mid] if part == 'a' else evts[mid:])

                                selected = events_in_verse
                                # Single-verse case with suffix
                                if (start_chapter == end_chapter and start_verse == end_verse and
                                    chapter == start_chapter and verse == start_verse and (start_part or end_part)):
                                    part = start_part or end_part
                                    selected = filter_by_part(events_in_verse, part)
                                else:
                                    # Range boundaries with suffixes
                                    if chapter == start_chapter and verse == start_verse and start_part:
                                        selected = filter_by_part(events_in_verse, start_part)
                                    if chapter == end_chapter and verse == end_verse and end_part:
                                        selected = filter_by_part(selected, end_part)

                                for event in selected:
                                    event_ids_in_macro.append(event['id'])

                            # Collect full verse text from XML map if available
                            verse_text = verse_text_map.get(key, '')
                            if verse_text:
                                # Apply a/b slicing on boundaries if necessary
                                if (start_chapter == end_chapter and start_verse == end_verse and
                                    chapter == start_chapter and verse == start_verse and (start_part or end_part)):
                                    part = start_part or end_part
                                    verse_text_use = _slice_text_by_part(verse_text, part)
                                else:
                                    verse_text_use = verse_text
                                    if chapter == start_chapter and verse == start_verse and start_part:
                                        verse_text_use = _slice_text_by_part(verse_text_use, start_part)
                                    if chapter == end_chapter and verse == end_verse and end_part:
                                        verse_text_use = _slice_text_by_part(verse_text_use, end_part)
                                if verse_text_use:
                                    collected_texts.append(verse_text_use.strip())

                        # Join collected verse texts for this gospel account
                        if collected_texts:
                            # Ensure a clean single-space separation
                            gospel_full_texts[gospel] = ' '.join(collected_texts).strip()
                else:
                    print(f"  Could not parse verse reference: {verse_ref}")
        
        # Create consolidated event according to user's policy:
        # - If only one gospel reports the event, use its full text as-is (integral)
        # - If multiple report, choose only the longest account
        active_accounts = [(g, t) for g, t in gospel_full_texts.items() if t]
        if active_accounts:
            if len(active_accounts) == 1:
                chosen_gospel, chosen_text = active_accounts[0]
                consolidated_text = chosen_text
                chosen = chosen_gospel
            else:
                # Pick the longest by character length
                chosen_gospel, chosen_text = max(active_accounts, key=lambda x: len(x[1]))
                consolidated_text = chosen_text
                chosen = chosen_gospel

            consolidated_event = {
                'id': f'macro_{i}',
                'text': consolidated_text,
                'description': description,
                'original_events': event_ids_in_macro,
                'chronology_index': i,
                'type': 'CONSOLIDATED_EVENT',
                'num_source_texts': 1,  # we include only one account per instruction
                'source_gospels': [g for g, t in active_accounts],
                'chosen_gospel': chosen
            }
            consolidated_events.append(consolidated_event)
            print(f"  Created macro_{i}: {description} (chosen account: {chosen})")
        else:
            print(f"  No events found for macro_{i}: {description}")
    
    print(f"\nFinal: Created {len(consolidated_events)} consolidated events")
    
    # Build directed graph
    G = nx.DiGraph()
    for event in consolidated_events:
        G.add_node(event['id'], **event)
    
    # Add BEFORE edges
    for i in range(len(consolidated_events) - 1):
        current_id = consolidated_events[i]['id']
        next_id = consolidated_events[i + 1]['id']
        G.add_edge(current_id, next_id, type='BEFORE')
    
    return consolidated_events, G

def run_gnn(events, chronology_table, embed_dim=768, docs=None):  # Match BART embed size
    consolidated_events, G = consolidate_and_build_graph(events, chronology_table, docs=docs)
    
    # Create node features for consolidated events
    node_features = torch.randn(len(G.nodes), embed_dim)
    adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.float)
    
    model = SimpleGCN(embed_dim, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(10):  # Dummy training
        out = model(node_features, adj)
        loss = torch.mean(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    enriched_events = {node: out[i].detach().numpy() for i, node in enumerate(G.nodes)}
    return enriched_events, G, consolidated_events
