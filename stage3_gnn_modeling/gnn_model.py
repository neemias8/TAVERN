import torch
import torch.nn as nn
import networkx as nx
from utils.helpers import create_graph

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

def parse_verse_range(verse_ref):
    """Parse verse reference like '21:1-7' or '15:18-16:4' into chapter and verse range.
    Handles letter suffixes: 'a' = first half of verse, 'b' = second half of verse.
    Handles cross-chapter references like '15:18-16:4'.
    Returns (start_chapter, start_verse, end_chapter, end_verse, start_part, end_part) where parts are 'a', 'b', or None.
    """
    if not verse_ref or not verse_ref.strip():
        return None
    
    try:
        import re
        verse_ref = verse_ref.strip()
        
        # Check for cross-chapter references like "15:18-16:4"
        cross_chapter_match = re.match(r'(\d+):(\d+)([ab]?)-(\d+):(\d+)([ab]?)$', verse_ref)
        if cross_chapter_match:
            start_chapter = int(cross_chapter_match.group(1))
            start_verse = int(cross_chapter_match.group(2))
            start_suffix = cross_chapter_match.group(3) if cross_chapter_match.group(3) else None
            end_chapter = int(cross_chapter_match.group(4))
            end_verse = int(cross_chapter_match.group(5))
            end_suffix = cross_chapter_match.group(6) if cross_chapter_match.group(6) else None
            
            return start_chapter, start_verse, end_chapter, end_verse, start_suffix, end_suffix
        
        # Handle single chapter references
        parts = verse_ref.split(':')
        if len(parts) != 2:
            return None
            
        chapter = int(parts[0])
        verse_part = parts[1]
        
        def parse_verse_with_suffix(verse_str):
            """Parse verse number with optional 'a' or 'b' suffix."""
            verse_str = verse_str.strip()
            # Check for letter suffix
            match = re.match(r'(\d+)([a-zA-Z]*)$', verse_str)
            if match:
                verse_num = int(match.group(1))
                suffix = match.group(2).lower() if match.group(2) else None
                # Only keep 'a' or 'b' suffixes, ignore others
                if suffix and suffix not in ['a', 'b']:
                    suffix = None
                return verse_num, suffix
            return None, None
        
        if '-' in verse_part:
            start_verse_str, end_verse_str = verse_part.split('-', 1)
            start_verse, start_part = parse_verse_with_suffix(start_verse_str)
            end_verse, end_part = parse_verse_with_suffix(end_verse_str)
            
            if start_verse is None or end_verse is None:
                return None
            return chapter, start_verse, chapter, end_verse, start_part, end_part
        else:
            verse, part = parse_verse_with_suffix(verse_part)
            if verse is None:
                return None
            return chapter, verse, chapter, verse, part, part
    except (ValueError, IndexError):
        return None

def consolidate_and_build_graph(events, chronology_table):
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
    
    for i, row in enumerate(chronology_table):  # Process ALL events
        description = row.get('description', f'Event {i}')
        print(f"\nProcessing chronology row {i}: {description}")
        
        event_texts = []
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
                        
                        for verse in range(verse_start, verse_end + 1):
                            key = (gospel, chapter, verse)
                            events_in_verse = gospel_events_by_verse.get(key, [])
                            
                            for event in events_in_verse:
                                event_text = event.get('text', '').strip()
                                if event_text and len(event_text) > 10:
                                    
                                    # Apply part filtering for 'a' and 'b' suffixes
                                    filtered_text = event_text
                                    should_filter = False
                                    
                                    # Only apply filtering at the beginning or end of the range
                                    if (chapter == start_chapter and verse == start_verse and 
                                        verse == end_verse and start_chapter == end_chapter):
                                        # Single verse with suffix
                                        if start_part == 'a':
                                            filtered_text = split_text_intelligently(event_text, 'first')
                                            should_filter = True
                                        elif start_part == 'b':
                                            filtered_text = split_text_intelligently(event_text, 'second')
                                            should_filter = True
                                    elif chapter == start_chapter and verse == start_verse and start_part:
                                        # Starting verse of a range with suffix
                                        if start_part == 'a':
                                            filtered_text = split_text_intelligently(event_text, 'first')
                                            should_filter = True
                                        elif start_part == 'b':
                                            filtered_text = split_text_intelligently(event_text, 'second')
                                            should_filter = True
                                    elif chapter == end_chapter and verse == end_verse and end_part:
                                        # Ending verse of a range with suffix
                                        if end_part == 'a':
                                            filtered_text = split_text_intelligently(event_text, 'first')
                                            should_filter = True
                                        elif end_part == 'b':
                                            filtered_text = split_text_intelligently(event_text, 'second')
                                            should_filter = True
                                    
                                    if should_filter:
                                        print(f"      Filtered to {len(filtered_text)} chars (was {len(event_text)})")
                                    
                                    if filtered_text and len(filtered_text) > 10:
                                        if filtered_text not in event_texts:  # Avoid duplicates
                                            event_texts.append(filtered_text)
                                        event_ids_in_macro.append(event['id'])
                                        print(f"    Added: {filtered_text[:50]}...")
                else:
                    print(f"  Could not parse verse reference: {verse_ref}")
        
        # Create consolidated event if we have content
        if event_texts:
            consolidated_text = ' '.join(event_texts)
            
            consolidated_event = {
                'id': f'macro_{i}',
                'text': consolidated_text,
                'description': description,
                'original_events': event_ids_in_macro,
                'chronology_index': i,
                'type': 'CONSOLIDATED_EVENT'
            }
            consolidated_events.append(consolidated_event)
            print(f"  Created macro_{i}: {description} ({len(event_texts)} texts)")
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

def run_gnn(events, chronology_table, embed_dim=768):  # Match BART embed size
    consolidated_events, G = consolidate_and_build_graph(events, chronology_table)
    
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