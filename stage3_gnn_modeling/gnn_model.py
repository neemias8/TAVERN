import torch
import torch.nn as nn
import networkx as nx
from utils.helpers import create_graph

class SimpleGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        return torch.matmul(adj, self.linear(x))

def parse_verse_range(verse_ref):
    """Parse verse reference like '21:1-7' into chapter and verse range."""
    if not verse_ref or not verse_ref.strip():
        return None
    
    try:
        parts = verse_ref.strip().split(':')
        if len(parts) != 2:
            return None
            
        chapter = int(parts[0])
        verse_part = parts[1]
        
        if '-' in verse_part:
            start_verse, end_verse = verse_part.split('-')
            return chapter, int(start_verse), int(end_verse)
        else:
            verse = int(verse_part)
            return chapter, verse, verse
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
                    chapter, start_verse, end_verse = parsed
                    print(f"  Looking for {gospel} chapter {chapter}, verses {start_verse}-{end_verse}")
                    
                    # Find events in this verse range
                    for verse in range(start_verse, end_verse + 1):
                        key = (gospel, chapter, verse)
                        events_in_verse = gospel_events_by_verse.get(key, [])
                        
                        for event in events_in_verse:
                            event_text = event.get('text', '').strip()
                            if event_text and len(event_text) > 10:
                                if event_text not in event_texts:  # Avoid duplicates
                                    event_texts.append(event_text)
                                    event_ids_in_macro.append(event['id'])
                                    print(f"    Added: {event_text[:50]}...")
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