import networkx as nx
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def generate_summary(consolidated_events, G):
    """Generate summary by concatenating all consolidated events in chronological order."""
    
    print(f"Generating consolidated narrative from {len(consolidated_events)} events")
    
    # Use the graph to get a topological sort of consolidated events
    try:
        sorted_event_ids = list(nx.topological_sort(G))
        print(f"Topological sort successful: {len(sorted_event_ids)} events")
    except nx.NetworkXUnfeasible:
        print("Warning: Cycle detected in graph, falling back to chronology order.")
        sorted_event_ids = [event['id'] for event in consolidated_events]

    # Create a mapping from event ID to event data for easy lookup
    event_id_to_data = {event['id']: event for event in consolidated_events}
    
    print(f"Event mapping created with {len(event_id_to_data)} events")

    # Create the consolidated narrative by concatenating all events
    narrative_parts = []
    
    for i, eid in enumerate(sorted_event_ids):
        if eid in event_id_to_data:
            event_data = event_id_to_data[eid]
            text = event_data.get('text', '').strip()
            
            if text:
                # Format with just a number reference, like a verse
                event_section = f"{i+1} {text}"
                narrative_parts.append(event_section)
                print(f"Added event {i+1}: {text[:50]}...")

    # Join all parts into final consolidated narrative with appropriate spacing
    consolidated_narrative = " ".join(narrative_parts)
    
    print(f"Final consolidated narrative:")
    print(f"- Total events: {len(narrative_parts)}")
    print(f"- Total length: {len(consolidated_narrative)} characters")
    print(f"- Preview: {consolidated_narrative[:300]}...")
    
    # Write to file
    with open('outputs/summary.txt', 'w', encoding='utf-8') as f:
        f.write(consolidated_narrative)
    
    return consolidated_narrative