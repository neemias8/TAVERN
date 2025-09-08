import networkx as nx
import torch
from transformers import LEDForConditionalGeneration, LEDTokenizer

# Lazy-load model/tokenizer to avoid import-time failures and allow fallback
_led_model = None
_led_tokenizer = None

def _ensure_led():
    global _led_model, _led_tokenizer
    if _led_model is not None and _led_tokenizer is not None:
        return _led_model, _led_tokenizer
    try:
        _led_model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
        _led_tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
        return _led_model, _led_tokenizer
    except Exception:
        # Keep as None to trigger heuristic fallback
        _led_model = None
        _led_tokenizer = None
        return None, None

def clean_final_text(text):
    """Final cleaning to remove remaining artifacts and normalize text."""
    import re
    
    # Remove any remaining malformed tags
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{2,}', '.', text)  # Multiple dots
    text = re.sub(r'[,]{2,}', ',', text)  # Multiple commas
    
    # Remove broken words and artifacts
    text = re.sub(r'\b[a-zA-Z]\b(?:\s[a-zA-Z]\b)+', ' ', text)  # Single letters
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    
    # Fix common OCR/encoding issues
    text = re.sub(r'["""]', '"', text)  # Smart quotes
    text = re.sub(r"[''']", "'", text)  # Smart apostrophes
    
    # Ensure proper sentence structure
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Space after punctuation
    
    return text.strip()

def _split_sentences(text):
    import re
    parts = re.split(r'[.!?]+', text)
    return [p.strip() for p in parts if p and p.strip()]

def _jaccard(a, b):
    A = set(str(a).lower().split()); B = set(str(b).lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def reduce_redundancy(text, threshold=0.65):
    """Remove near-duplicate sentences and tighten whitespace."""
    sents = _split_sentences(text)
    kept = []
    for s in sents:
        if not any(_jaccard(s, k) >= threshold for k in kept):
            kept.append(s)
    out = '. '.join(kept)
    if out and not out.endswith('.'):
        out += '.'
    return clean_final_text(out)

def clean_malformed_separators(text):
    """Clean up malformed document separators and HTML-like tags."""
    import re
    
    # List of malformed separators to normalize
    malformed_patterns = [
        r'<doc[,\-\s]*sep[,\-\s]*>',  # Matches <doc-sep>, <doc,sep>, <doc- sep>, etc.
        r'<[a-zA-Z]+\-sep[a-zA-Z]*>',  # Matches <x-sepan>, <DO-seP>, etc.
        r'<DOC[,\-\s]*sep[a-zA-Z]*>',  # Matches <DOC-sepan>, <DOC-seP>, etc.
        r'<[a-zA-Z]*sep[,\-\s]*[a-zA-Z]*>',  # General pattern for separator variations
    ]
    
    # Replace all malformed separators with standard <doc-sep>
    for pattern in malformed_patterns:
        text = re.sub(pattern, ' <doc-sep> ', text, flags=re.IGNORECASE)
    
    # Remove HTML-like tags that aren't doc-sep
    html_pattern = r'<[^>]*(?:html|citation|footnote|xhtml|x-p|v-sep|hen|doc-p|nowiki)[^>]*>'
    text = re.sub(html_pattern, ' ', text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def consolidate_event_with_primera(text, num_accounts):
    """Consolidate multiple gospel accounts using PRIMERA multi-document summarization."""
    
    if num_accounts <= 1:
        return clean_malformed_separators(text)
    
    print(f"Consolidating event with {num_accounts} accounts using PRIMERA...")
    
    try:
        model, tokenizer = _ensure_led()
        if model is None or tokenizer is None:
            raise RuntimeError("LED model/tokenizer unavailable; using heuristic fallback")
        # Clean malformed separators first
        clean_text = clean_malformed_separators(text)
        
        # For PRIMERA, we don't need a complex prompt - just the documents
        # PRIMERA was trained to understand multi-document input with <doc-sep>
        
        # Tokenize the clean text directly
        inputs = tokenizer(clean_text, return_tensors="pt", max_length=4096, truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        # Configure global attention (at least first token)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        
        # Generate consolidated version with better parameters
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                max_length=420,  # a bit shorter to reduce redundancy
                min_length=80,
                num_beams=3,
                length_penalty=1.0,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                do_sample=False,  # Deterministic for consistency
                early_stopping=True,
                forced_bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode result and clean
        consolidated = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Remove any remaining prompt artifacts
        if "Merge" in consolidated or "different gospel" in consolidated:
            # Find the first meaningful sentence
            sentences = consolidated.split('.')
            meaningful_start = 0
            for i, sentence in enumerate(sentences):
                if not any(word in sentence.lower() for word in ['merge', 'gospel', 'version', 'story']):
                    meaningful_start = i
                    break
            consolidated = '.'.join(sentences[meaningful_start:]).strip()
        
        # Final cleanup
        consolidated = clean_malformed_separators(consolidated)
        
        # Ensure it starts with a capital letter
        if consolidated and not consolidated[0].isupper():
            consolidated = consolidated[0].upper() + consolidated[1:]
        
        return reduce_redundancy(consolidated)
        
    except Exception as e:
        print(f"PRIMERA consolidation failed: {e}")
        # Fallback: split by docs and consolidate heuristically
        cleaned = clean_malformed_separators(text)
        parts = [p.strip() for p in cleaned.split('<doc-sep>') if p.strip()]
        if len(parts) > 1:
            return reduce_redundancy(consolidate_long_text(parts))
        return reduce_redundancy(cleaned)

def consolidate_long_text(texts, similarity_threshold=0.8):
    """Fallback consolidation using sentence-level deduplication."""
    import re
    
    # Split all texts into sentences
    all_sentences = []
    for text in texts:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sentences.extend(sentences)
    
    # Remove duplicate sentences based on word overlap
    unique_sentences = []
    for sentence in all_sentences:
        is_duplicate = False
        sentence_words = set(sentence.lower().split())
        
        for existing in unique_sentences:
            existing_words = set(existing.lower().split())
            if len(sentence_words) > 0 and len(existing_words) > 0:
                overlap = len(sentence_words & existing_words) / len(sentence_words | existing_words)
                if overlap > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
    
    print(f"Long text consolidation: {len(texts)} texts -> {len(unique_sentences)} unique sentences")
    return '. '.join(unique_sentences) + '.'

def generate_summary(consolidated_events, G):
    """Generate summary by consolidating events with PRIMERA and organizing chronologically."""
    
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

    # Create the consolidated narrative by processing each event
    narrative_parts = []
    
    for i, eid in enumerate(sorted_event_ids):
        if eid in event_id_to_data:
            event_data = event_id_to_data[eid]
            raw_text = event_data.get('text', '').strip()
            num_sources = event_data.get('num_source_texts', 1)
            
            if raw_text:
                # Apply PRIMERA consolidation if multiple sources
                if num_sources > 1 and '<doc-sep>' in raw_text:
                    print(f"Consolidating event {i+1} with {num_sources} accounts using PRIMERA...")
                    consolidated_text = consolidate_event_with_primera(raw_text, num_sources)
                    print(f"Added event {i+1}: {consolidated_text[:100]}...")
                else:
                    # Single source, clean but use as-is
                    consolidated_text = clean_malformed_separators(raw_text)
                    print(f"Added event {i+1}: {consolidated_text[:100]}...")
                
                # Extra pass to reduce redundancy and clean artifacts
                consolidated_text = reduce_redundancy(consolidated_text)
                
                # Format with event number
                event_section = f"{i+1} {consolidated_text}"
                narrative_parts.append(event_section)

    # Join all parts into final consolidated narrative
    consolidated_narrative = " ".join(narrative_parts)
    
    print(f"Final consolidated narrative:")
    print(f"- Total events: {len(narrative_parts)}")
    print(f"- Total length: {len(consolidated_narrative)} characters")
    print(f"- Preview: {consolidated_narrative[:300]}...")
    
    # Write to file
    with open('outputs/summary.txt', 'w', encoding='utf-8') as f:
        f.write(consolidated_narrative)
    
    return consolidated_narrative
