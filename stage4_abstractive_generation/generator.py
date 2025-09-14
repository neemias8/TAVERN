import os
import networkx as nx
import torch
import numpy as np
from transformers import LEDForConditionalGeneration, LEDTokenizer, pipeline
from utils.helpers import simple_word_embedding
_led_model = None
_led_tokenizer = None

def _ensure_hf_cache():
    """Ensure a local Hugging Face cache directory is set and exists.

    Honors existing envs; otherwise defaults to ./hf_cache.
    Returns the base cache directory path.
    """
    base = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
    if not base:
        base = os.path.join(os.getcwd(), 'hf_cache')
        os.environ.setdefault('HF_HOME', base)
    # Sub-caches for transformers and datasets
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(base, 'transformers'))
    os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(base, 'datasets'))
    os.environ.setdefault('XDG_CACHE_HOME', base)
    try:
        os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
        os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)
    except Exception:
        pass
    return base
def _ensure_led():
    # Ensure a local cache is configured before any downloads
    _ensure_hf_cache()
    use_env = os.environ.get("TAVERN_USE_PRIMERA","" ).lower()
    if use_env in ("0","false","no","off"):
        return None, None
    local_dir = os.environ.get("TAVERN_PRIMERA_LOCAL_DIR", "").strip()
    model_override = os.environ.get("TAVERN_PRIMERA_MODEL","" ).strip()
    model_names = []
    if local_dir:
        model_names.append(local_dir)
    if model_override:
        model_names.append(model_override)
    model_names += ["allenai/PRIMERA","allenai/led-base-16384"]
    global _led_model, _led_tokenizer
    if _led_model is not None and _led_tokenizer is not None:
        return _led_model, _led_tokenizer
    last_err = None
    for name in model_names:
        try:
            model = LEDForConditionalGeneration.from_pretrained(name, cache_dir=os.environ.get("TRANSFORMERS_CACHE", None))
            tokenizer = LEDTokenizer.from_pretrained(name, cache_dir=os.environ.get("TRANSFORMERS_CACHE", None))
            # Ensure <doc-sep> is a special token and resize embeddings
            try:
                added = tokenizer.add_special_tokens({"additional_special_tokens": ["<doc-sep>"]})
                if added and added > 0:
                    model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
            # Optionally reduce attention window to mitigate padding/memory
            try:
                aw = getattr(model.config, "attention_window", None)
                if isinstance(aw, (list, tuple)) and len(aw) > 0:
                    model.config.attention_window = [512] * len(aw)
            except Exception:
                pass
            _led_model, _led_tokenizer = model, tokenizer
            return _led_model, _led_tokenizer
        except Exception as e:
            last_err = e
            continue
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
        
        # Tokenize the clean text directly, padding aligned to attention window
        attn_win = None
        try:
            aw = getattr(model.config, 'attention_window', None)
            if isinstance(aw, (list, tuple)) and len(aw) > 0:
                attn_win = int(aw[0])
            elif isinstance(aw, int):
                attn_win = aw
        except Exception:
            attn_win = None
        if attn_win is None or attn_win <= 0:
            attn_win = 512
        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding=True,
            pad_to_multiple_of=attn_win,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        # Configure global attention: first token and <doc-sep> tokens
        global_attention_mask = torch.zeros_like(input_ids)
        try:
            doc_sep_id = tokenizer.convert_tokens_to_ids('<doc-sep>')
            if doc_sep_id is not None and doc_sep_id >= 0:
                global_attention_mask = global_attention_mask.masked_fill(input_ids.eq(doc_sep_id), 1)
        except Exception:
            pass
        global_attention_mask[:, 0] = 1
        
        # Generate consolidated version with better parameters
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                max_length=420,  # a bit shorter to reduce redundancy
                min_length=80,
                num_beams=4,
                length_penalty=1.05,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
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
        # Try BART/DistilBART summarization first (if available)
        try:
            summarizer = _ensure_bart()
            if summarizer is not None:
                inp = cleaned
                if len(inp) > 4000:
                    inp = inp[:4000]
                out = summarizer(inp, max_length=240, min_length=80, do_sample=False, num_beams=4, truncation=True)
                cand = out[0]['summary_text'] if isinstance(out, list) and out else ''
                if cand:
                    return reduce_redundancy(cand)
        except Exception as _e:
            print(f"BART fallback failed: {_e}")
        # Fallback to extractive selection
        try:
            if parts:
                summary = _extractive_consolidate(parts, max_sentences=6, alpha=0.7)
                return reduce_redundancy(summary)
        except Exception:
            pass
        if len(parts) > 1:
            return reduce_redundancy(consolidate_long_text(parts))
        return reduce_redundancy(cleaned)
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
                # Apply PRIMERA consolidation if multiple sources with explicit doc separators
                if num_sources > 1 and '<doc-sep>' in raw_text:
                    print(f"Consolidating event {i+1} with {num_sources} accounts using PRIMERA...")
                    consolidated_text = consolidate_event_with_primera(raw_text, num_sources)
                    print(f"Added event {i+1}: {consolidated_text[:100]}...")
                    # Reduce redundancy only when we truly merged multiple accounts
                    consolidated_text = reduce_redundancy(consolidated_text)
                else:
                    # Single chosen account: preserve integral text as-is (no summarization/reduction)
                    consolidated_text = raw_text.strip()
                    print(f"Added event {i+1}: {consolidated_text[:100]}...")
                
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



def _extractive_consolidate(texts, max_sentences=6, alpha=0.7):
    """Heuristic extractive consolidation across multiple documents.

    - Splits texts into sentences, embeds with a lightweight hashing embedding,
      ranks by centroid similarity with MMR diversity, and returns top sentences.
    - alpha controls balance between relevance (to centroid) and diversity.
    """
    import re
    sents = []
    for d_idx, tx in enumerate(texts):
        parts = re.split(r'[.!?]+', tx)
        for p in parts:
            s = (p or '').strip()
            if len(s) > 10:
                sents.append((s, d_idx))
    if not sents:
        return ' '.join(t.strip() for t in texts if t and t.strip())
    # Deduplicate early by Jaccard
    def _j(a,b):
        A=set(a.lower().split()); B=set(b.lower().split())
        return (len(A&B)/len(A|B)) if A and B else 0.0
    uniq=[]; seen=[]
    for s,_d in sents:
        if not any(_j(s,x)>=0.85 for x in seen):
            uniq.append((s,_d)); seen.append(s)
        if len(uniq) > 200:
            break
    sents = uniq
    # Embeddings
    vecs = []
    for s,_ in sents:
        try:
            v = simple_word_embedding(s, dim=300)
        except Exception:
            v = np.zeros(300, dtype=np.float32)
        vecs.append(v)
    vecs = np.stack(vecs, axis=0)
    # Centroid
    centroid = vecs.mean(axis=0)
    def _cos(a,b):
        na=np.linalg.norm(a); nb=np.linalg.norm(b)
        return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else 0.0
    scores = np.array([_cos(v, centroid) for v in vecs])
    # MMR selection
    selected_idx=[]; candidate_idx=list(range(len(sents)))
    while candidate_idx and len(selected_idx) < max_sentences:
        best=None; best_val=-1e9
        for i in candidate_idx:
            rel = scores[i]
            div = 0.0
            if selected_idx:
                div = max(_cos(vecs[i], vecs[j]) for j in selected_idx)
            val = alpha*rel - (1.0-alpha)*div
            if val > best_val:
                best_val=val; best=i
        selected_idx.append(best)
        candidate_idx.remove(best)
    # Preserve chronological order by original occurrence across docs
    selected = [sents[i][0] for i in selected_idx]
    # Try to sort by earliest appearance in concatenated docs
    order = { s: idx for idx,s in enumerate([x for t in texts for x in re.split(r'[.!?]+', t) if (x or '').strip()]) }
    selected.sort(key=lambda s: order.get(s, 1e9))
    out = '. '.join(selected)
    if out and not out.endswith('.'):
        out += '.'
    return out



# Lazy-load BART/DistiBART summarizer for fallback
_bart_summarizer = None

def _ensure_bart():
    global _bart_summarizer
    use_env = os.environ.get('TAVERN_USE_BART', '').lower()
    if use_env in ('0','false','no','off'):
        return None
    if _bart_summarizer is not None:
        return _bart_summarizer
    try:
        _ensure_hf_cache()
        local_dir = os.environ.get('TAVERN_BART_LOCAL_DIR', '').strip()
        model_name = local_dir or os.environ.get('TAVERN_BART_MODEL', 'sshleifer/distilbart-cnn-12-6')
        device = 0 if torch.cuda.is_available() else -1
        _bart_summarizer = pipeline('summarization', model=model_name, device=device)
        return _bart_summarizer
    except Exception:
        _bart_summarizer = None
        return None


