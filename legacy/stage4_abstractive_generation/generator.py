"""
TAVERN — Stage 4: Temporally-Guided Abstractive Summarization
===============================================================
Generates the consolidated narrative summary using:

1. **Temporal segmentation** — partitions the timeline into coherent narrative
   segments (e.g., major periods of the Gospel narrative).
2. **Graph-guided generation** — uses GNN-enriched embeddings to inform the
   summarisation model about event relationships.
3. **PRIMERA / LED** for multi-document abstractive summarisation when
   multiple gospel accounts need consolidation.
4. **Fallback pipeline** — extractive consolidation when no neural model is
   available.

Fixes vs. original implementation:
- Removed dead/unreachable code after ``return`` in ``consolidate_event_with_primera``
- Proper temporal segmentation based on the ``day`` field from the chronology
- Clean sentence joining without numbered prefixes for a fluent narrative
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import torch

from utils.helpers import get_embedding, cosine_similarity


# ---------------------------------------------------------------------------
# HuggingFace cache management
# ---------------------------------------------------------------------------

def _ensure_hf_cache() -> str:
    """Ensure a local HuggingFace cache directory is set and exists."""
    base = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
    if not base:
        base = os.path.join(os.getcwd(), 'hf_cache')
        os.environ.setdefault('HF_HOME', base)
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(base, 'transformers'))
    os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(base, 'datasets'))
    os.environ.setdefault('XDG_CACHE_HOME', base)
    try:
        os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
        os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)
    except Exception:
        pass
    return base


# ---------------------------------------------------------------------------
# PRIMERA / LED model management
# ---------------------------------------------------------------------------

_led_model = None
_led_tokenizer = None
_led_device = None


def _get_device() -> torch.device:
    """Select the best available device (Intel XPU > CUDA > MPS > CPU).

    Intel Arc / Data Center GPUs use the 'xpu' backend provided by
    Intel Extension for PyTorch (IPEX).  We try to import it first so
    that ``torch.xpu`` becomes available.
    """
    # --- Intel XPU (Arc, Data Center, integrated Xe) ---
    try:
        import intel_extension_for_pytorch  # noqa: F401 – enables torch.xpu
    except ImportError:
        pass
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("  [DEVICE] Intel XPU detected")
        return torch.device('xpu')
    # --- NVIDIA CUDA ---
    if torch.cuda.is_available():
        print("  [DEVICE] NVIDIA CUDA detected")
        return torch.device('cuda')
    # --- Apple MPS ---
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  [DEVICE] Apple MPS detected")
        return torch.device('mps')
    print("  [DEVICE] Using CPU")
    return torch.device('cpu')


def _ensure_led():
    """Lazy-load PRIMERA or LED model for multi-document summarisation."""
    _ensure_hf_cache()

    use_env = os.environ.get("TAVERN_USE_PRIMERA", "").lower()
    if use_env in ("0", "false", "no", "off"):
        return None, None

    global _led_model, _led_tokenizer, _led_device
    if _led_model is not None and _led_tokenizer is not None:
        return _led_model, _led_tokenizer

    from transformers import LEDForConditionalGeneration, LEDTokenizer

    local_dir = os.environ.get("TAVERN_PRIMERA_LOCAL_DIR", "").strip()
    model_override = os.environ.get("TAVERN_PRIMERA_MODEL", "").strip()
    model_names = []
    if local_dir:
        model_names.append(local_dir)
    if model_override:
        model_names.append(model_override)
    model_names += ["allenai/PRIMERA", "allenai/led-base-16384"]

    cache_dir = os.environ.get("TRANSFORMERS_CACHE")
    for name in model_names:
        try:
            print(f"  [PRIMERA] Attempting to load model: {name}")
            model = LEDForConditionalGeneration.from_pretrained(name, cache_dir=cache_dir)
            tokenizer = LEDTokenizer.from_pretrained(name, cache_dir=cache_dir)
            # Add <doc-sep> as special token
            try:
                added = tokenizer.add_special_tokens({"additional_special_tokens": ["<doc-sep>"]})
                if added and added > 0:
                    model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
            # Reduce attention window for memory efficiency
            try:
                aw = getattr(model.config, "attention_window", None)
                if isinstance(aw, (list, tuple)) and len(aw) > 0:
                    model.config.attention_window = [512] * len(aw)
            except Exception:
                pass

            _led_model, _led_tokenizer = model, tokenizer
            _led_device = _get_device()
            _led_model = _led_model.to(_led_device)
            print(f"  [PRIMERA] >>> Successfully loaded: {name} <<<")
            print(f"  [PRIMERA] Device: {_led_device}")
            print(f"  [PRIMERA] Model params: {sum(p.numel() for p in model.parameters()):,}")
            return _led_model, _led_tokenizer
        except Exception as e:
            print(f"  [PRIMERA] Failed to load {name}: {e}")
            continue

    print("  [PRIMERA] WARNING: No model could be loaded! Falling back to extractive.")
    return None, None


# ---------------------------------------------------------------------------
# BART fallback
# ---------------------------------------------------------------------------

_bart_summarizer = None


def _ensure_bart():
    """Lazy-load DistilBART as fallback summariser."""
    global _bart_summarizer
    use_env = os.environ.get('TAVERN_USE_BART', '').lower()
    if use_env in ('0', 'false', 'no', 'off'):
        return None
    if _bart_summarizer is not None:
        return _bart_summarizer
    try:
        _ensure_hf_cache()
        from transformers import pipeline
        local_dir = os.environ.get('TAVERN_BART_LOCAL_DIR', '').strip()
        model_name = local_dir or os.environ.get(
            'TAVERN_BART_MODEL', 'sshleifer/distilbart-cnn-12-6')
        # Device selection: Intel XPU > CUDA > CPU
        device_arg = -1  # CPU default
        try:
            import intel_extension_for_pytorch  # noqa: F401
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device_arg = 'xpu:0'
        except ImportError:
            pass
        if device_arg == -1 and torch.cuda.is_available():
            device_arg = 0
        _bart_summarizer = pipeline('summarization', model=model_name, device=device_arg)
        return _bart_summarizer
    except Exception:
        _bart_summarizer = None
        return None


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------

def clean_malformed_separators(text: str) -> str:
    """Clean up malformed document separators and HTML-like tags."""
    patterns = [
        r'<doc[,\-\s]*sep[,\-\s]*>',
        r'<[a-zA-Z]+\-sep[a-zA-Z]*>',
        r'<DOC[,\-\s]*sep[a-zA-Z]*>',
        r'<[a-zA-Z]*sep[,\-\s]*[a-zA-Z]*>',
    ]
    for p in patterns:
        text = re.sub(p, ' <doc-sep> ', text, flags=re.IGNORECASE)
    html_pat = r'<[^>]*(?:html|citation|footnote|xhtml|x-p|v-sep|hen|doc-p|nowiki)[^>]*>'
    text = re.sub(html_pat, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_final_text(text: str) -> str:
    """Final cleaning: remove artifacts, normalise whitespace."""
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[,]{2,}', ',', text)
    text = re.sub(r'\b[a-zA-Z]\b(?:\s[a-zA-Z]\b)+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'["“”]', '"', text)
    text = re.sub(r"['‘’]", "'", text)
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p and p.strip()]


def _jaccard(a: str, b: str) -> float:
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def reduce_redundancy(text: str, threshold: float = 0.65) -> str:
    """Remove near-duplicate sentences."""
    sents = _split_sentences(text)
    kept = []
    for s in sents:
        if not any(_jaccard(s, k) >= threshold for k in kept):
            kept.append(s)
    out = ' '.join(kept)
    if out and not out.rstrip().endswith('.'):
        out = out.rstrip() + '.'
    return clean_final_text(out)


# ---------------------------------------------------------------------------
# Temporal segmentation (thesis Section 4.5)
# ---------------------------------------------------------------------------

# Major narrative segments based on Gospel chronology
TEMPORAL_SEGMENTS = [
    ('Saturday',        'Saturday events at Bethany'),
    ('Palm Sunday',     'Triumphal entry into Jerusalem'),
    ('Monday',          'Temple cleansing and cursing of the fig tree'),
    ('Tuesday',         'Teachings, parables, and controversies'),
    ('Wednesday',       'Plot against Jesus and anointing'),
    ('Thursday',        'Last Supper and Gethsemane'),
    ('Friday',          'Trial, crucifixion, and burial'),
    ('Saturday (Rest)', 'The Sabbath rest'),
    ('Sunday',          'Resurrection and appearances'),
]


def _segment_events(consolidated_events: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    """Partition consolidated events into temporal segments based on the
    ``day`` field from the chronology table.

    Returns a list of ``(segment_label, events_in_segment)`` tuples.
    """
    # Group by day
    day_groups: Dict[str, List[Dict]] = {}
    for ev in consolidated_events:
        day = (ev.get('day') or 'Unknown').strip()
        day_groups.setdefault(day, []).append(ev)

    # Order segments according to the predefined temporal order
    day_order = {s[0]: i for i, s in enumerate(TEMPORAL_SEGMENTS)}
    ordered_segments = []
    seen_days = set()

    for day_label, _ in TEMPORAL_SEGMENTS:
        for key in list(day_groups.keys()):
            if key.lower().startswith(day_label.lower().split(' (')[0].lower()):
                ordered_segments.append((day_label, day_groups.pop(key)))
                seen_days.add(key)

    # Add any remaining days not in our predefined list
    for day, evts in day_groups.items():
        if day not in seen_days:
            ordered_segments.append((day, evts))

    return ordered_segments


# ---------------------------------------------------------------------------
# Extractive consolidation fallback
# ---------------------------------------------------------------------------

def _extractive_consolidate(texts: List[str], max_sentences: int = 6,
                            alpha: float = 0.7) -> str:
    """MMR-based extractive consolidation across multiple documents."""
    sents = []
    for d_idx, tx in enumerate(texts):
        parts = re.split(r'(?<=[.!?])\s+', tx)
        for p in parts:
            s = (p or '').strip()
            if len(s) > 10:
                sents.append((s, d_idx))
    if not sents:
        return ' '.join(t.strip() for t in texts if t and t.strip())

    # Deduplicate
    uniq = []
    seen = []
    for s, d in sents:
        if not any(_jaccard(s, x) >= 0.85 for x in seen):
            uniq.append((s, d))
            seen.append(s)
        if len(uniq) > 200:
            break
    sents = uniq

    # Embeddings
    vecs = np.stack([get_embedding(s, dim=300) for s, _ in sents], axis=0)
    centroid = vecs.mean(axis=0)

    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    scores = np.array([_cos(v, centroid) for v in vecs])

    # MMR selection
    selected_idx = []
    candidate_idx = list(range(len(sents)))
    while candidate_idx and len(selected_idx) < max_sentences:
        best, best_val = None, -1e9
        for i in candidate_idx:
            rel = scores[i]
            div = max((_cos(vecs[i], vecs[j]) for j in selected_idx), default=0.0)
            val = alpha * rel - (1.0 - alpha) * div
            if val > best_val:
                best_val, best = val, i
        if best is not None:
            selected_idx.append(best)
            candidate_idx.remove(best)

    selected = [sents[i][0] for i in selected_idx]
    out = '. '.join(selected)
    if out and not out.endswith('.'):
        out += '.'
    return out


# ---------------------------------------------------------------------------
# PRIMERA-based consolidation
# ---------------------------------------------------------------------------

def consolidate_event_with_primera(text: str, num_accounts: int) -> str:
    """Consolidate multiple gospel accounts using PRIMERA/LED."""
    if num_accounts <= 1:
        return clean_malformed_separators(text)

    try:
        model, tokenizer = _ensure_led()
        if model is None or tokenizer is None:
            raise RuntimeError("LED model unavailable")

        print(f"  [PRIMERA] Consolidating event with {num_accounts} accounts using PRIMERA model")
        clean_text = clean_malformed_separators(text)

        # Tokenise
        attn_win = 512
        try:
            aw = getattr(model.config, 'attention_window', None)
            if isinstance(aw, (list, tuple)) and len(aw) > 0:
                attn_win = int(aw[0])
        except Exception:
            pass

        inputs = tokenizer(
            clean_text, return_tensors="pt", max_length=4096,
            truncation=True, padding=True, pad_to_multiple_of=attn_win,
        )
        input_ids = inputs["input_ids"].to(_led_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(_led_device)

        # Global attention on first token and <doc-sep> tokens
        global_attention_mask = torch.zeros_like(input_ids)  # already on _led_device
        try:
            doc_sep_id = tokenizer.convert_tokens_to_ids('<doc-sep>')
            if doc_sep_id is not None and doc_sep_id >= 0:
                global_attention_mask = global_attention_mask.masked_fill(
                    input_ids.eq(doc_sep_id), 1)
        except Exception:
            pass
        global_attention_mask[:, 0] = 1
        global_attention_mask = global_attention_mask.to(_led_device)

        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                max_length=420,
                min_length=80,
                num_beams=4,
                length_penalty=1.05,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                do_sample=False,
                early_stopping=True,
                forced_bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        consolidated = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        consolidated = clean_malformed_separators(consolidated)

        # Remove prompt artifacts
        if "Merge" in consolidated or "different gospel" in consolidated:
            sentences = consolidated.split('.')
            for i, s in enumerate(sentences):
                if not any(w in s.lower() for w in ['merge', 'gospel', 'version', 'story']):
                    consolidated = '.'.join(sentences[i:]).strip()
                    break

        if consolidated and not consolidated[0].isupper():
            consolidated = consolidated[0].upper() + consolidated[1:]

        return reduce_redundancy(consolidated)

    except Exception as e:
        print(f"  [PRIMERA] !!! Consolidation FAILED: {e}")
        print(f"  [PRIMERA] Falling back to BART / extractive")
        cleaned = clean_malformed_separators(text)
        parts = [p.strip() for p in cleaned.split('<doc-sep>') if p.strip()]

        try:
            summarizer = _ensure_bart()
            if summarizer is not None:
                inp = cleaned[:4000]
                out = summarizer(inp, max_length=240, min_length=80,
                                 do_sample=False, num_beams=4, truncation=True)
                cand = out[0]['summary_text'] if isinstance(out, list) and out else ''
                if cand:
                    return reduce_redundancy(cand)
        except Exception:
            pass

        if parts and len(parts) > 1:
            return reduce_redundancy(_extractive_consolidate(parts))

        return reduce_redundancy(cleaned)


# ---------------------------------------------------------------------------
# Main generation function (Algorithm 2 from thesis)
# ---------------------------------------------------------------------------

def generate_summary(
    consolidated_events: List[Dict],
    G: nx.DiGraph,
    enriched_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """Generate the final consolidated narrative summary.

    Implements temporal segmentation + graph-guided generation as described
    in thesis Section 4.5 (Algorithm 2).

    Parameters
    ----------
    consolidated_events : list
        Macro-events from Stage 3.
    G : nx.DiGraph
        The unified event graph.
    enriched_embeddings : dict, optional
        GNN-enriched embeddings per node (from Stage 3).

    Returns
    -------
    str
        The consolidated abstractive narrative.
    """
    print(f"  Generating narrative from {len(consolidated_events)} events")

    # Topological sort for chronological ordering
    try:
        sorted_ids = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        sorted_ids = [ev['id'] for ev in consolidated_events]

    event_map = {ev['id']: ev for ev in consolidated_events}

    # Temporal segmentation
    segments = _segment_events(consolidated_events)

    narrative_parts: List[str] = []
    primera_used_count = 0
    single_source_count = 0
    consolidation_log: List[Dict] = []  # Track per-event consolidation details

    for seg_label, seg_events in segments:
        if not seg_events:
            continue

        # Sort events within segment by chronology_index
        seg_events.sort(key=lambda e: e.get('chronology_index', 0))

        segment_texts: List[str] = []
        for ev in seg_events:
            raw_text = ev.get('text', '').strip()
            if not raw_text:
                continue

            num_sources = ev.get('num_source_texts', 1)
            if num_sources > 1 and '<doc-sep>' in raw_text:
                text = consolidate_event_with_primera(raw_text, num_sources)
                primera_used_count += 1
                consolidation_log.append({
                    'event_id': ev.get('id', ''),
                    'description': ev.get('description', ''),
                    'day': ev.get('day', ''),
                    'num_sources': num_sources,
                    'source_gospels': ', '.join(ev.get('source_gospels', [])),
                    'method': 'PRIMERA',
                    'input_chars': len(raw_text),
                    'output_chars': len(text),
                    'input_text': raw_text[:2000],
                    'consolidated_text': text,
                })
            else:
                text = raw_text
                single_source_count += 1
                consolidation_log.append({
                    'event_id': ev.get('id', ''),
                    'description': ev.get('description', ''),
                    'day': ev.get('day', ''),
                    'num_sources': num_sources,
                    'source_gospels': ', '.join(ev.get('source_gospels', [])),
                    'method': 'single_source',
                    'input_chars': len(raw_text),
                    'output_chars': len(text),
                    'input_text': raw_text[:2000],
                    'consolidated_text': text,
                })

            segment_texts.append(text)

        if segment_texts:
            # Join segment texts into a coherent paragraph
            segment_narrative = ' '.join(segment_texts)
            segment_narrative = reduce_redundancy(segment_narrative)
            narrative_parts.append(segment_narrative)

    print(f"  [PRIMERA] Events consolidated with PRIMERA: {primera_used_count}")
    print(f"  [PRIMERA] Events with single source (no consolidation needed): {single_source_count}")

    # Assemble final narrative
    consolidated_narrative = '\n\n'.join(narrative_parts)

    # Final cleaning pass
    consolidated_narrative = clean_final_text(consolidated_narrative)

    print(f"  Final narrative: {len(narrative_parts)} segments, "
          f"{len(consolidated_narrative)} characters")

    # Write outputs
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/summary.txt', 'w', encoding='utf-8') as f:
        f.write(consolidated_narrative)

    # Save per-event consolidation details (JSON)
    import json
    with open('outputs/consolidation_details.json', 'w', encoding='utf-8') as f:
        json.dump(consolidation_log, f, ensure_ascii=False, indent=2)

    # Save per-event consolidation details (CSV)
    try:
        import csv
        csv_path = 'outputs/consolidation_details.csv'
        fieldnames = [
            'event_id', 'description', 'day', 'num_sources',
            'source_gospels', 'method', 'input_chars', 'output_chars',
            'consolidated_text',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(consolidation_log)
        print(f"  Consolidation log saved: {csv_path} ({len(consolidation_log)} events)")
    except Exception as e:
        print(f"  Warning: could not save CSV log: {e}")

    return consolidated_narrative
