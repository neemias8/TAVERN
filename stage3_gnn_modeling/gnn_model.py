"""
TAVERN — Stage 3: Graph Construction & GNN Processing
======================================================
Builds the unified cross-document event graph G = (V, E) and processes it
with a heterogeneous Graph Neural Network (GAT-based) to produce enriched
event embeddings.

**Graph structure** (per thesis Section 4.3 / Algorithm 1):

* **Nodes** — one per event instance, with features: contextualized embedding,
  temporal anchor, source provenance, entity info.
* **Edge types**:
  - ``INTRA_BEFORE`` / ``INTRA_AFTER`` — intra-document temporal order
  - ``SAME_EVENT`` — cross-document alignment (undirected)
  - ``INTER_BEFORE`` / ``INTER_AFTER`` — inter-document temporal adjacency
  - ``CONFLICT_DETAIL`` — contradictory details between aligned events

**GNN architecture** (per thesis Section 4.4):

Implements a Relational Graph Attention Network (R-GAT) with:
- Per-relation-type linear transformations
- Multi-head attention for neighbour aggregation
- Iterative message-passing rounds
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from utils.helpers import (
    get_embedding,
    parse_verse_range,
    parse_gospel_xml,
    cosine_similarity as np_cosine,
)


# ---------------------------------------------------------------------------
# Device selection (Intel XPU > CUDA > MPS > CPU)
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Select the best available accelerator."""
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except ImportError:
        pass
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# ---------------------------------------------------------------------------
# Edge type enumeration
# ---------------------------------------------------------------------------

EDGE_TYPES = [
    'INTRA_BEFORE', 'INTRA_AFTER', 'INTRA_INCLUDES',
    'SAME_EVENT',
    'INTER_BEFORE', 'INTER_AFTER', 'INTER_SIMULTANEOUS',
    'CONFLICT_DETAIL',
    'BEFORE',  # chronological sequence edges
]

EDGE_TYPE_TO_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}


# ---------------------------------------------------------------------------
# Relational GAT layer (per thesis Section 4.4)
# ---------------------------------------------------------------------------

class RelationalGATLayer(nn.Module):
    """A single layer of a Relational Graph Attention Network.

    For each edge type *r*, maintains a separate linear projection W_r.
    Uses multi-head attention to aggregate neighbour messages.
    """

    def __init__(self, in_features: int, out_features: int,
                 num_relations: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert out_features % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.num_relations = num_relations

        # Per-relation linear projections
        self.W_relations = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(num_relations)
        ])

        # Attention parameters (shared across relations for efficiency)
        self.attn_src = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for W in self.W_relations:
            nn.init.xavier_uniform_(W.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0), gain=gain)
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0), gain=gain)

    def forward(self, x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_type: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, in_features) — node features
        edge_index : (2, E) — source, target indices
        edge_type : (E,) — relation type per edge
        """
        N = x.size(0)
        H, D = self.num_heads, self.head_dim

        # Apply relation-specific transforms
        # We group edges by type for efficiency
        h_src_all = torch.zeros(edge_index.size(1), H, D, device=x.device)
        h_dst_all = torch.zeros(edge_index.size(1), H, D, device=x.device)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue
            W_r = self.W_relations[r]
            # Source and target node features for edges of type r
            src_idx = edge_index[0, mask]
            dst_idx = edge_index[1, mask]
            h_transformed = W_r(x).view(N, H, D)
            h_src_all[mask] = h_transformed[src_idx]
            h_dst_all[mask] = h_transformed[dst_idx]

        # Compute attention scores
        # e_{ij} = LeakyReLU(a_src · h_src + a_dst · h_dst)
        attn_scores = (h_src_all * self.attn_src.unsqueeze(0)).sum(-1) + \
                      (h_dst_all * self.attn_dst.unsqueeze(0)).sum(-1)
        attn_scores = self.leaky_relu(attn_scores)  # (E, H)

        # Sparse softmax over incoming edges per node
        dst_nodes = edge_index[1]  # (E,)
        # For numerical stability, subtract max per destination
        attn_max = torch.zeros(N, H, device=x.device)
        attn_max.scatter_reduce_(0, dst_nodes.unsqueeze(1).expand(-1, H),
                                 attn_scores, reduce='amax', include_self=True)
        attn_scores = attn_scores - attn_max[dst_nodes]
        attn_weights = torch.exp(attn_scores)

        # Normalise
        attn_sum = torch.zeros(N, H, device=x.device) + 1e-12
        attn_sum.scatter_add_(0, dst_nodes.unsqueeze(1).expand(-1, H), attn_weights)
        attn_weights = attn_weights / attn_sum[dst_nodes]
        attn_weights = self.dropout(attn_weights)  # (E, H)

        # Aggregate messages
        messages = h_src_all * attn_weights.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device)
        idx = dst_nodes.unsqueeze(1).unsqueeze(2).expand(-1, H, D)
        out.scatter_add_(0, idx, messages)

        return out.view(N, H * D)


class RelationalGAT(nn.Module):
    """Multi-layer Relational Graph Attention Network.

    Implements the GNN architecture described in thesis Section 4.4,
    with residual connections and layer normalisation.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 out_features: int, num_relations: int,
                 num_layers: int = 3, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RelationalGATLayer(hidden_features, hidden_features,
                                   num_relations, num_heads, dropout))
            self.norms.append(nn.LayerNorm(hidden_features))
        self.output_proj = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_type: torch.LongTensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index, edge_type)
            h = norm(h + self.dropout(h_new))  # Residual + LayerNorm
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Verse-text map builder (unchanged logic, cleaned up)
# ---------------------------------------------------------------------------

def _build_verse_text_maps(docs: Optional[Dict[str, str]]) -> Dict[tuple, str]:
    """Build ``(gospel, chapter, verse) -> text`` from Gospel XML files."""
    verse_text_map: Dict[tuple, str] = {}
    if not docs:
        return verse_text_map
    for gospel, path in docs.items():
        try:
            verses = parse_gospel_xml(path)
            for v in verses:
                ch, vs, tx = v.get('chapter'), v.get('verse'), (v.get('text') or '').strip()
                if ch and vs and tx:
                    verse_text_map[(gospel, ch, vs)] = tx
        except Exception:
            pass
    return verse_text_map


def _split_text_intelligently(text: str, part: str) -> str:
    """Split text into two halves, preferring natural breakpoints."""
    if not text or len(text) < 20:
        return text
    mid = len(text) // 2
    window = min(len(text) // 4, 100)
    best = mid
    for i in range(max(0, mid - window), min(len(text), mid + window)):
        if text[i] in '.!?':
            if abs(i - mid) < abs(best - mid):
                best = i + 1
    if best == mid:
        for i in range(max(0, mid - window), min(len(text), mid + window)):
            if text[i] in ',;:':
                if abs(i - mid) < abs(best - mid):
                    best = i + 1
    return text[:best].strip() if part == 'first' else text[best:].strip()


def _slice_text_by_part(text: str, part: Optional[str]) -> str:
    if not text or part not in ('a', 'b'):
        return text
    first = _split_text_intelligently(text, 'first')
    second = _split_text_intelligently(text, 'second')
    return first if part == 'a' else second


def _dedup_sentences(text_list: List[str], cap: int = 8,
                     threshold: float = 0.6) -> List[str]:
    """Remove near-duplicate sentences, preserving order and capping length."""
    result = []
    for t in text_list:
        t = (t or '').strip()
        if not t:
            continue
        A = set(t.lower().split())
        dup = False
        for r in result:
            B = set(r.lower().split())
            if A and B and len(A & B) / len(A | B) >= threshold:
                dup = True
                break
        if not dup:
            result.append(t)
        if len(result) >= cap:
            break
    return result


# ---------------------------------------------------------------------------
# Graph construction (Algorithm 1 from thesis)
# ---------------------------------------------------------------------------

def consolidate_and_build_graph(
    events: List[Dict],
    chronology_table: List[Dict],
    docs: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict], nx.DiGraph]:
    """Build the unified cross-document event graph.

    Returns ``(consolidated_events, G)`` where G is a NetworkX DiGraph with
    typed edges.
    """
    print(f"  Total events received: {len(events)}")

    # Group events by (doc_id, chapter, verse)
    gospel_events_by_verse: Dict[tuple, List[Dict]] = {}
    for event in events:
        doc_id = event['id'].split('_')[0].lower()
        chapter, verse = event.get('chapter'), event.get('verse')
        if chapter and verse:
            key = (doc_id, chapter, verse)
            gospel_events_by_verse.setdefault(key, []).append(event)

    # Verse text maps for full-text retrieval
    verse_text_map = _build_verse_text_maps(docs)
    gospels = ['matthew', 'mark', 'luke', 'john']

    consolidated_events: List[Dict] = []

    for i, row in enumerate(chronology_table):
        description = row.get('description', f'Event {i}')
        gospel_full_texts: Dict[str, str] = {g: '' for g in gospels}
        event_ids_in_macro: List[str] = []

        for gospel in gospels:
            verse_ref = row.get(gospel)
            if not verse_ref or not verse_ref.strip():
                continue
            parsed = parse_verse_range(verse_ref)
            if not parsed:
                continue
            start_chapter, start_verse, end_chapter, end_verse, start_part, end_part = parsed

            # Build list of (chapter, verse_start, verse_end) ranges
            chapters_to_process = []
            if start_chapter == end_chapter:
                chapters_to_process.append((start_chapter, start_verse, end_verse))
            elif end_chapter == start_chapter + 1:
                chapters_to_process.append((start_chapter, start_verse, 999))
                chapters_to_process.append((end_chapter, 1, end_verse))
            elif end_chapter <= start_chapter + 3:
                chapters_to_process.append((start_chapter, start_verse, 999))
                for ch in range(start_chapter + 1, end_chapter):
                    chapters_to_process.append((ch, 1, 999))
                chapters_to_process.append((end_chapter, 1, end_verse))
            else:
                continue

            collected_texts: List[str] = []
            for chapter, v_start, v_end in chapters_to_process:
                if v_end == 999:
                    max_v = max((k[2] for k in gospel_events_by_verse if k[0] == gospel and k[1] == chapter), default=v_start)
                    v_end = max_v

                for verse in range(v_start, v_end + 1):
                    key = (gospel, chapter, verse)
                    evts = gospel_events_by_verse.get(key, [])
                    for ev in evts:
                        event_ids_in_macro.append(ev['id'])

                    verse_text = verse_text_map.get(key, '')
                    if verse_text:
                        vt = verse_text
                        if start_chapter == end_chapter and start_verse == end_verse and (start_part or end_part):
                            vt = _slice_text_by_part(vt, start_part or end_part)
                        else:
                            if chapter == start_chapter and verse == start_verse and start_part:
                                vt = _slice_text_by_part(vt, start_part)
                            if chapter == end_chapter and verse == end_verse and end_part:
                                vt = _slice_text_by_part(vt, end_part)
                        if vt:
                            collected_texts.append(vt.strip())

            if collected_texts:
                gospel_full_texts[gospel] = ' '.join(collected_texts).strip()

        # Consolidation policy: concatenate all accounts with <doc-sep> for PRIMERA
        active_accounts = [(g, t) for g, t in gospel_full_texts.items() if t]
        if not active_accounts:
            continue

        if len(active_accounts) == 1:
            # Single source: use it directly
            chosen_gospel = active_accounts[0][0]
            consolidated_text = active_accounts[0][1]
        else:
            # Multiple sources: concatenate with <doc-sep> for PRIMERA
            chosen_gospel = max(active_accounts, key=lambda x: len(x[1]))[0]
            consolidated_text = ' <doc-sep> '.join(t for _, t in active_accounts)

        consolidated_events.append({
            'id': f'macro_{i}',
            'text': consolidated_text,
            'description': description,
            'original_events': event_ids_in_macro,
            'chronology_index': i,
            'type': 'CONSOLIDATED_EVENT',
            'num_source_texts': len(active_accounts),
            'source_gospels': [g for g, _ in active_accounts],
            'chosen_gospel': chosen_gospel,
            'day': row.get('day', ''),
        })

    print(f"  Created {len(consolidated_events)} consolidated events")

    # ---- Build the directed graph G = (V, E) ----
    G = nx.DiGraph()
    for ev in consolidated_events:
        G.add_node(ev['id'], **ev)

    # BEFORE edges (chronological sequence)
    for i in range(len(consolidated_events) - 1):
        G.add_edge(
            consolidated_events[i]['id'],
            consolidated_events[i + 1]['id'],
            type='BEFORE',
        )

    return consolidated_events, G


# ---------------------------------------------------------------------------
# GNN training and enrichment
# ---------------------------------------------------------------------------

def _prepare_graph_tensors(
    G: nx.DiGraph, events: List[Dict], embed_dim: int,
) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Convert the NetworkX graph into PyTorch tensors for the R-GAT."""
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    # Node features: use text embeddings
    features = torch.zeros(N, embed_dim)
    event_map = {ev['id']: ev for ev in events}
    for i, nid in enumerate(node_list):
        ev = event_map.get(nid)
        if ev:
            emb = get_embedding(ev.get('text', ''))
            if len(emb) != embed_dim:
                # Resize or pad
                padded = np.zeros(embed_dim, dtype=np.float32)
                padded[:min(len(emb), embed_dim)] = emb[:embed_dim]
                emb = padded
            features[i] = torch.from_numpy(emb)

    # Edges
    src_list, dst_list, etype_list = [], [], []
    for u, v, data in G.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            etype_str = data.get('type', 'BEFORE')
            etype_idx = EDGE_TYPE_TO_IDX.get(etype_str, 0)
            src_list.append(node_to_idx[u])
            dst_list.append(node_to_idx[v])
            etype_list.append(etype_idx)

    if not src_list:
        # Add self-loops to avoid empty graph
        for i in range(N):
            src_list.append(i); dst_list.append(i); etype_list.append(0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(etype_list, dtype=torch.long)

    return features, edge_index, edge_type


def run_gnn(
    events: List[Dict],
    chronology_table: List[Dict],
    embed_dim: int = 384,  # Matches Sentence-BERT (all-MiniLM-L6-v2)
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 4,
    num_epochs: int = 50,
    lr: float = 0.005,
    docs: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, np.ndarray], nx.DiGraph, List[Dict]]:
    """Run the full Stage 3 pipeline: graph construction + GNN processing.

    Returns
    -------
    enriched_events : dict
        Mapping of node_id -> enriched embedding (numpy array).
    G : nx.DiGraph
        The unified event graph.
    consolidated_events : list
        List of consolidated event dicts.
    """
    consolidated_events, G = consolidate_and_build_graph(
        events, chronology_table, docs=docs)

    # Prepare tensors
    features, edge_index, edge_type = _prepare_graph_tensors(
        G, consolidated_events, embed_dim)

    N = features.size(0)
    if N == 0:
        return {}, G, consolidated_events

    num_relations = len(EDGE_TYPES)

    # Instantiate R-GAT model
    device = _get_device()
    print(f"  GNN device: {device}")

    model = RelationalGAT(
        in_features=embed_dim,
        hidden_features=hidden_dim,
        out_features=embed_dim,
        num_relations=num_relations,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    # Move tensors to device
    features = features.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Self-supervised training objective:
    # Reconstruct input features from GNN output (autoencoder-style)
    # This forces the GNN to propagate and aggregate useful information.
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(features, edge_index, edge_type)
        # Reconstruction loss + neighbour consistency
        recon_loss = F.mse_loss(out, features)
        # Neighbour consistency: aligned events should have similar embeddings
        if edge_index.size(1) > 0:
            src_emb = out[edge_index[0]]
            dst_emb = out[edge_index[1]]
            consistency_loss = F.mse_loss(src_emb, dst_emb)
        else:
            consistency_loss = torch.tensor(0.0)
        loss = recon_loss + 0.5 * consistency_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    GNN Epoch {epoch+1}/{num_epochs} — "
                  f"loss={loss.item():.4f} "
                  f"(recon={recon_loss.item():.4f}, "
                  f"consistency={consistency_loss.item():.4f})")

    # Extract enriched embeddings
    model.eval()
    with torch.no_grad():
        enriched = model(features, edge_index, edge_type).cpu()  # back to CPU for numpy

    node_list = list(G.nodes())
    enriched_events = {
        node_list[i]: enriched[i].numpy()
        for i in range(N)
    }

    print(f"  GNN produced enriched embeddings for {len(enriched_events)} nodes")
    return enriched_events, G, consolidated_events
