"""
TAVERN — Stage 3: Relational Graph Attention Network (R-GAT)
=============================================================
Builds a heterogeneous temporal knowledge graph from Stage-2 alignments
and encodes it with a **Relational Graph Attention Network** (R-GAT).

Architecture
------------
* Per-relation linear transforms project neighbour features into a shared
  space before multi-head dot-product attention.
* Residual connections + LayerNorm after each message-passing layer.
* ``encode_graph()`` returns a dict mapping node-id → embedding vector.

Falls back gracefully to a simple mean-aggregation GCN when PyTorch is
unavailable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyTorch imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from utils.helpers import get_embedding, cosine_similarity

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

RELATION_TYPES = [
    'BEFORE', 'AFTER', 'SIMULTANEOUS', 'IS_INCLUDED', 'INCLUDES',
    'BEGINS', 'ENDS', 'BEGUN_BY', 'ENDED_BY',
    'DURING', 'DURING_INV', 'OVERLAP', 'OVERLAPPED_BY',
    'IDENTITY', 'IAFTER', 'IBEFORE',
    'EVIDENTIAL', 'MODAL', 'COUNTER_FACTIVE',  # SLINKs
    'ALIGNED',       # cross-document alignment edges
    'SEQUENTIAL',    # within-document narrative order
]

REL2IDX: Dict[str, int] = {r: i for i, r in enumerate(RELATION_TYPES)}
NUM_RELATIONS = len(RELATION_TYPES)


def build_temporal_graph(
        annotated_docs: Dict[str, List[Dict[str, Any]]],
        alignments: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Tuple[int, int, str]]]:
    """Build a heterogeneous temporal knowledge graph.

    Returns
    -------
    nodes : list of dict
        Each node is an event/TIMEX3 annotation dict with added ``'node_idx'``.
    edges : list of (src_idx, dst_idx, relation_type)
    """
    # ---- Collect all nodes ----
    nodes: List[Dict] = []
    node_id_map: Dict[str, int] = {}  # annotation id → node index

    for doc_id, annotations in annotated_docs.items():
        for ann in annotations:
            if ann['type'] in ('EVENT', 'TIMEX3'):
                if ann['id'] not in node_id_map:
                    idx = len(nodes)
                    node_id_map[ann['id']] = idx
                    nodes.append({**ann, 'node_idx': idx, 'doc_id': doc_id})

    # ---- Intra-document edges from TLINKs and SLINKs ----
    edges: List[Tuple[int, int, str]] = []

    for doc_id, annotations in annotated_docs.items():
        for ann in annotations:
            if ann['type'] == 'TLINK':
                src_id = ann.get('eventID', '')
                dst_id = ann.get('relatedToEvent') or ann.get('relatedToTime', '')
                rel = ann.get('relType', 'SEQUENTIAL')
                if src_id in node_id_map and dst_id in node_id_map:
                    edges.append((
                        node_id_map[src_id],
                        node_id_map[dst_id],
                        rel,
                    ))
            elif ann['type'] == 'SLINK':
                src_id = ann.get('eventID', '')
                dst_id = ann.get('subordinatedEvent', '')
                rel = ann.get('relType', 'MODAL')
                if src_id in node_id_map and dst_id in node_id_map:
                    edges.append((
                        node_id_map[src_id],
                        node_id_map[dst_id],
                        rel,
                    ))

    # ---- Cross-document alignment edges ----
    for alignment in alignments:
        ev_list = [e for e in alignment.get('events', []) if 'id' in e]
        for i in range(len(ev_list)):
            for j in range(i + 1, len(ev_list)):
                id_i = ev_list[i]['id']
                id_j = ev_list[j]['id']
                if id_i in node_id_map and id_j in node_id_map:
                    edges.append((node_id_map[id_i], node_id_map[id_j], 'ALIGNED'))
                    edges.append((node_id_map[id_j], node_id_map[id_i], 'ALIGNED'))

    return nodes, edges


# ---------------------------------------------------------------------------
# Numpy-based R-GAT (no PyTorch)
# ---------------------------------------------------------------------------

class _RelGATLayerNumpy:
    """Single R-GAT layer implemented in pure NumPy.

    Uses per-relation linear transforms + scaled dot-product attention,
    followed by residual + LayerNorm.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 num_relations: int = NUM_RELATIONS, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.num_relations = num_relations

        scale = math.sqrt(2.0 / (in_dim + out_dim))
        # Per-relation weight matrices [R, in_dim, out_dim]
        self.W_rel = rng.standard_normal(
            (num_relations, in_dim, out_dim)).astype(np.float32) * scale
        # Attention vectors per head [num_heads, 2*head_dim]
        self.attn_vec = rng.standard_normal(
            (num_heads, 2 * self.head_dim)).astype(np.float32) * 0.01
        # Residual projection (if dims differ)
        self.W_res = (
            rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
            if in_dim != out_dim else None
        )
        # LayerNorm parameters
        self.gamma = np.ones(out_dim, dtype=np.float32)
        self.beta = np.zeros(out_dim, dtype=np.float32)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True) + 1e-6
        return self.gamma * (x - mu) / sigma + self.beta

    def forward(self, h: np.ndarray,
                adj: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """h : (N, in_dim); adj : rel_type → [(src, dst), ...]."""
        N, in_dim = h.shape
        out = np.zeros((N, self.out_dim), dtype=np.float32)
        count = np.zeros(N, dtype=np.float32)

        for rel_type, edge_list in adj.items():
            r_idx = REL2IDX.get(rel_type, 0)
            W = self.W_rel[r_idx]  # (in_dim, out_dim)

            for src, dst in edge_list:
                h_src = h[src] @ W  # (out_dim,)
                h_dst = h[dst] @ W  # (out_dim,)

                # Multi-head attention
                attn_sum = np.zeros(self.out_dim, dtype=np.float32)
                for k in range(self.num_heads):
                    s = self.head_dim * k
                    e = s + self.head_dim
                    concat = np.concatenate([h_src[s:e], h_dst[s:e]])
                    score = float(np.dot(self.attn_vec[k], concat))
                    score = max(0.0, score)  # LeakyReLU(x, 0)
                    attn_sum[s:e] += score * h_src[s:e]

                out[dst] += attn_sum
                count[dst] += 1

        # Mean aggregation
        mask = count > 0
        out[mask] /= count[mask, None]

        # Residual
        if self.W_res is not None:
            res = h @ self.W_res
        else:
            res = h
        out = out + res

        # LayerNorm + ReLU
        out = self._layer_norm(out)
        out = np.maximum(0, out)  # ReLU
        return out


class RelGATNumpy:
    """Two-layer R-GAT (NumPy backend)."""

    def __init__(self, in_dim: int = 300, hidden_dim: int = 256,
                 out_dim: int = 128, num_heads: int = 4):
        self.layer1 = _RelGATLayerNumpy(in_dim, hidden_dim, num_heads)
        self.layer2 = _RelGATLayerNumpy(hidden_dim, out_dim, num_heads)

    def forward(self, h: np.ndarray,
                adj: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        h = self.layer1.forward(h, adj)
        h = self.layer2.forward(h, adj)
        return h


# ---------------------------------------------------------------------------
# PyTorch R-GAT (when available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _RelGATLayerTorch(nn.Module):
        """Single R-GAT layer (PyTorch)."""

        def __init__(self, in_dim: int, out_dim: int,
                     num_heads: int = 4, num_relations: int = NUM_RELATIONS):
            super().__init__()
            assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
            self.num_heads = num_heads
            self.head_dim = out_dim // num_heads
            self.out_dim = out_dim

            # Per-relation weight matrices
            self.W_rel = nn.Parameter(
                torch.empty(num_relations, in_dim, out_dim))
            nn.init.kaiming_uniform_(self.W_rel, a=math.sqrt(5))

            # Per-head attention vectors
            self.attn = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
            nn.init.xavier_uniform_(self.attn.unsqueeze(0))

            self.layer_norm = nn.LayerNorm(out_dim)
            self.res_proj = (
                nn.Linear(in_dim, out_dim, bias=False)
                if in_dim != out_dim else nn.Identity()
            )

        def forward(self, h: 'torch.Tensor',
                    adj: Dict[str, List[Tuple[int, int]]]) -> 'torch.Tensor':
            N = h.shape[0]
            out = torch.zeros(N, self.out_dim, device=h.device)
            count = torch.zeros(N, device=h.device)

            for rel_type, edge_list in adj.items():
                if not edge_list:
                    continue
                r_idx = REL2IDX.get(rel_type, 0)
                W = self.W_rel[r_idx]  # (in_dim, out_dim)

                srcs = [e[0] for e in edge_list]
                dsts = [e[1] for e in edge_list]

                h_src = h[srcs] @ W  # (E, out_dim)
                h_dst = h[dsts] @ W  # (E, out_dim)

                h_src_h = h_src.view(-1, self.num_heads, self.head_dim)
                h_dst_h = h_dst.view(-1, self.num_heads, self.head_dim)
                concat = torch.cat([h_src_h, h_dst_h], dim=-1)  # (E, H, 2*d)
                # attn shape: (H, 2*d)
                scores = (concat * self.attn.unsqueeze(0)).sum(-1)  # (E, H)
                scores = F.leaky_relu(scores, 0.2)

                # Expand scores to full head_dim
                scores_exp = scores.unsqueeze(-1).expand_as(h_src_h)  # (E,H,d)
                weighted = (scores_exp * h_src_h).view(-1, self.out_dim)  # (E, out)

                # Scatter into out
                dst_t = torch.tensor(dsts, dtype=torch.long, device=h.device)
                out.index_add_(0, dst_t, weighted)
                count.index_add_(0, dst_t, torch.ones(len(dsts), device=h.device))

            mask = count > 0
            out[mask] /= count[mask].unsqueeze(1)

            # Residual + LayerNorm + ReLU
            out = self.layer_norm(out + self.res_proj(h))
            return F.relu(out)

    class RelGATTorch(nn.Module):
        """Two-layer R-GAT (PyTorch backend)."""

        def __init__(self, in_dim: int = 300, hidden_dim: int = 256,
                     out_dim: int = 128, num_heads: int = 4):
            super().__init__()
            self.layer1 = _RelGATLayerTorch(in_dim, hidden_dim, num_heads)
            self.layer2 = _RelGATLayerTorch(hidden_dim, out_dim, num_heads)

        def forward(self, h: 'torch.Tensor',
                    adj: Dict[str, List[Tuple[int, int]]]) -> 'torch.Tensor':
            h = self.layer1(h, adj)
            h = self.layer2(h, adj)
            return h


# ---------------------------------------------------------------------------
# Public API: encode_graph
# ---------------------------------------------------------------------------

def encode_graph(
        nodes: List[Dict],
        edges: List[Tuple[int, int, str]],
        in_dim: int = 300,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_heads: int = 4,
) -> Dict[str, np.ndarray]:
    """Encode a temporal knowledge graph with a Relational GAT.

    Parameters
    ----------
    nodes : list of dict
        Node dicts (must have ``'id'`` and ``'text'`` keys).
    edges : list of (src_idx, dst_idx, rel_type)
    in_dim, hidden_dim, out_dim : int
        Layer dimensions.
    num_heads : int
        Number of attention heads.

    Returns
    -------
    dict
        Mapping ``node_id → embedding_array`` (shape ``(out_dim,)``).
    """
    if not nodes:
        return {}

    N = len(nodes)

    # ---- Build initial feature matrix ----
    h0 = np.zeros((N, in_dim), dtype=np.float32)
    for node in nodes:
        idx = node['node_idx']
        emb = get_embedding(node.get('text', ''), dim=in_dim)
        if emb.shape[0] != in_dim:
            tmp = np.zeros(in_dim, dtype=np.float32)
            tmp[:min(in_dim, emb.shape[0])] = emb[:in_dim]
            emb = tmp
        h0[idx] = emb

    # ---- Build adjacency dict ----
    adj: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for src, dst, rel in edges:
        adj[rel].append((src, dst))

    # ---- Forward pass ----
    if _TORCH_AVAILABLE:
        model = RelGATTorch(in_dim=in_dim, hidden_dim=hidden_dim,
                            out_dim=out_dim, num_heads=num_heads)
        model.eval()
        with torch.no_grad():
            h_tensor = torch.from_numpy(h0)
            out = model(h_tensor, dict(adj)).numpy()
    else:
        model_np = RelGATNumpy(in_dim=in_dim, hidden_dim=hidden_dim,
                               out_dim=out_dim, num_heads=num_heads)
        out = model_np.forward(h0, dict(adj))

    # ---- Return id → embedding mapping ----
    return {node['id']: out[node['node_idx']] for node in nodes}
