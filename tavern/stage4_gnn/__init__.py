"""
Stage 4 - relational graph attention (thesis Sections 7.3-7.4).

Per-relation transformations from relational graph convolution combined with the
learned neighbour weighting of graph attention, with the edge's confidence and
its asserted/derived flag entering the attention computation. That term is the
mechanism by which the annotation's own uncertainty reaches the learned
component.

Training is self-supervised, with a reconstruction objective and a relational
consistency objective. A caution the thesis states and this module honours: with
a few hundred nodes and no task supervision the network is not learning a
generalisable model of narrative structure, it is computing a structure-aware
smoothing of features the annotation already provides. The comparison that means
something is therefore against the same features UNPROPAGATED, which
`aggregate_without_propagation` supplies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import TavernConfig
from ..stage3_anchoring_alignment.graph import EDGE_TYPES, EventGraph

__all__ = ["train_and_score", "aggregate_without_propagation", "GNNResult"]


@dataclass
class GNNResult:
    node_scores: Dict[str, float] = field(default_factory=dict)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    losses: List[float] = field(default_factory=list)
    propagated: bool = True
    seed: int = 0
    epochs: int = 0

    def cluster_scores(self) -> Dict[str, float]:
        return self.node_scores


# ---------------------------------------------------------------------------
def aggregate_without_propagation(graph: EventGraph) -> GNNResult:
    """The ablation of Section 9.6: the same features, no message passing.

    A node's score is a learned-free linear read-out of its own feature vector,
    normalised within its cluster. This is the configuration against which
    message passing has to prove itself.
    """
    res = GNNResult(propagated=False)
    feats = graph.node_features
    if not feats:
        return res
    dim = len(next(iter(feats.values())))
    # centre and scale each dimension, then score by the projection onto the
    # dimensions that carry evidential strength
    cols = [[f[i] for f in feats.values()] for i in range(dim)]
    mean = [sum(c) / len(c) for c in cols]
    std = [max(1e-6, (sum((x - m) ** 2 for x in c) / len(c)) ** 0.5)
           for c, m in zip(cols, mean)]
    for nid, f in feats.items():
        z = [(x - m) / s for x, m, s in zip(f, mean, std)]
        res.node_scores[nid] = sum(z) / dim
        res.embeddings[nid] = z
    return res


# ---------------------------------------------------------------------------
def train_and_score(graph: EventGraph, cfg: TavernConfig,
                    seed: int = 13) -> GNNResult:
    try:
        import torch
    except ImportError:                       # pragma: no cover
        return aggregate_without_propagation(graph)
    return _train_torch(graph, cfg, seed)


def _train_torch(graph: EventGraph, cfg: TavernConfig, seed: int) -> GNNResult:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)

    nodes = list(graph.g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    X = torch.tensor([graph.node_features[n] for n in nodes],
                     dtype=torch.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    n, d_in = X.shape

    rel_index: Dict[str, List[Tuple[int, int, float, float]]] = {
        r: [] for r in EDGE_TYPES}
    for u, v, data in graph.g.edges(data=True):
        r = data["type"]
        rel_index.setdefault(r, []).append(
            (idx[u], idx[v], float(data.get("weight", 0.5)),
             float(data.get("asserted", 0.0))))

    same = [(idx[u], idx[v]) for u, v, dd in graph.g.edges(data=True)
            if dd["type"] == "SAME_EVENT"]
    conflict = [(idx[u], idx[v]) for u, v, dd in graph.g.edges(data=True)
                if dd["type"] == "CONFLICT"]

    class RGATLayer(nn.Module):
        def __init__(self, d_in: int, d_out: int, heads: int):
            super().__init__()
            self.heads = heads
            self.d_head = max(1, d_out // heads)
            self.d_out = self.d_head * heads
            self.self_w = nn.Linear(d_in, self.d_out, bias=False)
            self.rel_w = nn.ModuleDict({
                r: nn.Linear(d_in, self.d_out, bias=False) for r in rel_index})
            self.attn = nn.ParameterDict({
                r: nn.Parameter(torch.randn(heads, 2 * self.d_head + 2) * 0.1)
                for r in rel_index})
            self.norm = nn.LayerNorm(self.d_out)

        def forward(self, h):
            out = self.self_w(h)
            logits: Dict[str, torch.Tensor] = {}
            msgs: Dict[str, torch.Tensor] = {}
            targets: Dict[str, torch.Tensor] = {}
            for r, edges in rel_index.items():
                if not edges:
                    continue
                src = torch.tensor([e[0] for e in edges], dtype=torch.long)
                dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
                cij = torch.tensor([[e[2], e[3]] for e in edges],
                                   dtype=torch.float32)
                wr = self.rel_w[r](h).view(-1, self.heads, self.d_head)
                hi = wr[dst]
                hj = wr[src]
                c = cij.unsqueeze(1).expand(-1, self.heads, -1)
                feat = torch.cat([hi, hj, c], dim=-1)
                e = F.leaky_relu((feat * self.attn[r]).sum(-1), 0.2)
                logits[r] = e
                msgs[r] = hj
                targets[r] = dst
            if logits:
                # softmax over all relations and neighbours of a node jointly,
                # as Equation eq:attn specifies
                cat_e = torch.cat([logits[r] for r in logits], dim=0)
                cat_t = torch.cat([targets[r] for r in logits], dim=0)
                cat_m = torch.cat([msgs[r] for r in logits], dim=0)
                mx = torch.full((n, self.heads), -1e9)
                mx = mx.index_reduce_(0, cat_t, cat_e, "amax",
                                      include_self=True)
                ex = torch.exp(cat_e - mx[cat_t])
                den = torch.zeros(n, self.heads).index_add_(0, cat_t, ex)
                alpha = ex / (den[cat_t] + 1e-12)
                agg = torch.zeros(n, self.heads, self.d_head).index_add_(
                    0, cat_t, cat_m * alpha.unsqueeze(-1))
                out = out + agg.reshape(n, self.d_out)
            return self.norm(F.elu(out))

    class RGAT(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            d = d_in
            for _ in range(cfg.gnn_layers):
                layers.append(RGATLayer(d, cfg.gnn_hidden, cfg.gnn_heads))
                d = layers[-1].d_out
            self.layers = nn.ModuleList(layers)
            self.decoder = nn.Linear(d, d_in)
            self.readout = nn.Linear(d, 1)
            self.d_out = d

        def forward(self, x):
            h = x
            for i, layer in enumerate(self.layers):
                new = layer(h)
                h = new if new.shape == h.shape else new
            return h

    model = RGAT()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.gnn_lr)
    res = GNNResult(propagated=True, seed=seed)
    margin = 0.2

    for epoch in range(cfg.gnn_epochs):
        opt.zero_grad()
        H = model(X)
        rec = F.mse_loss(model.decoder(H), X)
        Hn = F.normalize(H, dim=-1)
        loss_rel = H.new_zeros(())
        if same:
            i = torch.tensor([a for a, _ in same], dtype=torch.long)
            j = torch.tensor([b for _, b in same], dtype=torch.long)
            loss_rel = loss_rel + (1 - (Hn[i] * Hn[j]).sum(-1)).mean()
        if conflict:
            i = torch.tensor([a for a, _ in conflict], dtype=torch.long)
            j = torch.tensor([b for _, b in conflict], dtype=torch.long)
            loss_rel = loss_rel + torch.clamp(
                (Hn[i] * Hn[j]).sum(-1) - margin, min=0).mean()
        loss = rec + cfg.gnn_lambda_rel * loss_rel
        loss.backward()
        opt.step()
        res.losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        H = model(X)
        scores = model.readout(H).squeeze(-1)
    res.epochs = cfg.gnn_epochs
    for nid, i in idx.items():
        res.node_scores[nid] = float(scores[i])
        res.embeddings[nid] = H[i].tolist()
    return res


def mean_over_seeds(graph: EventGraph, cfg: TavernConfig) -> GNNResult:
    """Mean node score over the configured seeds.

    The graph network is not deterministic, so every configuration involving it
    is reported as the mean over a stated number of seeded runs (thesis
    Section 9.7).
    """
    acc: Dict[str, float] = {}
    runs = 0
    last: Optional[GNNResult] = None
    for seed in cfg.seeds:
        r = train_and_score(graph, cfg, seed)
        for k, v in r.node_scores.items():
            acc[k] = acc.get(k, 0.0) + v
        runs += 1
        last = r
    out = GNNResult(propagated=True, epochs=cfg.gnn_epochs)
    out.node_scores = {k: v / max(1, runs) for k, v in acc.items()}
    out.embeddings = last.embeddings if last else {}
    out.losses = last.losses if last else []
    return out
