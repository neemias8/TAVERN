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

def build_event_graph(events, chronology_table):
    G = create_graph(events)
    # Add edges from alignments/relations
    for event in events:
        for rel in event['relations']:
            G.add_edge(event['id'], rel['to'], type=rel['type'])
    
    # Add temporal edges from chronology order
    for i, row in enumerate(chronology_table[:-1]):
        next_row = chronology_table[i+1]
        # Assume events in row i BEFORE next_row
        current_events = [e['id'] for e in events if row['id'] in e.get('chrono_id', '')]
        next_events = [e['id'] for e in events if next_row['id'] in e.get('chrono_id', '')]
        for ce in current_events:
            for ne in next_events:
                G.add_edge(ce, ne, type='BEFORE')
    return G

def run_gnn(events, chronology_table, embed_dim=768):  # Match BART embed size
    G = build_event_graph(events, chronology_table)
    adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.float)
    node_features = torch.randn(len(G.nodes), embed_dim)
    
    model = SimpleGCN(embed_dim, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(10):  # Dummy training
        out = model(node_features, adj)
        loss = torch.mean(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    enriched_events = {node: out[i].detach().numpy() for i, node in enumerate(G.nodes)}
    return enriched_events