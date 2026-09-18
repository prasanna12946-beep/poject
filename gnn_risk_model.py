"""
Day 2 - GraphSAGE risk-propagation model
Predicts a "failure probability" score per node given the N-tier graph.
No real failure-history labels exist yet at hackathon scale, so this
trains against a synthetic ground truth built from structural features
(in-degree centrality + criticality_score + inferred-edge confidence) —
be upfront about this in the jury Q&A: "labels are a structural proxy;
swap in real disruption-history labels for production."

Requires: torch, torch-geometric
    pip install torch torch-geometric --break-system-packages
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COUNTRIES = ["USA", "China", "India", "Germany", "Vietnam", "Brazil",
             "Japan", "South Korea", "Mexico", "Netherlands"]
INDUSTRIES = ["Electronics", "Automotive", "Textiles", "Pharma",
              "Agrifood", "Semiconductors", "Logistics", "Chemicals"]


class RiskGraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.out(x)).squeeze(-1)


def nx_to_pyg(G: nx.DiGraph) -> tuple[Data, list]:
    node_ids = list(G.nodes)
    idx = {n: i for i, n in enumerate(node_ids)}

    features = []
    for n in node_ids:
        attrs = G.nodes[n]
        country_oh = [1.0 if attrs.get("country") == c else 0.0 for c in COUNTRIES]
        industry_oh = [1.0 if attrs.get("industry") == i else 0.0 for i in INDUSTRIES]
        crit = [float(attrs.get("criticality_score", 0.5))]
        tier = [float(attrs.get("tier", 1))]
        features.append(country_oh + industry_oh + crit + tier)

    x = torch.tensor(features, dtype=torch.float)
    edges = [(idx[u], idx[v]) for u, v in G.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Structural proxy label: nodes with high in-degree (many dependents)
    # AND high criticality are the ones whose failure should score higher.
    in_deg = dict(G.in_degree())
    max_deg = max(in_deg.values()) or 1
    y = torch.tensor([
        0.5 * (in_deg[n] / max_deg) + 0.5 * G.nodes[n].get("criticality_score", 0.5)
        for n in node_ids
    ], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, node_ids


def train(data: Data, epochs=200, lr=0.01):
    model = RiskGraphSAGE(in_dim=data.x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = F.mse_loss(pred, data.y)
        loss.backward()
        opt.step()
        if epoch % 40 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
    return model


if __name__ == "__main__":
    with open(DATA_DIR / "trade_graph_ntier.json") as f:
        graph_json = json.load(f)
    G = nx.node_link_graph(graph_json, directed=True)

    data, node_ids = nx_to_pyg(G)
    model = train(data)

    model.eval()
    with torch.no_grad():
        scores = model(data.x, data.edge_index).numpy()

    risk_scores = {nid: round(float(s), 4) for nid, s in zip(node_ids, scores)}
    with open(DATA_DIR / "risk_scores.json", "w") as f:
        json.dump(risk_scores, f, indent=2)

    top5 = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 highest-risk nodes:")
    for nid, score in top5:
        print(f"  {nid} ({G.nodes[nid]['name']}): {score}")
    print(f"\nSaved all scores to {DATA_DIR / 'risk_scores.json'}")
