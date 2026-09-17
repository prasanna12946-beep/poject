"""
Day 1 - Core Graph Database & Anchor Node Ingestion
Loads Tier-1 nodes/edges into a NetworkX DiGraph (the topological layer
the blueprint calls for) and persists it in two formats:
  - GraphML: for Gephi / visual inspection / handoff to Day 3 (3D globe)
  - node-link JSON: easiest to feed into the GNN pipeline (Day 2) and
    into the frontend later

For a hackathon timeline, NetworkX in-memory + serialized file IS your
"graph database" — don't waste Day 1 standing up Neo4j unless someone
on the team already knows it cold.
"""
import json
from pathlib import Path
import networkx as nx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_graph(nodes_csv=DATA_DIR / "anchor_nodes.csv",
               edges_csv=DATA_DIR / "anchor_edges.csv") -> nx.DiGraph:
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], **row.drop("node_id").to_dict())
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"],
                    relation=row["relation"],
                    trade_volume_usd_m=row["trade_volume_usd_m"])
    return G


def basic_stats(G: nx.DiGraph):
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    if nx.is_weakly_connected(G):
        print("Graph is weakly connected")
    else:
        n_components = nx.number_weakly_connected_components(G)
        print(f"Graph has {n_components} disconnected components "
              f"(expected pre-Day-2 synthetic expansion)")
    # top nodes by in-degree = most depended-upon suppliers = risk hotspots
    top = sorted(G.in_degree, key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 most-depended-upon nodes (highest in-degree):")
    for node_id, deg in top:
        print(f"  {node_id} ({G.nodes[node_id]['name']}): {deg} dependents")


def export(G: nx.DiGraph):
    nx.write_graphml(G, DATA_DIR / "trade_graph.graphml")

    graph_json = nx.node_link_data(G)
    with open(DATA_DIR / "trade_graph.json", "w") as f:
        json.dump(graph_json, f, indent=2)

    print(f"Exported: {DATA_DIR / 'trade_graph.graphml'}")
    print(f"Exported: {DATA_DIR / 'trade_graph.json'}")


if __name__ == "__main__":
    G = load_graph()
    basic_stats(G)
    export(G)
