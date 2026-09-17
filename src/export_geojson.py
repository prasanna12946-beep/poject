"""
Day 3 - GeoJSON export for the WebGL 3D Globe (Globe.gl)
Assigns each country a representative lat/lon (capital-ish centroid)
and jitters node positions around it so companies in the same country
don't stack exactly on top of each other on the globe.
"""
import json
import random
from pathlib import Path
import networkx as nx

random.seed(3)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Rough centroids, good enough for a hackathon globe demo
COUNTRY_COORDS = {
    "USA": (37.09, -95.71), "China": (35.86, 104.20), "India": (20.59, 78.96),
    "Germany": (51.17, 10.45), "Vietnam": (14.06, 108.28), "Brazil": (-14.24, -51.93),
    "Japan": (36.20, 138.25), "South Korea": (35.91, 127.77), "Mexico": (23.63, -102.55),
    "Netherlands": (52.13, 5.29),
}


def jitter(lat, lon, spread=2.5):
    return lat + random.uniform(-spread, spread), lon + random.uniform(-spread, spread)


def build_geojson(G: nx.DiGraph, risk_scores: dict):
    node_features = []
    coords = {}
    for n, attrs in G.nodes(data=True):
        base_lat, base_lon = COUNTRY_COORDS.get(attrs["country"], (0, 0))
        lat, lon = jitter(base_lat, base_lon)
        coords[n] = (lat, lon)
        node_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "id": n, "name": attrs.get("name"), "country": attrs.get("country"),
                "industry": attrs.get("industry"), "tier": attrs.get("tier"),
                "risk_score": risk_scores.get(n, 0.0),
            },
        })

    edge_features = []
    for u, v, attrs in G.edges(data=True):
        lat1, lon1 = coords[u]
        lat2, lon2 = coords[v]
        edge_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
            "properties": {"source": u, "target": v,
                            "trade_volume_usd_m": attrs.get("trade_volume_usd_m")},
        })

    return (
        {"type": "FeatureCollection", "features": node_features},
        {"type": "FeatureCollection", "features": edge_features},
    )


if __name__ == "__main__":
    with open(DATA_DIR / "trade_graph_ntier.json") as f:
        G = nx.node_link_graph(json.load(f), directed=True, link="edges")

    risk_path = DATA_DIR / "risk_scores.json"
    risk_scores = json.load(open(risk_path)) if risk_path.exists() else {}

    nodes_geo, edges_geo = build_geojson(G, risk_scores)

    with open(DATA_DIR / "nodes.geojson", "w") as f:
        json.dump(nodes_geo, f, indent=2)
    with open(DATA_DIR / "edges.geojson", "w") as f:
        json.dump(edges_geo, f, indent=2)

    print(f"Wrote {len(nodes_geo['features'])} node points, "
          f"{len(edges_geo['features'])} edge lines")
