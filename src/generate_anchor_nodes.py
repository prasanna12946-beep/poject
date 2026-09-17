"""
Day 1 - Layer 1: Open-source Anchor Nodes
Generates a synthetic-but-realistic Tier-1 dataset standing in for
public corporate filings / port manifests, since scraping real trade
records mid-hackathon isn't feasible. Swap load_real_sources() in
later once you have actual filings/UN Comtrade data.

Output: data/anchor_nodes.csv, data/anchor_edges.csv
"""
import csv
import random
from pathlib import Path

random.seed(42)

INDUSTRIES = ["Electronics", "Automotive", "Textiles", "Pharma",
              "Agrifood", "Semiconductors", "Logistics", "Chemicals"]
COUNTRIES = ["USA", "China", "India", "Germany", "Vietnam", "Brazil",
             "Japan", "South Korea", "Mexico", "Netherlands"]
NAME_PREFIXES = ["Orion", "Vertex", "Nimbus", "Atlas", "Kestrel", "Solace",
                  "Ferrovia", "Brightloom", "Cascadia", "Ironclad", "Meridian",
                  "Solstice", "Quantum", "Harbor", "Evergreen", "Zenith",
                  "Palisade", "Northstar", "Tundra", "Beacon"]
NAME_SUFFIXES = ["Industries", "Group", "Manufacturing", "Logistics Co.",
                  "Materials", "Holdings", "Components", "Trading Ltd.",
                  "Systems", "Foundry", "Textiles Ltd.", "Chemicals Inc."]


def fake_company_name(i: int) -> str:
    return f"{NAME_PREFIXES[i % len(NAME_PREFIXES)]} {NAME_SUFFIXES[(i * 7) % len(NAME_SUFFIXES)]}"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_nodes(n_companies: int = 120):
    nodes = []
    for i in range(n_companies):
        nodes.append({
            "node_id": f"N{i:04d}",
            "name": fake_company_name(i),
            "country": random.choice(COUNTRIES),
            "industry": random.choice(INDUSTRIES),
            "tier": 1,  # anchor layer = Tier-1 ground truth
            "criticality_score": round(random.uniform(0.1, 1.0), 3),
        })
    return nodes


def generate_edges(nodes, avg_suppliers_per_company: int = 3):
    """Buyer -> Supplier trade links (Tier-1 only; deeper tiers come
    from the synthetic generator in Day 2)."""
    edges = []
    ids = [n["node_id"] for n in nodes]
    for n in nodes:
        n_links = random.randint(1, avg_suppliers_per_company)
        suppliers = random.sample([i for i in ids if i != n["node_id"]], n_links)
        for s in suppliers:
            edges.append({
                "source": n["node_id"],       # buyer
                "target": s,                  # supplier
                "relation": "sources_from",
                "trade_volume_usd_m": round(random.uniform(0.5, 50), 2),
            })
    return edges


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    nodes = generate_nodes()
    edges = generate_edges(nodes)

    write_csv(nodes, DATA_DIR / "anchor_nodes.csv",
              ["node_id", "name", "country", "industry", "tier", "criticality_score"])
    write_csv(edges, DATA_DIR / "anchor_edges.csv",
              ["source", "target", "relation", "trade_volume_usd_m"])

    print(f"Generated {len(nodes)} nodes, {len(edges)} Tier-1 edges")
    print(f"Written to {DATA_DIR}")
