"""
Day 4 - GenAI Decision Agent (RAG)
Blueprint names Pinecone/Milvus, which need accounts + API keys — risky
to depend on mid-hackathon with flaky wifi. This uses a LOCAL FAISS
index instead (same RAG pattern, zero external dependency), and calls
an LLM for the final mitigation write-up. Swap in Pinecone/Milvus later
by replacing FaissStore with their client — the retrieval interface
below is designed to be a drop-in swap.

Requires: pip install faiss-cpu sentence-transformers --break-system-packages
Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) in your environment for the
generation step; retrieval works standalone without it.
"""
import json
import os
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Stand-in "Live Event RAG Feed" (Layer 4) — in the real build this is
# populated by a news-ingestion pipeline. Hand-authored snippets here
# so the demo has something grounded to retrieve during rehearsal.
KNOWLEDGE_BASE = [
    {"id": "kb1", "text": "Vietnam port strikes historically resolve fastest when "
                            "buyers reroute through Laem Chabang, Thailand within 5-7 days."},
    {"id": "kb2", "text": "Semiconductor export controls typically push affected buyers "
                            "toward South Korean and Japanese fabs as alternate sourcing."},
    {"id": "kb3", "text": "Flood-related disruptions in Indian agrifood corridors are "
                            "mitigated by shifting sourcing to Bangladesh or Vietnam textile mills."},
    {"id": "kb4", "text": "Tier-2 chemical suppliers in Netherlands and Germany offer the "
                            "shortest re-routing lead time for EU-bound electronics manufacturers."},
]


class FaissStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts = [d["text"] for d in KNOWLEDGE_BASE]
        embeddings = self.model.encode(self.texts, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype="float32"))

    def query(self, text: str, k: int = 2):
        q_emb = self.model.encode([text], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(q_emb, dtype="float32"), k)
        return [self.texts[i] for i in idxs[0]]


def generate_mitigation_plan(scenario: dict, affected_nodes: list, retrieved_context: list) -> str:
    """Calls Claude (or any LLM) grounded in retrieved context.
    Falls back to a template if no API key is set, so the pipeline
    never breaks during rehearsal."""
    prompt = f"""A supply chain shock has occurred:
Type: {scenario['type']} in {scenario['region']}
Affected industries: {', '.join(scenario['affected_industries'])}
Severity: {scenario['severity']}

Top at-risk nodes: {', '.join(n['name'] for n in affected_nodes[:5])}

Grounding context:
{chr(10).join('- ' + c for c in retrieved_context)}

Write a 3-bullet mitigation plan naming alternative trade routes/nodes,
grounded ONLY in the context above. Do not invent facts not supported by it."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("[No ANTHROPIC_API_KEY set — showing retrieval only]\n"
                + "\n".join(f"- {c}" for c in retrieved_context))

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


if __name__ == "__main__":
    with open(DATA_DIR / "shock_scenarios.json") as f:
        scenarios = json.load(f)
    with open(DATA_DIR / "trade_graph_ntier.json") as f:
        import networkx as nx
        G = nx.node_link_graph(json.load(open(DATA_DIR / "trade_graph_ntier.json")), directed=True, link="edges")

    scenario = scenarios[0]  # SHOCK-001: Vietnam port strike
    affected_nodes = [
        {"id": n, **attrs} for n, attrs in G.nodes(data=True)
        if attrs.get("country") == scenario["region"]
        and attrs.get("industry") in scenario["affected_industries"]
    ]

    store = FaissStore()
    context = store.query(f"{scenario['type']} {scenario['region']} mitigation")

    plan = generate_mitigation_plan(scenario, affected_nodes, context)
    print(f"Scenario: {scenario['type']} — {scenario['region']}")
    print(f"Affected nodes found: {len(affected_nodes)}")
    print("\nMitigation plan:\n" + plan)
