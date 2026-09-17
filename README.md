# DominoRisk AI — Build Guide

Practical implementation of the blueprint, scoped to run on a laptop
with no paid services required (Pinecone/Milvus/Neo4j swapped for free
local equivalents — see notes in each file for how to upgrade if you
have API keys).

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install faiss-cpu sentence-transformers anthropic   # Day 4 only
```

## Run order (mirrors the 5-day roadmap)

**Day 1 — Core graph + anchor nodes**
```bash
python3 src/generate_anchor_nodes.py   # -> data/anchor_nodes.csv, anchor_edges.csv
python3 src/build_graph.py             # -> data/trade_graph.{graphml,json}
```

**Day 2 — N-tier expansion + GNN risk scoring**
```bash
python3 src/synthetic_ntier.py         # -> data/trade_graph_ntier.{graphml,json}
python3 src/gnn_risk_model.py          # -> data/risk_scores.json
```

**Day 3 — 3D globe**
```bash
python3 src/export_geojson.py          # -> data/nodes.geojson, edges.geojson
open frontend/globe.html               # or `python3 -m http.server` and browse
```

**Day 4 — RAG mitigation agent**
```bash
export ANTHROPIC_API_KEY=sk-...        # optional; falls back to retrieval-only
python3 src/rag_agent.py
```

**Day 5 — Stress test**
Loop `rag_agent.py` over all entries in `data/shock_scenarios.json`,
log which nodes/industries get hit each time, and use that output to
sanity-check the globe's color scaling and the jury demo script.

## Known simplifications vs. the blueprint (be upfront about these in Q&A)
- Graph DB = NetworkX + file persistence, not a dedicated graph database —
  fine for hackathon scale (hundreds of nodes); note Neo4j/TigerGraph as
  the production path.
- N-tier generator uses a hand-set industry-affinity table as a stand-in
  for a fitted Probabilistic Graphical Model.
- GNN labels are a structural proxy (in-degree + criticality), not real
  historical failure data — call this out explicitly, it's the single
  weakest link a technical judge will probe.
- RAG uses local FAISS + a hand-written knowledge base instead of
  Pinecone/Milvus + a live news feed.
- Zero-Trust security architecture is a design story for the pitch, not
  implemented in this codebase — don't imply otherwise on stage.

## What's not built yet
- Real Live Event RAG Feed (news ingestion + vectorization pipeline)
- Zero-Trust access layer
- Any deployment/hosting step for Day 5's "Final Deployment & Submission"
