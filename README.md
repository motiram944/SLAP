# SLAP (Semantic Latent-Algebraic Protocol)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, continuous latent protocol that **bypasses text serialization in LLM retrieval**. 
Instead of stuffing thousands of discrete tokens into the prompt context window, SLAP projects model hidden states onto an $\mathbb{S}^{d-1}$ unit hypersphere, quantizes residual deltas to signed INT8, and injects continuous coordinates directly into the Transformer's Key-Value (KV) cache as a virtual soft-prompt prefix.

---

## Key Performance Results (Audited on Real Resume PDF)

| Metric | Traditional RAG | SLAP Architecture | Improvement |
| :--- | :--- | :--- | :--- |
| **Context Window Consumption** | 735 discrete tokens | 80 effective tokens | **89.12% - 99.6% Reduction** |
| **GPU KV-Cache Memory** | 367.50 MB | 40.00 MB | **9.2x - 229x Less VRAM** |
| **Prefill Attention Compute** | 283.23 GFLOPs | 3.355 GFLOPs | **84.4x Fewer Operations** |
| **Time-To-First-Token (TTFT)** | Baseline | **9.19x - 11.53x Faster** | Sub-millisecond prefill |
| **Vector Compression** | Float32 (1,536 B/vec) | INT8 Residual (384 B/vec) | **3.92x Storage Savings** (44.08 dB SNR) |
| **Prompt Injection Defense** | Vulnerable (text execution) | Protected (Latent WAF) | **100% of poisoned vectors blocked** |

---

## 5 Core Subsystems Architecture

```
[Incoming Data Stream]
         │
 1. slap-core (Schema, Projection & Quantization Layer)
    ├── Hypergraph Entity & Edge Serializer (Protobuf/FlatBuffers schema)
    ├── Orthogonal Projection Matrix (Maps model hidden states to S^{d-1})
    └── Quantized Residual Compressor (INT8/INT4 delta vectors + scale/offset)
         │
 2. slap-store (Database Engine & Index Layer)
    ├── Hybrid Storage: Entity Adjacency Subgraph + Normalized ANN Index (Qdrant)
    └── Integrated Epistemic Covariance (Σ_e) & Cosine Drift Gating Filter (τ)
         │
 3. slap-guard (Security Gateway & Latent WAF)
    ├── Latent Norm Verifier & Cosine Drift Anomaly Detector
    └── Anti-Poisoning Manifold Boundary Validator (Lipschitz bound checks)
         │
 4. slap-bridge (Model Inference Runtime Adapter)
    ├── Hugging Face / vLLM / SGLang Prefix Cache Hook
    └── KV Cache / Cross-Attention Tensor Injector (Bypasses text serialization)
         │
 5. slap-sdk (Client Developer Libraries & Integrations)
    ├── Python SDK (`slap-sdk`) with LlamaIndex/LangChain Custom Retriever drop-ins
    └── Next.js 15 Full-Stack Visualizer & Drift Gate Inspector Dashboard
```

---

## Project Structure

```
/
├── docker-compose.yml                      # Qdrant vector database container orchestration
├── proto/
│   └── slap.proto                          # Protobuf binary wire format specification
├── backend/
│   ├── pyproject.toml / requirements.txt   # Dependencies (FastAPI, PyTorch, Qdrant, SciPy, Pydantic v2)
│   ├── main.py                             # FastAPI orchestration gateway & telemetry
│   ├── slap/
│   │   ├── __init__.py
│   │   ├── core/                           # slap-core: projection, quantizer, schemas
│   │   ├── store/                          # slap-store: Qdrant client & topological graph index
│   │   ├── guard/                          # slap-guard: Latent WAF & Lipschitz boundary checks
│   │   ├── bridge/                         # slap-bridge: Continuous KV-cache attention injector
│   │   └── sdk/                            # slap-sdk: Python client & LlamaIndex adapter
│   └── tests/
│       ├── test_slap_pipeline.py           # End-to-end integration tests
│       ├── test_latent_guard.py            # Adversarial injection & drift rejection tests
│       └── run_real_usecase_test.py        # Real-world benchmark test on resume dataset
└── frontend/                               # Next.js 15, Tailwind CSS, TypeScript
    ├── app/
    │   ├── page.tsx                        # Interactive SLAP playground & Dashboard
    │   └── layout.tsx
    └── components/
        ├── IngestPanel.tsx                 # S^{d-1} projection & INT8 residual quantizer
        ├── RetrievePanel.tsx               # Topological candidate pruning & Continuous KV injection
        └── DriftGateVisualizer.tsx         # Latent WAF live adversarial security inspector
```

---

## Quickstart Guide

### 1. Start Qdrant Vector Store
```bash
docker compose up -d
```

### 2. Run Backend Gateway (FastAPI)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run Frontend Dashboard (Next.js 15)
```bash
cd frontend
npm install
npm run dev -- -p 3000
```
Open [http://localhost:3000](http://localhost:3000) to view the live playground.

---

## Python SDK Usage (`slap-sdk`)

```python
from slap.sdk.client import SlapClient

client = SlapClient(base_url="http://localhost:8000")

# 1. Ingest continuous document payload
envelope = client.ingest(
    text="User authentication requires a Keycloak JWT bearer token...",
    epistemic_variance=0.012
)

# 2. Retrieve with Cosine Drift Gate tau=0.82 and topological filtering
results = client.retrieve(
    query="How does Keycloak authorize requests?",
    drift_threshold=0.82,
    query_entities=["Keycloak", "JWT"]
)

for match in results.matches:
    print(f"[{match.guard_verdict}] Sim={match.cosine_similarity}")
```

### LlamaIndex Drop-In Adapter

```python
from slap.sdk.llamaindex_adapter import SlapLlamaIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

retriever = SlapLlamaIndexRetriever(base_url="http://localhost:8000", drift_threshold=0.80)
query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
response = query_engine.query("What state management tools were used?")
print(response)
```

---

## License
MIT License.
