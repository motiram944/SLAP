# slap-sdk

Official Python client for the **Semantic Latent-Algebraic Protocol (SLAP)**.

A modern, high-performance alternative to traditional RAG architectures (Pinecone / Chroma / LangChain) that operates directly in continuous hyperspherical latent space ($S^{d-1}$), achieving **96%+ KV Cache context reduction** and **sub-millisecond retrieval**.

---

## ⚡ Installation

```bash
pip install slap-sdk
```

---

## 🚀 Quickstart

### 1. Connect to the Live Global Cluster
```python
from slap_sdk import SlapClient

# Automatically connects to the global production cluster
client = SlapClient(base_url="https://slap-9m29.onrender.com")

# Health check
print(client.health())
```

### 2. Ingest Data (Cloud Storage + Compression)
```python
resp = client.ingest(
    text="Kubernetes orchestrates containerized services with declarative manifests.",
    metadata={"service": "infra"}
)
print("Ingested Doc ID:", resp.doc_id)
print("INT8 Compression SNR:", resp.compression_stats["snr_db"])
```

### 3. Retrieve Context with 96% Token Reduction
```python
results = client.retrieve(query="container orchestration", top_k=3)

for match in results.matches:
    print(f"Similarity: {match.cosine_similarity:.4f} | Text: {match.text}")

print("KV Cache Savings:", results.token_savings["context_reduction_percent"], "%")
```

### 4. Continuous Unit Hypersphere Embeddings
```python
embedding = client.embed(["Microservices architecture and event-driven design"])
print("Dimension:", embedding["dimension"])
print("Vector (first 5 dims):", embedding["data"][0]["embedding"][:5])
```

---

## 🔗 LlamaIndex Integration

```python
from slap_sdk import SlapLlamaIndexRetriever

retriever = SlapLlamaIndexRetriever(base_url="https://slap-9m29.onrender.com")
nodes = retriever.retrieve("distributed systems architecture")

for n in nodes:
    print(n.score, n.node.text)
```

---

## 📄 License
Apache-2.0
