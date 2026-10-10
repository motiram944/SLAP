"""FastAPI Orchestration Gateway for SLAP.

Coordinates the 5 core subsystems:
1. slap-core: Orthogonal S^{d-1} projection & INT8 residual quantization
2. slap-store: In-memory topological graph pruning + Qdrant ANN search
3. slap-guard: Latent WAF manifold verification, spectral anomaly, and drift gating
4. slap-bridge: LLM continuous KV-cache prefix tensor injection & token savings
5. slap-sdk: Exposed API endpoints for SDK clients and Next.js visualizer
"""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query


from fastapi.middleware.cors import CORSMiddleware

from slap.core.schema import (
    CanonicalLatent,
    GuardInspectRequest,
    GuardVerdict,
    HyperEdge,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    SlapEnvelope,
    compute_digest,
    extract_hypergraph_entities,
)
from slap.core.projection import HypersphereProjector
from slap.core.quantizer import ResidualQuantizer
from slap.store.graph_index import TopologicalGraphIndex
from slap.store.qdrant_store import QdrantSlapStore
from slap.guard.latent_waf import LatentWAF
from slap.bridge.attention_injector import ContinuousAttentionBridge

app = FastAPI(
    title="SLAP Gateway (Semantic Latent-Algebraic Protocol)",
    description="Production-grade continuous latent orchestration gateway",
    version="0.1.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize core subsystems
projector = HypersphereProjector(dimension=384, seed=42)
quantizer = ResidualQuantizer()
graph_index = TopologicalGraphIndex()
qdrant_store = QdrantSlapStore(dimension=384)
latent_waf = LatentWAF(norm_tolerance=1e-4, default_drift_threshold=0.82, default_max_variance=0.05)
bridge = ContinuousAttentionBridge(latent_dim=384, llm_hidden_dim=4096)

# Global telemetry tracking
telemetry = {
    "total_ingested": 0,
    "total_retrievals": 0,
    "total_tokens_saved": 0,
    "waf_blocks": 0,
    "waf_passes": 0,
}


def _ingest_document(
    text: str,
    custom_edges: Optional[List[HyperEdge]],
    metadata: Optional[Dict[str, Any]],
    epistemic_var: float,
) -> IngestResponse:
    doc_id = str(uuid.uuid4())
    digest = compute_digest(text)

    # 1. Project to S^{d-1}
    z_raw, z_norm = projector.embed_and_project(text)

    # 2. Latent WAF Check on projected vector (Audit telemetry)
    waf_result = latent_waf.inspect_vector(z_norm, epistemic_variance=epistemic_var)
    if not np.all(np.isfinite(z_norm)):
        telemetry["waf_blocks"] += 1
        raise HTTPException(status_code=400, detail="Corrupted non-finite tensor")

    if waf_result.passed:
        telemetry["waf_passes"] += 1
    else:
        telemetry["waf_blocks"] += 1

    # 3. Residual Compression (INT8)
    residual = quantizer.compute_residual(z_raw, z_norm)

    q_bytes, scale, zero_point, comp_stats = quantizer.quantize_int8(residual)

    # 4. Hypergraph Entity Extraction
    extracted_edges = custom_edges if custom_edges else extract_hypergraph_entities(text)

    # Build Envelope
    latent = CanonicalLatent.from_bytes(
        hypersphere_coord=z_norm.tolist(),
        residual_bytes=q_bytes,
        residual_scale=scale,
        residual_zero_point=zero_point,
        epistemic_variance=epistemic_var,
        dimension=projector.dimension,
    )

    envelope = SlapEnvelope(
        id=doc_id,
        edges=extracted_edges,
        latent=latent,
        raw_digest=digest,
        text=text,
        metadata=metadata or {},
    )

    # 5. Index into Graph Store & Qdrant
    graph_index.add_envelope(doc_id, extracted_edges)
    qdrant_store.upsert_envelope(envelope)

    # 6. Compute Token Savings
    token_savings = bridge.compute_token_savings(text, num_latents=1)

    # Telemetry updates
    telemetry["total_ingested"] += 1
    telemetry["total_tokens_saved"] += token_savings["tokens_bypassed"]

    return IngestResponse(
        envelope=envelope,
        compression_stats=comp_stats,
        token_savings=token_savings,
    )


@app.on_event("startup")
def bootstrap_sample_data():
    """Seed initial enterprise documents for instant playground exploration."""
    sample_docs = [
        {
            "text": "User authentication requires a Keycloak JWT bearer token with RS256 signatures to authorize downstream microservice requests.",
            "metadata": {"category": "security", "author": "Architecture Team"},
            "variance": 0.012,
        },
        {
            "text": "The Qdrant vector database indexes S^{d-1} continuous hypersphere coordinates using Cosine distance for sub-millisecond ANN search.",
            "metadata": {"category": "database", "author": "Data Platform"},
            "variance": 0.018,
        },
        {
            "text": "Latent WAF guards the inference gateway by checking Lipschitz bounds and filtering high-frequency adversarial perturbations.",
            "metadata": {"category": "defense", "author": "SecOps"},
            "variance": 0.022,
        },
        {
            "text": "Continuous KV cache injection prepends virtual soft-prompt tokens directly to LLM attention, bypassing discrete text serialization.",
            "metadata": {"category": "deep_learning", "author": "Compiler Group"},
            "variance": 0.015,
        },
    ]

    for item in sample_docs:
        try:
            _ingest_document(
                text=item["text"],
                custom_edges=None,
                metadata=item["metadata"],
                epistemic_var=item["variance"],
            )
        except Exception as e:
            print(f"Bootstrap doc failed: {e}")

# Run initial bootstrap
bootstrap_sample_data()



@app.get("/")
def root():
    return {
        "protocol": "SLAP",
        "description": "Semantic Latent-Algebraic Protocol Gateway",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "qdrant_vectors": qdrant_store.count(),
        "graph_entities": graph_index.stats()["unique_entities"],
        "telemetry": telemetry,
    }


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest):
    return _ingest_document(
        text=payload.text,
        custom_edges=payload.edges,
        metadata=payload.metadata,
        epistemic_var=payload.epistemic_variance or 0.01,
    )


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(payload: RetrieveRequest):
    start_time = time.time()
    telemetry["total_retrievals"] += 1

    # 1. Project query to S^{d-1}
    _, q_norm = projector.embed_and_project(payload.query)

    # 2. Extract query entities if not explicitly passed
    query_ents = payload.query_entities
    if not query_ents:
        extracted = extract_hypergraph_entities(payload.query)
        query_ents = [e.source for e in extracted] + [e.target for e in extracted]
        query_ents = list(dict.fromkeys(query_ents))  # deduplicate preserving order

    # 3. Topological Candidate Pruning
    total_docs = qdrant_store.count()
    candidate_ids = graph_index.prune_candidates(query_ents, max_hops=1)
    pruned_count = max(0, total_docs - len(candidate_ids))

    # 4. Search ANN in Qdrant with Drift Gate evaluation
    matches = qdrant_store.search_latents(
        query_vector=q_norm.tolist(),
        top_k=payload.top_k,
        candidate_ids=candidate_ids if len(candidate_ids) > 0 else None,
        drift_threshold=payload.drift_threshold,
        max_variance=payload.max_epistemic_variance,
    )

    # Count WAF blocks vs passes
    for m in matches:
        if m.passed_drift_gate:
            telemetry["waf_passes"] += 1
        else:
            telemetry["waf_blocks"] += 1

    # Token savings calculation for retrieved documents
    combined_text = " ".join(m.text for m in matches if m.passed_drift_gate)
    token_savings = bridge.compute_token_savings(combined_text, num_latents=max(1, len(matches)))

    elapsed_ms = round((time.time() - start_time) * 1000.0, 2)

    return RetrieveResponse(
        matches=matches,
        total_candidates=len(candidate_ids),
        topological_pruned=pruned_count,
        execution_time_ms=elapsed_ms,
        token_savings=token_savings,
        query_entities=query_ents,
    )


@app.post("/api/v1/embed")
def embed_endpoint(payload: Dict[str, Any]):
    """Global Embedding API: Generates S^{d-1} continuous normalized vector embeddings."""
    texts = payload.get("texts") or ([payload.get("text")] if payload.get("text") else [])
    if not texts:
        raise HTTPException(status_code=400, detail="Must provide 'text' string or 'texts' array")

    data = []
    for idx, t in enumerate(texts):
        z_raw, z_norm = projector.embed_and_project(str(t))
        residual = quantizer.compute_residual(z_raw, z_norm)
        _, scale, zp, stats = quantizer.quantize_int8(residual)
        data.append({
            "index": idx,
            "embedding": z_norm.tolist(),
            "dimension": projector.dimension,
            "norm": round(float(np.linalg.norm(z_norm)), 6),
            "compression_stats": stats,
        })

    return {
        "object": "list",
        "data": data,
        "model": "slap-continuous-hypersphere-v1",
        "dimension": projector.dimension,
    }


@app.post("/api/v1/rag")
def rag_search_endpoint(payload: Dict[str, Any]):
    """Standard RAG Search API: Returns retrieved documents & context for any external LLM."""
    query = payload.get("query", "")
    top_k = payload.get("top_k", 5)
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    req = RetrieveRequest(query=query, top_k=top_k, drift_threshold=0.0)
    ret_res = retrieve_endpoint(req)

    documents = [
        {
            "id": m.id,
            "text": m.text,
            "similarity": m.cosine_similarity,
            "metadata": m.metadata,
        }
        for m in ret_res.matches
    ]
    context = "\n\n".join([f"[{d['id'][:8]}]: {d['text']}" for d in documents])

    return {
        "query": query,
        "documents": documents,
        "context": context,
        "total_results": len(documents),
        "execution_time_ms": ret_res.execution_time_ms,
    }



@app.post("/api/v1/guard/inspect", response_model=GuardVerdict)
def guard_inspect_endpoint(payload: GuardInspectRequest):
    return latent_waf.inspect_vector(
        vector=payload.vector,
        epistemic_variance=payload.epistemic_variance,
        drift_threshold=payload.drift_threshold,
        max_variance=payload.max_variance,
    )


@app.post("/api/v1/query")
async def query_with_gemini_pro(payload: RetrieveRequest):
    """Retrieve with SLAP Latent WAF, and pass verified context to Google Gemini Pro."""
    # 1. Run SLAP Continuous Retrieval
    ret_response = retrieve_endpoint(payload)
    verified = [m for m in ret_response.matches if m.passed_drift_gate]

    gemini_key = os.getenv("GEMINI_API_KEY")
    answer = ""

    if not verified:
        answer = "SLAP Latent WAF Alert: No documents passed the Cosine Drift Gate or Epistemic Covariance threshold. Query blocked from reaching Gemini to prevent hallucination."
    elif gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-pro")
            
            context_text = "\n\n".join([f"[Source {i+1}]: {m.text}" for i, m in enumerate(verified)])
            system_prompt = (
                "You are an expert AI reasoning assistant backed by the SLAP (Semantic Latent-Algebraic Protocol). "
                "Answer the user query accurately and comprehensively using strictly the verified context retrieved below.\n\n"
                f"VERIFIED CONTEXT:\n{context_text}\n\n"
                f"USER QUERY:\n{payload.query}"
            )
            response = model.generate_content(system_prompt)
            answer = response.text
        except Exception as e:
            answer = f"Gemini Pro Execution Note: {str(e)}"
    else:
        answer = (
            f"SLAP verified {len(verified)} continuous envelopes (passed Latent WAF). "
            "To generate the final LLM response with Gemini Pro, set your GEMINI_API_KEY in backend/.env!"
        )

    return {
        "answer": answer,
        "matches": ret_response.matches,
        "verified_count": len(verified),
        "total_candidates": ret_response.total_candidates,
        "token_savings": ret_response.token_savings,
        "execution_time_ms": ret_response.execution_time_ms,
    }


@app.post("/api/v1/bridge/inject")
def bridge_inject_endpoint(payload: Dict[str, Any]):

    """Project a continuous latent vector to LLM dimension and simulate KV cache injection."""
    vector = payload.get("vector", [])
    if not vector:
        raise HTTPException(status_code=400, detail="Missing vector in payload")

    virtual_token = bridge.project_to_virtual_token(vector)
    kv_summary = bridge.synthesize_kv_prefix(virtual_token)

    return {
        "status": "SUCCESS",
        "virtual_token_dimension": int(virtual_token.shape[0]),
        "virtual_token_l2_norm": round(float(np.linalg.norm(virtual_token)), 4),
        "kv_cache_injection": kv_summary,
    }


@app.get("/api/v1/graph/subgraph")
def graph_subgraph_endpoint(doc_ids: Optional[str] = Query(None)):
    target_ids = doc_ids.split(",") if doc_ids else list(graph_index.doc_to_edges.keys())
    return graph_index.get_subgraph(target_ids)


@app.get("/api/v1/stats")
def stats_endpoint():
    return {
        "telemetry": telemetry,
        "graph": graph_index.stats(),
        "store_count": qdrant_store.count(),
        "projection_dim": projector.dimension,
        "llm_hidden_dim": bridge.llm_hidden_dim,
    }
