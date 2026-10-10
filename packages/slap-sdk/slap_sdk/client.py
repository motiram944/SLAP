"""SLAP Python Client (slap-sdk).

Client library for ingesting, retrieving, and embedding continuous hyperspherical
latent vectors with topological graph acceleration.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import httpx

from slap_sdk.models import (
    GuardInspectRequest,
    GuardVerdict,
    HyperEdge,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)


class SlapClient:
    """Official developer client for the Semantic Latent-Algebraic Protocol (SLAP)."""

    def __init__(self, base_url: str = "https://slap-9m29.onrender.com", timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health(self) -> Dict[str, Any]:
        """Check status of connected storage cluster and inference gateway."""
        resp = self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    def ingest(
        self,
        text: str,
        edges: Optional[List[Union[HyperEdge, Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        epistemic_variance: float = 0.01,
    ) -> IngestResponse:
        """Ingest raw domain text into the SLAP manifold.
        Quantizes residual deltas to INT8 and indexes into Qdrant Cloud.
        """
        parsed_edges = None
        if edges:
            parsed_edges = [
                e if isinstance(e, HyperEdge) else HyperEdge(**e)
                for e in edges
            ]

        req = IngestRequest(
            text=text,
            edges=parsed_edges,
            metadata=metadata or {},
            epistemic_variance=epistemic_variance,
        )

        resp = self._http.post("/api/v1/ingest", json=req.model_dump())
        resp.raise_for_status()
        return IngestResponse(**resp.json())

    def retrieve(
        self,
        query: str,
        drift_threshold: float = 0.0,
        query_entities: Optional[List[str]] = None,
        top_k: int = 5,
        max_epistemic_variance: float = 1.0,
    ) -> RetrieveResponse:
        """Retrieve continuous latents with topological graph pruning.
        Provides up to 96%+ KV Cache context reduction vs traditional RAG.
        """
        req = RetrieveRequest(
            query=query,
            drift_threshold=drift_threshold,
            query_entities=query_entities,
            top_k=top_k,
            max_epistemic_variance=max_epistemic_variance,
        )

        resp = self._http.post("/api/v1/retrieve", json=req.model_dump())
        resp.raise_for_status()
        return RetrieveResponse(**resp.json())

    def embed(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Compute continuous unit hypersphere S^{d-1} embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        resp = self._http.post("/api/v1/embed", json={"texts": texts})
        resp.raise_for_status()
        return resp.json()

    def inspect_guard(
        self,
        vector: List[float],
        epistemic_variance: float = 0.01,
        drift_threshold: float = 0.82,
        max_variance: float = 0.05,
    ) -> GuardVerdict:
        """Inspect vector manifold bounds and drift telemetry."""
        req = GuardInspectRequest(
            vector=vector,
            epistemic_variance=epistemic_variance,
            drift_threshold=drift_threshold,
            max_variance=max_variance,
        )
        resp = self._http.post("/api/v1/guard/inspect", json=req.model_dump())
        resp.raise_for_status()
        return GuardVerdict(**resp.json())
