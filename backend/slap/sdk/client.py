"""SLAP Python SDK (slap-sdk).

High-level developer client library for ingesting data, performing topological
pruned ANN retrieval, and inspecting Latent WAF telemetry.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import httpx

from slap.core.schema import (
    GuardInspectRequest,
    GuardVerdict,
    HyperEdge,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    SlapEnvelope,
)


class SlapClient:
    """Developer-facing SDK client for interacting with the SLAP gateway."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 10.0):
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
        """Check gateway health status."""
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
        Automatically maps to S^{d-1}, quantizes residual delta to INT8,
        registers hypergraph topology, and indexes into Qdrant.
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
        drift_threshold: float = 0.82,
        query_entities: Optional[List[str]] = None,
        top_k: int = 5,
        max_epistemic_variance: float = 0.05,
    ) -> RetrieveResponse:
        """Retrieve continuous latents with topological graph pruning and drift gating."""
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

    def inspect_guard(
        self,
        vector: List[float],
        epistemic_variance: float = 0.01,
        drift_threshold: float = 0.82,
        max_variance: float = 0.05,
    ) -> GuardVerdict:
        """Query Latent WAF security gateway to inspect vector integrity and manifold bounds."""
        req = GuardInspectRequest(
            vector=vector,
            epistemic_variance=epistemic_variance,
            drift_threshold=drift_threshold,
            max_variance=max_variance,
        )

        resp = self._http.post("/api/v1/guard/inspect", json=req.model_dump())
        resp.raise_for_status()
        return GuardVerdict(**resp.json())

    def get_subgraph(self, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch topological entity-relation graph nodes and links for visualization."""
        params = {}
        if doc_ids:
            params["doc_ids"] = ",".join(doc_ids)
        resp = self._http.get("/api/v1/graph/subgraph", params=params)
        resp.raise_for_status()
        return resp.json()
