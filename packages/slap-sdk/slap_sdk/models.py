"""Data schemas for slap-sdk."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HyperEdge(BaseModel):
    source: str
    relation: str
    target: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class CanonicalLatent(BaseModel):
    id: str
    unit_vector: List[float]
    quantized_residual: str
    residual_scale: float
    residual_zero_point: int
    epistemic_variance: float = 0.01
    dimension: int = 384


class IngestRequest(BaseModel):
    text: str
    edges: Optional[List[HyperEdge]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    epistemic_variance: float = 0.01


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    envelope: Dict[str, Any]
    compression_stats: Dict[str, Any]
    token_savings: Dict[str, Any]


class RetrievedMatch(BaseModel):
    id: str
    text: str
    cosine_similarity: float
    epistemic_variance: float
    edges: List[HyperEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    passed_drift_gate: bool = True
    guard_verdict: str = "VERIFIED"
    reconstructed_norm: float = 1.0
    snr_db: float = 48.0


class RetrieveRequest(BaseModel):
    query: str
    drift_threshold: float = 0.0
    query_entities: Optional[List[str]] = None
    top_k: int = 5
    max_epistemic_variance: float = 1.0


class RetrieveResponse(BaseModel):
    matches: List[RetrievedMatch]
    total_candidates: int
    topological_pruned: int
    execution_time_ms: float
    token_savings: Dict[str, Any]
    query_entities: List[str] = Field(default_factory=list)


class GuardInspectRequest(BaseModel):
    vector: List[float]
    epistemic_variance: float = 0.01
    drift_threshold: float = 0.82
    max_variance: float = 0.05


class GuardVerdict(BaseModel):
    is_admissible: bool
    status: str
    reason: str
    telemetry: Dict[str, Any] = Field(default_factory=dict)
