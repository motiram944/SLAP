"""SLAP Core Data Models & Schemas.
Pydantic v2 schemas and Protobuf-compatible mappings for HyperEdges,
CanonicalLatents, and SlapEnvelopes.
"""

from __future__ import annotations
import base64
import hashlib
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HyperEdge(BaseModel):
    """Hypergraph entity-relation edge connecting entities in continuous topology."""
    source: str = Field(..., description="Source entity node identifier")
    relation: str = Field(..., description="Semantic relation or predicate")
    target: str = Field(..., description="Target entity or attribute node")
    properties: Dict[str, str] = Field(default_factory=dict, description="Edge property map")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction or edge confidence score")


class CanonicalLatent(BaseModel):
    """Canonical continuous latent coordinates mapped to S^{d-1} with quantized residual."""
    hypersphere_coord: List[float] = Field(..., description="Normalized vector on unit sphere S^{d-1}")
    quantized_residual: str = Field(..., description="Base64-encoded INT8/INT4 residual delta bytes")
    residual_scale: float = Field(default=1.0, description="Dynamic quantization scale factor")
    residual_zero_point: int = Field(default=0, description="Signed quantization zero point offset")
    epistemic_variance: float = Field(default=0.01, description="Model epistemic uncertainty / covariance Σ_e")
    dimension: int = Field(default=384, description="Latent vector dimensionality")

    def get_residual_bytes(self) -> bytes:
        return base64.b64decode(self.quantized_residual)

    @classmethod
    def from_bytes(
        cls,
        hypersphere_coord: List[float],
        residual_bytes: bytes,
        residual_scale: float,
        residual_zero_point: int,
        epistemic_variance: float,
        dimension: int,
    ) -> CanonicalLatent:
        return cls(
            hypersphere_coord=hypersphere_coord,
            quantized_residual=base64.b64encode(residual_bytes).decode("ascii"),
            residual_scale=residual_scale,
            residual_zero_point=residual_zero_point,
            epistemic_variance=epistemic_variance,
            dimension=dimension,
        )


class SlapEnvelope(BaseModel):
    """Top-level SLAP envelope encapsulating topological graph and algebraic latent state."""
    id: str = Field(..., description="Unique envelope UUID or hash")
    edges: List[HyperEdge] = Field(default_factory=list, description="Hypergraph semantic edges")
    latent: CanonicalLatent = Field(..., description="Canonical S^{d-1} latent representation")
    raw_digest: str = Field(..., description="SHA-256 digest of original raw payload")
    text: Optional[str] = Field(default=None, description="Original raw text content (stored in payload)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary envelope metadata")
    timestamp_ns: int = Field(default_factory=lambda: int(time.time() * 1e9))


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to ingest into SLAP protocol")
    edges: Optional[List[HyperEdge]] = Field(default=None, description="Optional custom hyperedges")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom document metadata")
    epistemic_variance: Optional[float] = Field(default=0.01, description="Assigned epistemic covariance")


class IngestResponse(BaseModel):
    envelope: SlapEnvelope
    compression_stats: Dict[str, Any]
    token_savings: Dict[str, Any]


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    drift_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Cosine drift gating threshold (0.0 = permissive, all matches returned)")
    query_entities: Optional[List[str]] = Field(default=None, description="Optional explicit entity filters")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum matches to return")
    max_epistemic_variance: float = Field(default=1.0, description="Maximum allowable epistemic variance (1.0 = permissive)")



class RetrievedMatch(BaseModel):
    id: str
    text: str
    cosine_similarity: float
    epistemic_variance: float
    edges: List[HyperEdge]
    metadata: Dict[str, Any]
    passed_drift_gate: bool
    guard_verdict: str
    reconstructed_norm: float
    snr_db: Optional[float] = None


class RetrieveResponse(BaseModel):
    matches: List[RetrievedMatch]
    total_candidates: int
    topological_pruned: int
    execution_time_ms: float
    token_savings: Dict[str, Any]
    query_entities: List[str]


class GuardInspectRequest(BaseModel):
    vector: List[float] = Field(..., description="Latent vector to inspect")
    epistemic_variance: float = Field(default=0.01)
    drift_threshold: float = Field(default=0.82)
    max_variance: float = Field(default=0.05)


class GuardVerdict(BaseModel):
    passed: bool
    reason: str
    l2_norm: float
    norm_error: float
    spectral_energy: float
    is_spectral_anomaly: bool
    epistemic_variance: float
    lipschitz_ratio: float


def extract_hypergraph_entities(text: str) -> List[HyperEdge]:
    """Lightweight deterministic semantic hypergraph edge extractor.
    Extracts entities, predicates, and relations from domain text.
    """
    edges: List[HyperEdge] = []
    
    # Common relation patterns in systems/domain text
    relation_patterns = [
        (r"([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)\s+(requires|needs|depends on|authenticates with)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)", "requires"),
        (r"([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)\s+(issues|generates|produces|yields)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)", "issues"),
        (r"([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)\s+(validates|verifies|guards|filters)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)", "validates"),
        (r"([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)\s+(connects to|queries|indexes|stores)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)", "indexes"),
        (r"([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)\s+(projects|maps|compresses)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)", "projects"),
        (r"([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)\s+(injects|bypasses|caches)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?)", "injects"),
    ]

    for pattern, default_rel in relation_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            src, rel, tgt = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            if len(src) > 1 and len(tgt) > 1 and src.lower() != tgt.lower():
                edges.append(
                    HyperEdge(
                        source=src.title(),
                        relation=rel.lower(),
                        target=tgt.title(),
                        properties={"extracted_pattern": default_rel},
                        confidence=0.92,
                    )
                )

    # Key term extractor for standalone entity connectivity
    tech_keywords = [
        "Keycloak", "JWT", "Bearer Token", "OAuth2", "Authentication", "Authorization",
        "Qdrant", "Vector Database", "Hypersphere", "S^{d-1}", "Projection", "Quantization",
        "Residual Delta", "INT8", "INT4", "Latent WAF", "Lipschitz", "Cosine Drift",
        "KV Cache", "Soft Prompt", "Prefix Cache", "Attention", "FastAPI", "Next.js",
        "Epistemic Covariance", "Continuous Latent", "LLM", "Embedding",
        # Web & Frontend Ecosystem
        "React.js", "React", "Vue.js", "Vue", "Nuxt.js", "Nuxt", "Redux", "Pinia",
        "JavaScript", "HTML5", "CSS3", "Tailwind CSS", "Bootstrap", "React Bootstrap",
        "Material UI", "REST APIs", "REST", "AWS", "Jest", "React Testing Library",
        "WordPress", "WooCommerce", "Oxygen Builder", "Webflow", "Git", "GitHub",
        "Sadaf Ansari", "Saifee Creations", "Hexcode Designs", "Qurfy", "Echo", "WisdomCircle",
    ]
    found_keywords = [kw for kw in tech_keywords if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE)]
    
    if len(found_keywords) >= 2:
        for i in range(len(found_keywords) - 1):
            edges.append(
                HyperEdge(
                    source=found_keywords[i],
                    relation="relates_to",
                    target=found_keywords[i + 1],
                    properties={"co_occurrence": "true"},
                    confidence=0.88,
                )
            )


    # Fallback default edge if none matched
    if not edges:
        words = [w for w in re.findall(r"\b[A-Za-z]{3,}\b", text) if w.lower() not in {"the", "and", "for", "with", "from", "that", "this"}]
        src = words[0].title() if words else "Document"
        tgt = words[-1].title() if len(words) > 1 else "Topic"
        edges.append(
            HyperEdge(
                source=src,
                relation="describes",
                target=tgt,
                properties={"heuristic": "fallback"},
                confidence=0.75,
            )
        )

    return edges


def compute_digest(text: str) -> str:
    """Compute SHA-256 digest of input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
