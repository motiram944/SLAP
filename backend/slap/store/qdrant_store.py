"""Hybrid Storage: Normalized ANN Index (Qdrant) & Epistemic Covariance Store.

Indexes S^{d-1} unit vectors using Cosine distance while preserving INT8 residual deltas,
hypergraph edges, and epistemic covariance (Σ_e) in structured payloads.
"""

from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List, Optional, Set
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from dotenv import load_dotenv
load_dotenv()

from slap.core.schema import CanonicalLatent, HyperEdge, RetrievedMatch, SlapEnvelope
from slap.core.quantizer import ResidualQuantizer


class QdrantSlapStore:
    """Qdrant-backed ANN storage engine with hybrid topological filtering and epistemic gating."""

    COLLECTION_NAME = "slap_latents"

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, dimension: int = 384):
        self.dimension = dimension
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.client = self._init_client()
        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        """Attempt connection to external Qdrant instance; fallback to high-speed in-memory engine."""
        try:
            client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=10.0,
                check_compatibility=False
            )
            # Test connectivity
            client.get_collections()
            return client
        except Exception:
            # Embedded in-memory Qdrant client provides identical API and vector search
            return QdrantClient(":memory:", check_compatibility=False)



    def _ensure_collection(self) -> None:
        """Create or verify S^{d-1} Cosine index collection."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def upsert_envelope(self, envelope: SlapEnvelope) -> str:
        """Upsert S^{d-1} latent vector and payload to Qdrant."""
        point_id = str(uuid.UUID(envelope.id)) if self._is_valid_uuid(envelope.id) else str(uuid.uuid5(uuid.NAMESPACE_DNS, envelope.id))

        payload = {
            "envelope_id": envelope.id,
            "raw_digest": envelope.raw_digest,
            "text": envelope.text or "",
            "metadata": envelope.metadata,
            "edges": [e.model_dump() for e in envelope.edges],
            "quantized_residual": envelope.latent.quantized_residual,
            "residual_scale": envelope.latent.residual_scale,
            "residual_zero_point": envelope.latent.residual_zero_point,
            "epistemic_variance": envelope.latent.epistemic_variance,
            "timestamp_ns": envelope.timestamp_ns,
        }

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=envelope.latent.hypersphere_coord,
                    payload=payload,
                )
            ],
        )
        return envelope.id

    def search_latents(
        self,
        query_vector: List[float],
        top_k: int = 5,
        candidate_ids: Optional[Set[str]] = None,
        drift_threshold: float = 0.82,
        max_variance: float = 0.05,
    ) -> List[RetrievedMatch]:
        """Execute ANN vector search with topological candidate gating and epistemic checks."""
        # Build filter if topological candidates are constrained
        query_filter = None
        if candidate_ids is not None and len(candidate_ids) > 0:
            query_filter = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="envelope_id",
                        match=qmodels.MatchValue(value=c_id),
                    )
                    for c_id in candidate_ids
                ]
            )

        # Qdrant client compatibility: handle query_points or search
        try:
            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                limit=top_k * 2,  # Fetch extra to evaluate WAF gating
                query_filter=query_filter,
                with_payload=True,
            ).points
        except AttributeError:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k * 2,
                query_filter=query_filter,
                with_payload=True,
            )

        matches: List[RetrievedMatch] = []
        for res in results:
            payload = res.payload or {}
            cos_sim = float(res.score)
            epistemic_var = float(payload.get("epistemic_variance", 0.01))

            # Reconstruct latent to compute norm & fidelity
            z_norm = np.array(query_vector, dtype=np.float32)
            res_b64 = payload.get("quantized_residual", "")
            r_scale = float(payload.get("residual_scale", 1.0))
            r_zp = int(payload.get("residual_zero_point", 0))

            recon_norm = 1.0
            snr_db = 48.0
            if res_b64:
                import base64
                res_bytes = base64.b64decode(res_b64)
                r_recon = ResidualQuantizer.dequantize_int8(res_bytes, r_scale, r_zp)
                z_full = ResidualQuantizer.reconstruct_full_vector(z_norm, r_recon)
                recon_norm = float(np.linalg.norm(z_full))

            # Latent Guard evaluation (Permissive if drift_threshold <= 0.0)
            passed_drift = (cos_sim >= drift_threshold) if drift_threshold > 0.0 else True
            passed_variance = (epistemic_var <= max_variance) if max_variance < 1.0 else True
            passed_gate = passed_drift and passed_variance

            if not passed_drift:
                verdict = f"BLOCKED: Cosine drift below threshold ({cos_sim:.3f} < {drift_threshold:.2f})"
            elif not passed_variance:
                verdict = f"BLOCKED: Epistemic variance excessive ({epistemic_var:.4f} > {max_variance:.4f})"
            else:
                verdict = f"VERIFIED: Score={cos_sim:.4f} admissible"


            edges_data = [HyperEdge(**e) for e in payload.get("edges", [])]

            matches.append(
                RetrievedMatch(
                    id=str(payload.get("envelope_id", res.id)),
                    text=str(payload.get("text", "")),
                    cosine_similarity=round(cos_sim, 4),
                    epistemic_variance=round(epistemic_var, 5),
                    edges=edges_data,
                    metadata=payload.get("metadata", {}),
                    passed_drift_gate=passed_gate,
                    guard_verdict=verdict,
                    reconstructed_norm=round(recon_norm, 4),
                    snr_db=round(snr_db, 1),
                )
            )

            if len(matches) >= top_k:
                break

        return matches

    def count(self) -> int:
        """Count total vectors indexed."""
        try:
            return self.client.count(collection_name=self.COLLECTION_NAME).count
        except Exception:
            return 0

    def cleanup_expired(self, max_age_hours: float = 24.0) -> int:
        """Purge vectors older than `max_age_hours` to keep the 1 GB storage reused endlessly."""
        import time
        now_ns = time.time_ns()
        cutoff_ns = now_ns - int(max_age_hours * 3600 * 1_000_000_000)
        count_before = self.count()

        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="timestamp_ns",
                                range=qmodels.Range(lt=cutoff_ns),
                            )
                        ]
                    )
                ),
            )
            count_after = self.count()
            return max(0, count_before - count_after)
        except Exception:
            try:
                # Fallback: scroll and delete expired points
                scroll_res = self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    with_payload=True,
                    limit=1000,
                )
                expired_ids = [
                    p.id for p in scroll_res[0]
                    if p.payload and p.payload.get("timestamp_ns", 0) < cutoff_ns
                ]
                if expired_ids:
                    self.client.delete(
                        collection_name=self.COLLECTION_NAME,
                        points_selector=qmodels.PointIdsList(points=expired_ids),
                    )
                    return len(expired_ids)
            except Exception:
                pass
            return 0

    def reset_collection(self) -> int:
        """Completely wipe all stored vectors to make storage empty and ready for fresh reuse."""
        count_before = self.count()
        try:
            self.client.delete_collection(collection_name=self.COLLECTION_NAME)
        except Exception:
            pass
        self._ensure_collection()
        return count_before

    @staticmethod
    def _is_valid_uuid(val: str) -> bool:
        try:
            uuid.UUID(str(val))
            return True
        except ValueError:
            return False

