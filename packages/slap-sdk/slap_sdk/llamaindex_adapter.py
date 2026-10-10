"""LlamaIndex Custom Retriever Adapter for SLAP."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from slap_sdk.client import SlapClient


@dataclass
class TextNode:
    text: str
    id_: str
    metadata: Dict[str, Any]


@dataclass
class NodeWithScore:
    node: TextNode
    score: float


class SlapLlamaIndexRetriever:
    """LlamaIndex CustomRetriever implementation for continuous algebraic retrieval."""

    def __init__(
        self,
        client: Optional[SlapClient] = None,
        base_url: str = "https://slap-9m29.onrender.com",
        drift_threshold: float = 0.0,
        top_k: int = 5,
        max_epistemic_variance: float = 1.0,
    ):
        self.client = client or SlapClient(base_url=base_url)
        self.drift_threshold = drift_threshold
        self.top_k = top_k
        self.max_epistemic_variance = max_epistemic_variance

    def _retrieve(self, query_str_or_bundle: Any) -> List[NodeWithScore]:
        if hasattr(query_str_or_bundle, "query_str"):
            query_text = query_str_or_bundle.query_str
        else:
            query_text = str(query_str_or_bundle)

        resp = self.client.retrieve(
            query=query_text,
            drift_threshold=self.drift_threshold,
            top_k=self.top_k,
            max_epistemic_variance=self.max_epistemic_variance,
        )

        nodes_with_scores: List[NodeWithScore] = []
        for match in resp.matches:
            if not match.passed_drift_gate:
                continue

            metadata = {
                **match.metadata,
                "epistemic_variance": match.epistemic_variance,
                "reconstructed_norm": match.reconstructed_norm,
                "guard_verdict": match.guard_verdict,
                "hyper_edges": [e.model_dump() for e in match.edges],
            }

            node = TextNode(
                text=match.text,
                id_=match.id,
                metadata=metadata,
            )

            nodes_with_scores.append(
                NodeWithScore(
                    node=node,
                    score=match.cosine_similarity,
                )
            )

        return nodes_with_scores

    def retrieve(self, query_str_or_bundle: Any) -> List[NodeWithScore]:
        return self._retrieve(query_str_or_bundle)
