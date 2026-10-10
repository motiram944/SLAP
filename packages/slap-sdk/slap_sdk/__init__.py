"""slap-sdk: Semantic Latent-Algebraic Protocol (SLAP) Python Client Library."""

from slap_sdk.client import SlapClient
from slap_sdk.llamaindex_adapter import SlapLlamaIndexRetriever
from slap_sdk.models import (
    HyperEdge,
    CanonicalLatent,
    IngestRequest,
    IngestResponse,
    RetrievedMatch,
    RetrieveRequest,
    RetrieveResponse,
    GuardVerdict,
)

__version__ = "0.1.0"
__all__ = [
    "SlapClient",
    "SlapLlamaIndexRetriever",
    "HyperEdge",
    "CanonicalLatent",
    "IngestRequest",
    "IngestResponse",
    "RetrievedMatch",
    "RetrieveRequest",
    "RetrieveResponse",
    "GuardVerdict",
]
