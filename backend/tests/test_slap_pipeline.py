"""Integration tests for SLAP core pipeline."""

import math
import numpy as np
import pytest

from slap.core.projection import HypersphereProjector
from slap.core.quantizer import ResidualQuantizer
from slap.core.schema import HyperEdge, extract_hypergraph_entities
from slap.store.graph_index import TopologicalGraphIndex
from slap.store.qdrant_store import QdrantSlapStore
from slap.bridge.attention_injector import ContinuousAttentionBridge


def test_hypersphere_orthonormal_projection():
    """Verify orthogonal projection preserves orthonormality and projects to S^{d-1}."""
    dim = 384
    projector = HypersphereProjector(dimension=dim, seed=42)

    # Verify W_ortho is orthonormal: W^T @ W == I
    wtw = projector.w_ortho.T @ projector.w_ortho
    identity = np.eye(dim, dtype=np.float32)
    assert np.allclose(wtw, identity, atol=1e-5), "W_ortho must be strictly orthonormal"

    # Embed and project
    text = "Keycloak authenticates requests via signed JWT tokens"
    z_raw, z_norm = projector.embed_and_project(text)

    assert z_raw.shape == (dim,)
    assert z_norm.shape == (dim,)

    # Verify S^{d-1} unit hypersphere norm
    norm = np.linalg.norm(z_norm)
    assert abs(norm - 1.0) < 1e-4, f"Projected vector must lie on S^{d-1}, got norm {norm}"


def test_quantized_residual_compression_snr():
    """Verify INT8 dynamic quantization yields SNR > 40 dB and accurate reconstruction."""
    dim = 384
    projector = HypersphereProjector(dimension=dim, seed=42)
    quantizer = ResidualQuantizer()

    z_raw, z_norm = projector.embed_and_project("Continuous latent algebra compresses context windows")
    residual = quantizer.compute_residual(z_raw, z_norm)

    q_bytes, scale, zero_point, stats = quantizer.quantize_int8(residual)

    assert len(q_bytes) == dim, "INT8 residual must be exactly 1 byte per dimension"
    assert stats["snr_db"] > 40.0, f"Reconstruction SNR should be > 40 dB, got {stats['snr_db']} dB"
    assert stats["compression_ratio"] >= 3.8, "Raw float32 to INT8 should achieve ~4x compression"

    # Dequantize and reconstruct full vector
    r_recon = quantizer.dequantize_int8(q_bytes, scale, zero_point)
    z_recon = quantizer.reconstruct_full_vector(z_norm, r_recon)

    mse = np.mean((z_raw - z_recon) ** 2)
    assert mse < 1e-4, f"Reconstruction MSE should be < 1e-4, got {mse}"


def test_topological_graph_index_pruning():
    """Verify in-memory hypergraph index connects entities and prunes candidates."""
    graph = TopologicalGraphIndex()

    edges_doc1 = [
        HyperEdge(source="Keycloak", relation="issues", target="JWT"),
        HyperEdge(source="JWT", relation="authorizes", target="Service"),
    ]
    edges_doc2 = [
        HyperEdge(source="Qdrant", relation="stores", target="Vector"),
        HyperEdge(source="Vector", relation="indexed_by", target="Cosine"),
    ]

    graph.add_envelope("doc_auth_01", edges_doc1)
    graph.add_envelope("doc_vector_02", edges_doc2)

    # Pruning with query matching doc1 entity
    candidates_auth = graph.prune_candidates(["Keycloak"])
    assert "doc_auth_01" in candidates_auth
    assert "doc_vector_02" not in candidates_auth

    # Pruning with 1-hop neighbor
    candidates_hop = graph.prune_candidates(["Service"])
    assert "doc_auth_01" in candidates_hop

    # Empty query should return all
    assert len(graph.prune_candidates([])) == 2


def test_attention_injector_virtual_tokens_and_savings():
    """Verify continuous attention injector projects to 4096 and calculates token savings."""
    bridge = ContinuousAttentionBridge(latent_dim=384, llm_hidden_dim=4096)
    dummy_latent = np.random.randn(384).astype(np.float32)

    virtual_token = bridge.project_to_virtual_token(dummy_latent)
    assert virtual_token.shape == (4096,), "Virtual token must match LLM hidden dimension"

    kv_summary = bridge.synthesize_kv_prefix(virtual_token)
    assert kv_summary["status"] == "PREFIX_CACHE_READY"
    assert kv_summary["virtual_tokens_count"] == 1

    savings = bridge.compute_token_savings("A" * 1000, num_latents=1)
    assert savings["tokens_bypassed"] > 200
    assert savings["context_reduction_percent"] > 90.0
