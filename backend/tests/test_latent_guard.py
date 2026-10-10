"""Adversarial injection, manifold boundary, and drift gating tests for Latent WAF."""

import numpy as np
import pytest

from slap.guard.latent_waf import LatentWAF


@pytest.fixture
def waf():
    return LatentWAF(
        norm_tolerance=1e-4,
        default_drift_threshold=0.82,
        default_max_variance=0.05,
        max_lipschitz_ratio=4.5,
        spectral_anomaly_threshold=0.65,
    )


def test_valid_latent_passes_waf(waf):
    """Valid unit vector on S^{d-1} with low epistemic variance passes all gates."""
    vec = np.random.randn(384)
    vec = vec / np.linalg.norm(vec)

    verdict = waf.inspect_vector(
        vector=vec,
        epistemic_variance=0.015,
        cosine_similarity=0.91,
    )

    assert verdict.passed is True
    assert "PASSED" in verdict.reason
    assert verdict.norm_error < 1e-4


def test_norm_deviation_rejected(waf):
    """Latent norm deviating from S^{d-1} by > 1e-4 must be rejected."""
    vec = np.random.randn(384)
    vec = (vec / np.linalg.norm(vec)) * 1.02  # 2% expansion beyond unit sphere

    verdict = waf.inspect_vector(vector=vec)

    assert verdict.passed is False
    assert "Manifold boundary violation" in verdict.reason
    assert verdict.norm_error > 1e-4


def test_nan_and_inf_rejected(waf):
    """NaN or Inf floating-point values must be immediately rejected."""
    vec = np.random.randn(384)
    vec[12] = np.nan

    verdict = waf.inspect_vector(vector=vec)
    assert verdict.passed is False
    assert "NaN or Infinite" in verdict.reason

    vec[12] = np.inf
    verdict_inf = waf.inspect_vector(vector=vec)
    assert verdict_inf.passed is False
    assert "NaN or Infinite" in verdict_inf.reason


def test_spectral_anomaly_attack_rejected(waf):
    """High-frequency adversarial perturbation (continuous injection) is rejected."""
    # Create high-frequency alternating noise vector
    high_freq_noise = np.array([(-1.0) ** i for i in range(384)], dtype=np.float64)
    high_freq_noise = high_freq_noise / np.linalg.norm(high_freq_noise)

    verdict = waf.inspect_vector(vector=high_freq_noise)
    assert verdict.passed is False
    assert (
        "adversarial spectral anomaly" in verdict.reason.lower()
        or "lipschitz bound exceeded" in verdict.reason.lower()
    )


def test_cosine_drift_gate_rejected(waf):
    """Retrieved candidate below drift threshold tau=0.82 must be blocked."""
    vec = np.random.randn(384)
    vec = vec / np.linalg.norm(vec)

    verdict = waf.inspect_vector(
        vector=vec,
        cosine_similarity=0.74,  # Below default 0.82
        drift_threshold=0.82,
    )

    assert verdict.passed is False
    assert "Semantic cosine drift" in verdict.reason


def test_excessive_epistemic_variance_rejected(waf):
    """Model epistemic uncertainty Sigma_e > 0.05 must trigger uncertainty block."""
    vec = np.random.randn(384)
    vec = vec / np.linalg.norm(vec)

    verdict = waf.inspect_vector(
        vector=vec,
        epistemic_variance=0.088,  # > 0.05
        max_variance=0.05,
    )

    assert verdict.passed is False
    assert "epistemic variance" in verdict.reason.lower()
