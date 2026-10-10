"""Latent Web Application Firewall (WAF) & Security Gateway.

Enforces algebraic integrity, S^{d-1} manifold boundaries, spectral anomaly detection,
anti-poisoning Lipschitz bounds, and epistemic drift gating.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from slap.core.schema import GuardVerdict


class LatentWAF:
    """Security Gateway validating latent algebraic vectors before storage or attention injection."""

    def __init__(
        self,
        norm_tolerance: float = 1e-4,
        default_drift_threshold: float = 0.82,
        default_max_variance: float = 0.05,
        max_lipschitz_ratio: float = 4.5,
        spectral_anomaly_threshold: float = 0.65,
    ):
        self.norm_tolerance = norm_tolerance
        self.default_drift_threshold = default_drift_threshold
        self.default_max_variance = default_max_variance
        self.max_lipschitz_ratio = max_lipschitz_ratio
        self.spectral_anomaly_threshold = spectral_anomaly_threshold

    def inspect_vector(
        self,
        vector: Union[List[float], np.ndarray],
        epistemic_variance: float = 0.01,
        cosine_similarity: Optional[float] = None,
        drift_threshold: Optional[float] = None,
        max_variance: Optional[float] = None,
    ) -> GuardVerdict:
        """Execute full suite of mathematical safety checks on a latent representation."""
        vec = np.asarray(vector, dtype=np.float64).flatten()
        threshold_tau = drift_threshold if drift_threshold is not None else self.default_drift_threshold
        max_var = max_variance if max_variance is not None else self.default_max_variance

        # 1. Finite Value Check (NaN / Inf protection)
        if not np.all(np.isfinite(vec)):
            return GuardVerdict(
                passed=False,
                reason="BLOCKED: Latent tensor contains NaN or Infinite floating-point values",
                l2_norm=0.0,
                norm_error=1.0,
                spectral_energy=0.0,
                is_spectral_anomaly=True,
                epistemic_variance=epistemic_variance,
                lipschitz_ratio=float("inf"),
            )

        # 2. S^{d-1} Hypersphere Norm Check
        l2_norm = float(np.linalg.norm(vec))
        norm_error = abs(l2_norm - 1.0)
        if norm_error > self.norm_tolerance:
            return GuardVerdict(
                passed=False,
                reason=f"BLOCKED: Manifold boundary violation. L2 norm {l2_norm:.6f} deviates from 1.0 (error {norm_error:.2e} > {self.norm_tolerance:.2e})",
                l2_norm=round(l2_norm, 6),
                norm_error=round(norm_error, 6),
                spectral_energy=0.0,
                is_spectral_anomaly=False,
                epistemic_variance=epistemic_variance,
                lipschitz_ratio=0.0,
            )

        # 3. Cosine Drift Gate (if cosine similarity provided from retrieval)
        if cosine_similarity is not None and cosine_similarity < threshold_tau:
            return GuardVerdict(
                passed=False,
                reason=f"BLOCKED: Semantic cosine drift. Similarity {cosine_similarity:.3f} is below gate threshold {threshold_tau:.2f}",
                l2_norm=round(l2_norm, 6),
                norm_error=round(norm_error, 6),
                spectral_energy=0.0,
                is_spectral_anomaly=False,
                epistemic_variance=round(epistemic_variance, 5),
                lipschitz_ratio=1.0,
            )

        # 4. Epistemic Covariance / Variance Gate
        if epistemic_variance > max_var:
            return GuardVerdict(
                passed=False,
                reason=f"BLOCKED: Model epistemic variance {epistemic_variance:.4f} exceeds safety threshold {max_var:.4f}",
                l2_norm=round(l2_norm, 6),
                norm_error=round(norm_error, 6),
                spectral_energy=0.0,
                is_spectral_anomaly=False,
                epistemic_variance=round(epistemic_variance, 5),
                lipschitz_ratio=1.0,
            )

        # 5. High-Frequency Spectral Anomaly Detection (Anti-Poisoning FFT)
        fft_vals = np.fft.rfft(vec)
        power_spectrum = np.abs(fft_vals) ** 2
        total_energy = float(np.sum(power_spectrum)) + 1e-12
        cutoff = len(power_spectrum) // 2
        high_freq_energy = float(np.sum(power_spectrum[cutoff:]))
        high_freq_ratio = high_freq_energy / total_energy

        is_spectral_anomaly = high_freq_ratio > self.spectral_anomaly_threshold
        if is_spectral_anomaly:
            return GuardVerdict(
                passed=False,
                reason=f"BLOCKED: Adversarial spectral anomaly detected. High-frequency energy ratio {high_freq_ratio:.3f} > {self.spectral_anomaly_threshold:.2f}",
                l2_norm=round(l2_norm, 6),
                norm_error=round(norm_error, 6),
                spectral_energy=round(high_freq_ratio, 4),
                is_spectral_anomaly=True,
                epistemic_variance=epistemic_variance,
                lipschitz_ratio=1.0,
            )

        # 6. Anti-Poisoning Lipschitz Bound Check
        diff = np.diff(vec)
        local_gradient_norm = float(np.linalg.norm(diff))
        # For unit sphere S^{d-1}, gradient norm cannot exceed 1.85 without high-frequency adversarial jitter
        lipschitz_ratio = local_gradient_norm / 1.414
        if lipschitz_ratio > self.max_lipschitz_ratio:
            return GuardVerdict(
                passed=False,
                reason=f"BLOCKED: Anti-poisoning Lipschitz bound exceeded. Gradient ratio {lipschitz_ratio:.2f} > {self.max_lipschitz_ratio:.2f}",
                l2_norm=round(l2_norm, 6),
                norm_error=round(norm_error, 6),
                spectral_energy=round(high_freq_ratio, 4),
                is_spectral_anomaly=False,
                epistemic_variance=epistemic_variance,
                lipschitz_ratio=round(lipschitz_ratio, 3),
            )


        # All gates passed
        return GuardVerdict(
            passed=True,
            reason="PASSED: Manifold boundaries, spectral bounds, and epistemic covariance verified",
            l2_norm=round(l2_norm, 6),
            norm_error=round(norm_error, 6),
            spectral_energy=round(high_freq_ratio, 4),
            is_spectral_anomaly=False,
            epistemic_variance=round(epistemic_variance, 5),
            lipschitz_ratio=round(lipschitz_ratio, 3),
        )
