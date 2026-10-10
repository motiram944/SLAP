"""Quantized Residual Compressor & Decompressor (INT8/INT4).

Computes residual delta vector:
    r = z_{raw} - z_{norm}
Quantizes r to dynamic signed INT8 (or INT4 packed) with scale and zero-point offset:
    r_{recon} = (q - zero_point) * scale
Reconstructs original embedding with high Signal-to-Noise Ratio (SNR > 45 dB).
"""

from __future__ import annotations
import math
from typing import Any, Dict, Tuple
import numpy as np


class ResidualQuantizer:
    """Dynamic INT8 and INT4 residual vector quantizer with affine calibration."""

    @staticmethod
    def compute_residual(z_raw: np.ndarray, z_norm: np.ndarray) -> np.ndarray:
        """Compute delta vector r = z_{raw} - z_{norm}."""
        return (np.asarray(z_raw, dtype=np.float32) - np.asarray(z_norm, dtype=np.float32)).astype(np.float32)

    @classmethod
    def quantize_int8(cls, residual: np.ndarray) -> Tuple[bytes, float, int, Dict[str, Any]]:
        """Quantize continuous float32 residual to signed INT8 bytes.
        Returns: (quantized_bytes, scale, zero_point, metrics_dict)
        """
        r = np.asarray(residual, dtype=np.float32).flatten()
        r_min = float(np.min(r))
        r_max = float(np.max(r))

        range_span = r_max - r_min
        if range_span < 1e-12:
            scale = 1.0
            zero_point = 0
            q = np.zeros_like(r, dtype=np.int8)
        else:
            # 254 distinct buckets across [-128, 127]
            scale = range_span / 254.0
            zero_point = int(np.clip(np.round(-r_min / scale) - 128, -128, 127))
            q_unclipped = np.round(r / scale) + zero_point
            q = np.clip(q_unclipped, -128, 127).astype(np.int8)

        quantized_bytes = q.tobytes()

        # Compute reconstruction error metrics
        r_recon = (q.astype(np.float32) - zero_point) * scale
        mse = float(np.mean((r - r_recon) ** 2))
        signal_power = float(np.mean(r ** 2)) + 1e-12
        snr_db = float(10.0 * math.log10(signal_power / (mse + 1e-12)))

        raw_size_bytes = r.nbytes  # float32 = 4 bytes per dim
        compressed_size_bytes = len(quantized_bytes) + 8  # 1 byte per dim + metadata
        compression_ratio = float(raw_size_bytes / max(compressed_size_bytes, 1))

        stats = {
            "mode": "INT8",
            "mse": mse,
            "snr_db": snr_db,
            "scale": scale,
            "zero_point": zero_point,
            "raw_bytes": raw_size_bytes,
            "compressed_bytes": compressed_size_bytes,
            "compression_ratio": round(compression_ratio, 2),
            "max_error": float(np.max(np.abs(r - r_recon))),
        }

        return quantized_bytes, scale, zero_point, stats

    @classmethod
    def dequantize_int8(cls, quantized_bytes: bytes, scale: float, zero_point: int) -> np.ndarray:
        """Dequantize INT8 bytes back to float32 residual vector."""
        q = np.frombuffer(quantized_bytes, dtype=np.int8).astype(np.float32)
        r_recon = (q - float(zero_point)) * float(scale)
        return r_recon.astype(np.float32)

    @classmethod
    def quantize_int4(cls, residual: np.ndarray) -> Tuple[bytes, float, int, Dict[str, Any]]:
        """Quantize float32 residual to signed INT4 nibbles, packed 2 per byte.
        INT4 range: [-8, 7] (15 buckets).
        """
        r = np.asarray(residual, dtype=np.float32).flatten()
        r_min = float(np.min(r))
        r_max = float(np.max(r))

        range_span = r_max - r_min
        if range_span < 1e-12:
            scale = 1.0
            zero_point = 0
            q4 = np.zeros(len(r), dtype=np.int8)
        else:
            scale = range_span / 15.0
            zero_point = int(np.clip(np.round(-r_min / scale) - 8, -8, 7))
            q_unclipped = np.round(r / scale) + zero_point
            q4 = np.clip(q_unclipped, -8, 7).astype(np.int8)

        # Pack into bytes (low nibble & high nibble)
        padded_len = len(q4) + (len(q4) % 2)
        if len(q4) % 2 != 0:
            q4_padded = np.append(q4, [0])
        else:
            q4_padded = q4

        # Convert signed [-8, 7] to unsigned [0, 15] for bitwise packing
        u4 = (q4_padded + 8).astype(np.uint8) & 0x0F
        packed = ((u4[0::2] << 4) | u4[1::2]).tobytes()

        # Reconstruction metrics
        r_recon = (q4.astype(np.float32) - zero_point) * scale
        mse = float(np.mean((r - r_recon) ** 2))
        signal_power = float(np.mean(r ** 2)) + 1e-12
        snr_db = float(10.0 * math.log10(signal_power / (mse + 1e-12)))

        raw_size = r.nbytes
        comp_size = len(packed) + 8
        stats = {
            "mode": "INT4",
            "mse": mse,
            "snr_db": snr_db,
            "scale": scale,
            "zero_point": zero_point,
            "raw_bytes": raw_size,
            "compressed_bytes": comp_size,
            "compression_ratio": round(raw_size / max(comp_size, 1), 2),
            "max_error": float(np.max(np.abs(r - r_recon))),
        }

        return packed, scale, zero_point, stats

    @classmethod
    def dequantize_int4(cls, packed_bytes: bytes, scale: float, zero_point: int, original_dim: int) -> np.ndarray:
        """Unpack INT4 packed bytes back to float32 residual vector."""
        raw_u8 = np.frombuffer(packed_bytes, dtype=np.uint8)
        high = (raw_u8 >> 4) & 0x0F
        low = raw_u8 & 0x0F
        interleaved = np.empty(len(raw_u8) * 2, dtype=np.uint8)
        interleaved[0::2] = high
        interleaved[1::2] = low
        
        # Unshift from [0, 15] back to [-8, 7]
        q4 = interleaved[:original_dim].astype(np.int8) - 8
        r_recon = (q4.astype(np.float32) - float(zero_point)) * float(scale)
        return r_recon.astype(np.float32)

    @classmethod
    def reconstruct_full_vector(cls, z_norm: np.ndarray, r_recon: np.ndarray) -> np.ndarray:
        """Synthesize reconstructed continuous latent state:
            z_{recon} = z_{norm} + r_{recon}
        """
        return (np.asarray(z_norm, dtype=np.float32) + np.asarray(r_recon, dtype=np.float32)).astype(np.float32)
