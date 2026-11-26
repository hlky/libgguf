"""Bounded, deterministic calibration recipes for native weight encoders.

Inputs are caller-owned F32 activations [sample,K] and weights [N,K]. This
module does not collect model activations or select evaluation datasets.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from .native_formats import NATIVE_QUANT_SIZES, NativeQuantizationType, native_row_size
from .native_numpy import _fit, _half, _payload, dequantize_rows, quantize_rows


@dataclass(frozen=True)
class CalibrationRecipe:
    """Encoder identity and explicit bounded calibration sample selection."""

    sample_indices: tuple[int, ...]
    dataset_digest: str
    mode: str = "diagonal"
    revision: int = 1
    scratch_bytes: int = 64 * 1024 * 1024
    sweeps: int = 4

    def validate(self, rows: int):
        if self.revision != 1 or self.mode not in {"diagonal", "output_error"}:
            raise ValueError("unsupported calibration recipe")
        if not self.dataset_digest or not 1 <= len(self.sample_indices) <= 4096:
            raise ValueError("calibration requires a dataset identity and 1..4096 explicit samples")
        if len(set(self.sample_indices)) != len(self.sample_indices) or any(
            type(i) is not int or i < 0 or i >= rows for i in self.sample_indices
        ):
            raise ValueError("calibration sample indices must be unique and in range")
        if not 1 <= self.sweeps <= 4 or not 0 < self.scratch_bytes <= 64 * 1024 * 1024:
            raise ValueError("calibration exceeds the bounded recipe")


def diagonal_importance(activations, recipe: CalibrationRecipe):
    """Accumulate ordered binary64 second moments and return normalized F32 importance."""
    x = np.asarray(activations)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("calibration activations must be a rank-two F32 array")
    recipe.validate(x.shape[0])
    if x.shape[1] * 16 > recipe.scratch_bytes:
        raise ValueError("scratch budget cannot hold diagonal accumulation")
    sums = np.zeros(x.shape[1], dtype=np.float64)
    digest = hashlib.sha256()
    for sample in recipe.sample_indices:
        row = x[sample]
        if not np.isfinite(row).all():
            raise ValueError("calibration activations must be finite")
        digest.update(struct.pack("<Q", sample))
        digest.update(row.astype("<f4", copy=False).tobytes())
        for k in range(x.shape[1]):
            value = float(row[k])
            sums[k] += value * value
    maximum = float(sums.max())
    if maximum:
        sums /= maximum
    return sums.astype(np.float32), {
        "encoder": "clip_grid" if recipe.mode == "diagonal" else "blocked_output_error",
        "encoder_revision": recipe.revision,
        "mode": recipe.mode,
        "dataset_digest": recipe.dataset_digest,
        "activation_digest": digest.hexdigest(),
        "sample_indices": list(recipe.sample_indices),
        "sample_count": len(recipe.sample_indices),
        "accumulation": "ordered_f64",
        "scratch_bytes": recipe.scratch_bytes,
        "sweeps": recipe.sweeps if recipe.mode == "output_error" else 0,
    }


def _proposals(values, format, importance):
    """The 17 canonical stored-scale proposals, in fixed clipping order."""
    if format != NativeQuantizationType.S4H256:
        eight = format == NativeQuantizationType.S8G32
        lo, hi = (-127, 127) if eight else (-8, 7)
        initial = max(float(v) / (hi if v >= 0 else lo) for v in values)
        for j in range(17):
            bits, scale = _half(initial * (16 + j) / 32)
            q = [max(lo, min(hi, round(float(v) / scale))) if scale else 0 for v in values]
            yield bits + _payload(q, eight)
    else:
        local = [_fit(values[g:g+32], importance[g:g+32], -8, 7)[1] for g in range(0, 256, 32)]
        for j in range(17):
            bits, base = _half(max(local) / 255 * (16 + j) / 32)
            codes = [max(1, min(255, round(s / base))) if s else 0 for s in local]
            q = []
            for k, v in enumerate(values):
                scale = float(np.float32(base) * np.float32(codes[k // 32]))
                q.append(max(-8, min(7, round(float(v) / scale))) if scale else 0)
            yield bits + bytes(codes) + _payload(q)


def _squared_error(values):
    total = 0.0
    for value in values.flat:
        total += float(value) * float(value)
    return total


def quantize_calibrated(weights, format, activations, recipe: CalibrationRecipe):
    """Return canonical bytes and provenance; output-error accepts only strict improvements.

    The output is not scratch. Caller-owned inputs are streamed one K column at
    a time; residuals cover at most 16 outputs and the explicitly selected rows.
    No BLAS reduction changes the ordered binary64 objective.
    """
    if not isinstance(weights, np.ndarray) or weights.dtype != np.float32 or weights.ndim != 2 or not weights.flags.c_contiguous or weights.shape[0] == 0:
        raise ValueError("bounded calibration requires caller-owned contiguous rank-two F32 weights")
    format = NativeQuantizationType(format)
    row_bytes = native_row_size(format, weights.shape[1])
    x = np.asarray(activations)
    if x.ndim != 2 or x.shape[1] != weights.shape[1]:
        raise ValueError("calibration K differs from weights")
    importance, provenance = diagonal_importance(x, recipe)
    samples = len(recipe.sample_indices)
    block_values, block_bytes, _ = NATIVE_QUANT_SIZES[format]
    # Residual + trial + one candidate difference + streamed activation column,
    # decoded output tile, diagonal and proposal temporaries. Reject before fitting.
    required = (samples * (16 + 3) * 8 if recipe.mode == "output_error" else 0) + min(16, len(weights)) * weights.shape[1] * 4 + weights.shape[1] * 16 + block_values * 64
    if required > recipe.scratch_bytes:
        raise ValueError("scratch budget cannot hold the fixed output-error tile")
    encoded = np.empty((len(weights), row_bytes), dtype=np.uint8)
    for first in range(0, len(weights), 16):
        encoded[first:first+16] = quantize_rows(weights[first:first+16], format, importance)
    if recipe.mode == "diagonal":
        return encoded, provenance
    indices = np.asarray(recipe.sample_indices, dtype=np.intp)
    accepted = 0
    for first in range(0, len(weights), 16):
        count = min(16, len(weights) - first)
        decoded = dequantize_rows(encoded[first:first+count], format, weights.shape[1])
        residual = np.zeros((samples, count), dtype=np.float64)
        for k in range(weights.shape[1]):
            column = x[indices, k].astype(np.float64)
            for n in range(count):
                residual[:, n] += column * (float(weights[first+n, k]) - float(decoded[n, k]))
        for _ in range(recipe.sweeps):
            improved = False
            for start in range(0, weights.shape[1], block_values):
                offset = start // block_values * block_bytes
                for n in range(count):
                    score = _squared_error(residual[:, n])
                    for proposal in _proposals(weights[first+n, start:start+block_values], format, importance[start:start+block_values]):
                        candidate = dequantize_rows(np.frombuffer(proposal, dtype=np.uint8).reshape(1, -1), format, block_values)[0]
                        trial = residual[:, n].copy()
                        for k in range(block_values):
                            trial += x[indices, start+k].astype(np.float64) * (float(decoded[n, start+k]) - float(candidate[k]))
                        candidate_score = _squared_error(trial)
                        if candidate_score < score:
                            residual[:, n] = trial
                            decoded[n, start:start+block_values] = candidate
                            encoded[first+n, offset:offset+block_bytes] = np.frombuffer(proposal, dtype=np.uint8)
                            score = candidate_score
                            accepted += 1
                            improved = True
            if not improved:
                break
    provenance["accepted_updates"] = accepted
    return encoded, provenance
