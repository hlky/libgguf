import numpy as np
import pytest

from libgguf.calibration import CalibrationRecipe, diagonal_importance, quantize_calibrated
from libgguf.native_formats import NativeQuantizationType as F
from libgguf.native_numpy import dequantize_rows


@pytest.mark.parametrize("format", list(F))
def test_output_error_is_deterministic_and_cannot_worsen_calibration_objective(format):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(9, 256)).astype(np.float32)
    x[:, 1::2] = x[:, ::2] * 0.75
    w = rng.normal(size=(2, 256)).astype(np.float32)
    diagonal = CalibrationRecipe(tuple(range(9)), "fixture-seed19")
    recipe = CalibrationRecipe(tuple(range(9)), "fixture-seed19", mode="output_error")
    initial, _ = quantize_calibrated(w, format, x, diagonal)
    actual, provenance = quantize_calibrated(w, format, x, recipe)
    repeated, repeated_provenance = quantize_calibrated(w, format, x, recipe)
    np.testing.assert_array_equal(actual, repeated)
    assert provenance == repeated_provenance
    def error(encoded):
        delta = w.astype(np.float64) - dequantize_rows(encoded, format, 256).astype(np.float64)
        residual = x.astype(np.float64) @ delta.T
        return np.square(residual).sum()
    assert error(actual) <= error(initial) + 1e-10


def test_diagonal_samples_and_bounds_are_explicit():
    x = np.array([[1, 2], [3, 4], [float("nan"), 0]], dtype=np.float32)
    importance, provenance = diagonal_importance(x, CalibrationRecipe((0, 1), "data"))
    np.testing.assert_array_equal(importance, np.array([0.5, 1], dtype=np.float32))
    assert provenance["sample_indices"] == [0, 1]
    for recipe in (CalibrationRecipe((0, 0), "data"), CalibrationRecipe((2,), "data"), CalibrationRecipe((0,), "data", scratch_bytes=1)):
        with pytest.raises(ValueError):
            diagonal_importance(x, recipe)


def test_output_error_rejects_insufficient_scratch():
    x = np.ones((2, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="scratch budget"):
        quantize_calibrated(x, F.S4G32, x, CalibrationRecipe((0, 1), "data", mode="output_error", scratch_bytes=4096))
