"""Native CPU backend, separate from the independent NumPy reference."""

import numpy as np

from . import _libgguf
from .native_formats import NativeQuantizationType, native_row_size
from .native_numpy import validate_rows


def quantize_rows(rows, format, importance=None):
    rows, format, importance = validate_rows(rows, format, importance)
    raw = _libgguf.native_quantize_rows_raw(int(format), *rows.shape, rows, importance)
    return np.frombuffer(raw, dtype=np.uint8).reshape(rows.shape[0], -1)


def dequantize_rows(encoded, format, columns):
    format = NativeQuantizationType(format)
    encoded = np.ascontiguousarray(encoded, dtype=np.uint8)
    if encoded.ndim != 2 or encoded.shape[0] <= 0 or encoded.shape[1] != native_row_size(format, columns):
        raise ValueError("encoded native row shape differs from its format")
    raw = _libgguf.native_dequantize_rows_raw(int(format), encoded.shape[0], columns, encoded)
    return np.frombuffer(raw, dtype=np.float32).reshape(encoded.shape[0], columns)
