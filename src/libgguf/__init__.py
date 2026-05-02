from __future__ import annotations

from typing import Any

import atexit

import numpy as np

from . import _libgguf
from .imatrix import load_imatrix
from .quantize import QuantResult, convert_to_gguf


def _qtype_value(qtype: int | Any) -> int:
    value = getattr(qtype, "value", qtype)
    return int(value)


def row_size(qtype: int | Any, n_per_row: int) -> int:
    return int(_libgguf.row_size(_qtype_value(qtype), int(n_per_row)))


def type_size(qtype: int | Any) -> int:
    return int(_libgguf.type_size(_qtype_value(qtype)))


def type_name(qtype: int | Any) -> str:
    return str(_libgguf.type_name(_qtype_value(qtype)))


def quantize_requires_imatrix(qtype: int | Any) -> bool:
    return bool(_libgguf.quantize_requires_imatrix(_qtype_value(qtype)))


_BLOCK_SIZES: dict[int, int] = {
    2: 32,
    3: 32,
    6: 32,
    7: 32,
    8: 32,
    10: 256,
    11: 256,
    12: 256,
    13: 256,
    14: 256,
    16: 256,
    17: 256,
    18: 256,
    19: 256,
    20: 32,
    21: 256,
    22: 256,
    23: 256,
    29: 256,
    34: 256,
    35: 256,
    39: 32,
    40: 64,
    41: 128,
}


def quantize_rows_raw(
    qtype: int | Any,
    src: Any,
    n_rows: int,
    n_per_row: int,
    imatrix: Any | None = None,
) -> bytes:
    return _libgguf.quantize_rows_raw(
        _qtype_value(qtype),
        src,
        int(n_rows),
        int(n_per_row),
        imatrix,
    )


def quantize_rows_into_raw(
    qtype: int | Any,
    src: Any,
    dst: Any,
    n_rows: int,
    n_per_row: int,
    imatrix: Any | None = None,
) -> int:
    return int(
        _libgguf.quantize_rows_into_raw(
            _qtype_value(qtype),
            src,
            dst,
            int(n_rows),
            int(n_per_row),
            imatrix,
        )
    )


def dequantize_rows_raw(
    qtype: int | Any,
    src: Any,
    n_rows: int,
    n_per_row: int,
) -> bytes:
    return _libgguf.dequantize_rows_raw(
        _qtype_value(qtype),
        src,
        int(n_rows),
        int(n_per_row),
    )


def dequantize_rows_into_raw(
    qtype: int | Any,
    src: Any,
    dst: Any,
    n_rows: int,
    n_per_row: int,
) -> int:
    return int(
        _libgguf.dequantize_rows_into_raw(
            _qtype_value(qtype),
            src,
            dst,
            int(n_rows),
            int(n_per_row),
        )
    )


def quantize_rows(data: np.ndarray, qtype: int | Any, imatrix: Any | None = None) -> np.ndarray:
    rows = np.ascontiguousarray(data, dtype=np.float32)
    if rows.ndim == 0:
        raise ValueError("Expected an array with at least one dimension")

    qtype_value = _qtype_value(qtype)
    n_rows = int(np.prod(rows.shape[:-1], dtype=np.int64)) if rows.ndim > 1 else 1
    n_per_row = int(rows.shape[-1])

    if imatrix is not None:
        quant_weights = np.ascontiguousarray(imatrix, dtype=np.float32)
    elif quantize_requires_imatrix(qtype_value):
        quant_weights = np.ascontiguousarray(
            np.sum((rows * rows).reshape((-1, n_per_row)), axis=0, dtype=np.float32),
            dtype=np.float32,
        )
    else:
        quant_weights = None

    bytes_per_row = row_size(qtype_value, n_per_row)
    out = np.empty((*rows.shape[:-1], bytes_per_row), dtype=np.uint8)
    quantize_rows_into_raw(qtype_value, rows, out, n_rows, n_per_row, quant_weights)
    return out


def dequantize_rows(data: Any, qtype: int | Any, n_per_row: int | None = None) -> np.ndarray:
    rows = np.ascontiguousarray(data, dtype=np.uint8)
    if rows.ndim == 0:
        raise ValueError("Expected an array with at least one dimension")

    qtype_value = _qtype_value(qtype)
    bytes_per_row = int(rows.shape[-1])
    if n_per_row is None:
        block_size = _BLOCK_SIZES.get(qtype_value)
        if block_size is None:
            raise ValueError("unsupported quantization type")
        block_bytes = type_size(qtype_value)
        if bytes_per_row % block_bytes != 0:
            raise ValueError("encoded row width is not a multiple of the quantization block size")
        n_per_row = bytes_per_row // block_bytes * block_size
    else:
        n_per_row = int(n_per_row)

    if row_size(qtype_value, n_per_row) != bytes_per_row:
        raise ValueError("encoded row width does not match n_per_row for this quantization type")

    n_rows = int(np.prod(rows.shape[:-1], dtype=np.int64)) if rows.ndim > 1 else 1
    out = np.empty((*rows.shape[:-1], n_per_row), dtype=np.float32)
    dequantize_rows_into_raw(qtype_value, rows, out, n_rows, n_per_row)
    return out


atexit.register(_libgguf.quantize_free)

__all__ = [
    "dequantize_rows",
    "dequantize_rows_into_raw",
    "dequantize_rows_raw",
    "quantize_requires_imatrix",
    "quantize_rows",
    "quantize_rows_into_raw",
    "quantize_rows_raw",
    "load_imatrix",
    "QuantResult",
    "convert_to_gguf",
    "row_size",
    "type_name",
    "type_size",
]
