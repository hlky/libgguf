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


atexit.register(_libgguf.quantize_free)

__all__ = [
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
