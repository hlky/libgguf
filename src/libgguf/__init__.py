from __future__ import annotations

from typing import Any

import atexit

import numpy as np

from . import _libgguf
from .imatrix import load_imatrix
from ._metadata import (
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    LlamaFileType,
    QK_K,
    quant_shape_from_byte_shape,
    quant_shape_to_byte_shape,
)


def _qtype_value(qtype: int | Any) -> int:
    value = getattr(qtype, "value", qtype)
    return int(value)


def row_size(qtype: int | Any, n_per_row: int) -> int:
    qtype_value = _qtype_value(qtype)
    n_per_row_value = int(n_per_row)
    if n_per_row_value <= 0:
        return 0
    block_size = _BLOCK_SIZES.get(qtype_value)
    if block_size is not None and block_size > 1 and n_per_row_value % block_size != 0:
        return 0
    return int(_libgguf.row_size(qtype_value, n_per_row_value))


def type_size(qtype: int | Any) -> int:
    return int(_libgguf.type_size(_qtype_value(qtype)))


def type_name(qtype: int | Any) -> str:
    return str(_libgguf.type_name(_qtype_value(qtype)))


def quantize_requires_imatrix(qtype: int | Any) -> bool:
    return bool(_libgguf.quantize_requires_imatrix(_qtype_value(qtype)))


_BLOCK_SIZES: dict[int, int] = {int(qtype): block_size for qtype, (block_size, _) in GGML_QUANT_SIZES.items()}

_STORAGE_QTYPE_DTYPES: dict[int, np.dtype[Any]] = {
    0: np.dtype(np.float32),
    1: np.dtype(np.float16),
}

_STORAGE_QTYPES = frozenset((*_STORAGE_QTYPE_DTYPES, 30))
_SUPPORTED_ROW_QTYPES = frozenset(
    qtype for qtype, block_size in _BLOCK_SIZES.items() if row_size(qtype, block_size) > 0
)
_SUPPORTED_QUANTIZED_QTYPES = frozenset(
    qtype for qtype in _SUPPORTED_ROW_QTYPES if qtype not in _STORAGE_QTYPES
)


def _validated_row_size(qtype_value: int, n_per_row: int, operation: str) -> int:
    if n_per_row <= 0:
        raise ValueError("n_per_row must be positive")
    if operation == "dequantize" and qtype_value not in _SUPPORTED_QUANTIZED_QTYPES:
        raise ValueError("unsupported quantization type or row width")

    bytes_per_row = row_size(qtype_value, n_per_row)
    if bytes_per_row > 0:
        return bytes_per_row

    block_size = _BLOCK_SIZES.get(qtype_value)
    if block_size is not None and block_size > 1 and n_per_row % block_size != 0:
        raise ValueError(f"{operation} row width must be a multiple of this quantization type's block size ({block_size})")

    raise ValueError("unsupported quantization type or row width")


def _validate_imatrix(imatrix: Any, n_per_row: int) -> np.ndarray:
    quant_weights = np.ascontiguousarray(imatrix, dtype=np.float32)
    if quant_weights.ndim != 1:
        raise ValueError("imatrix must be a one-dimensional float32 array")
    if quant_weights.shape[0] != n_per_row:
        raise ValueError("imatrix length must match n_per_row")
    return quant_weights


def quantize_rows_raw(
    qtype: int | Any,
    src: Any,
    n_rows: int,
    n_per_row: int,
    imatrix: Any | None = None,
) -> bytes:
    qtype_value = _qtype_value(qtype)
    n_per_row_value = int(n_per_row)
    _validated_row_size(qtype_value, n_per_row_value, "quantize")
    return _libgguf.quantize_rows_raw(
        qtype_value,
        src,
        int(n_rows),
        n_per_row_value,
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
    qtype_value = _qtype_value(qtype)
    n_per_row_value = int(n_per_row)
    _validated_row_size(qtype_value, n_per_row_value, "quantize")
    return int(
        _libgguf.quantize_rows_into_raw(
            qtype_value,
            src,
            dst,
            int(n_rows),
            n_per_row_value,
            imatrix,
        )
    )


def dequantize_rows_raw(
    qtype: int | Any,
    src: Any,
    n_rows: int,
    n_per_row: int,
) -> bytes:
    qtype_value = _qtype_value(qtype)
    n_per_row_value = int(n_per_row)
    _validated_row_size(qtype_value, n_per_row_value, "dequantize")
    return _libgguf.dequantize_rows_raw(
        qtype_value,
        src,
        int(n_rows),
        n_per_row_value,
    )


def dequantize_rows_into_raw(
    qtype: int | Any,
    src: Any,
    dst: Any,
    n_rows: int,
    n_per_row: int,
) -> int:
    qtype_value = _qtype_value(qtype)
    n_per_row_value = int(n_per_row)
    _validated_row_size(qtype_value, n_per_row_value, "dequantize")
    return int(
        _libgguf.dequantize_rows_into_raw(
            qtype_value,
            src,
            dst,
            int(n_rows),
            n_per_row_value,
        )
    )


def quantize_rows(data: np.ndarray, qtype: int | Any, imatrix: Any | None = None) -> np.ndarray:
    qtype_value = _qtype_value(qtype)
    if qtype_value in _STORAGE_QTYPES:
        return store_rows(data, qtype_value)

    rows = np.ascontiguousarray(data, dtype=np.float32)
    if rows.ndim == 0:
        raise ValueError("Expected an array with at least one dimension")

    n_rows = int(np.prod(rows.shape[:-1], dtype=np.int64)) if rows.ndim > 1 else 1
    n_per_row = int(rows.shape[-1])

    if imatrix is not None:
        quant_weights = _validate_imatrix(imatrix, n_per_row)
    elif quantize_requires_imatrix(qtype_value):
        quant_weights = np.ascontiguousarray(
            np.sum((rows * rows).reshape((-1, n_per_row)), axis=0, dtype=np.float32),
            dtype=np.float32,
        )
    else:
        quant_weights = None

    bytes_per_row = _validated_row_size(qtype_value, n_per_row, "quantize")
    out = np.empty((*rows.shape[:-1], bytes_per_row), dtype=np.uint8)
    quantize_rows_into_raw(qtype_value, rows, out, n_rows, n_per_row, quant_weights)
    return out


def store_rows(data: np.ndarray, qtype: int | Any) -> np.ndarray:
    rows = np.ascontiguousarray(data, dtype=np.float32)
    if rows.ndim == 0:
        raise ValueError("Expected an array with at least one dimension")

    qtype_value = _qtype_value(qtype)
    if qtype_value not in _STORAGE_QTYPES:
        raise ValueError("store_rows only supports F32, F16, and BF16 storage types")

    n_rows = int(np.prod(rows.shape[:-1], dtype=np.int64)) if rows.ndim > 1 else 1
    n_per_row = int(rows.shape[-1])
    bytes_per_row = _validated_row_size(qtype_value, n_per_row, "store")
    out = np.empty((*rows.shape[:-1], bytes_per_row), dtype=np.uint8)
    quantize_rows_into_raw(qtype_value, rows, out, n_rows, n_per_row)

    dtype = _STORAGE_QTYPE_DTYPES.get(qtype_value)
    if dtype is None:
        return out
    return out.view(dtype).reshape(rows.shape)


def dequantize_rows(data: Any, qtype: int | Any, n_per_row: int | None = None) -> np.ndarray:
    rows = np.ascontiguousarray(data, dtype=np.uint8)
    if rows.ndim == 0:
        raise ValueError("Expected an array with at least one dimension")

    qtype_value = _qtype_value(qtype)
    bytes_per_row = int(rows.shape[-1])
    if n_per_row is None:
        if qtype_value not in _SUPPORTED_QUANTIZED_QTYPES:
            raise ValueError("unsupported quantization type")
        block_size = _BLOCK_SIZES[qtype_value]
        block_bytes = type_size(qtype_value)
        if bytes_per_row % block_bytes != 0:
            raise ValueError("encoded row width is not a multiple of the quantization block size")
        n_per_row = bytes_per_row // block_bytes * block_size
    else:
        n_per_row = int(n_per_row)

    if _validated_row_size(qtype_value, n_per_row, "dequantize") != bytes_per_row:
        raise ValueError("encoded row width does not match n_per_row for this quantization type")

    n_rows = int(np.prod(rows.shape[:-1], dtype=np.int64)) if rows.ndim > 1 else 1
    out = np.empty((*rows.shape[:-1], n_per_row), dtype=np.float32)
    dequantize_rows_into_raw(qtype_value, rows, out, n_rows, n_per_row)
    return out


atexit.register(_libgguf.quantize_free)

_CONVERSION_EXPORTS = frozenset(
    {
        "QuantResult",
        "convert_safetensors_to_gguf_native",
        "convert_to_gguf",
    }
)

_INSPECT_EXPORTS = frozenset(
    {
        "GGUFFile",
        "GGUFFormatError",
        "GGUFMetadataValue",
        "GGUFTensorInfo",
        "GGUFValidationIssue",
        "GGUFValidationResult",
        "inspect_gguf",
        "open_gguf",
        "read_gguf_header",
        "validate_gguf",
    }
)


def __getattr__(name: str) -> Any:
    if name in _CONVERSION_EXPORTS:
        from . import quantize

        return getattr(quantize, name)
    if name in _INSPECT_EXPORTS:
        from . import inspect

        return getattr(inspect, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "dequantize_rows",
    "dequantize_rows_into_raw",
    "dequantize_rows_raw",
    "quantize_requires_imatrix",
    "quantize_rows",
    "quantize_rows_into_raw",
    "quantize_rows_raw",
    "store_rows",
    "load_imatrix",
    "QuantResult",
    "convert_safetensors_to_gguf_native",
    "convert_to_gguf",
    "GGUFFile",
    "GGUFFormatError",
    "GGUFMetadataValue",
    "GGUFTensorInfo",
    "GGUFValidationIssue",
    "GGUFValidationResult",
    "inspect_gguf",
    "open_gguf",
    "read_gguf_header",
    "validate_gguf",
    "row_size",
    "type_name",
    "type_size",
    "GGMLQuantizationType",
    "LlamaFileType",
    "GGML_QUANT_SIZES",
    "QK_K",
    "quant_shape_to_byte_shape",
    "quant_shape_from_byte_shape",
]
