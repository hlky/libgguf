from __future__ import annotations

from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
GGML_TYPE_F32 = 0

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "B",
    GGUF_TYPE_INT8: "b",
    GGUF_TYPE_UINT16: "H",
    GGUF_TYPE_INT16: "h",
    GGUF_TYPE_UINT32: "I",
    GGUF_TYPE_INT32: "i",
    GGUF_TYPE_FLOAT32: "f",
    GGUF_TYPE_BOOL: "?",
    GGUF_TYPE_UINT64: "Q",
    GGUF_TYPE_INT64: "q",
    GGUF_TYPE_FLOAT64: "d",
}


def load_imatrix(path: str | Path) -> dict[str, np.ndarray]:
    """Load llama.cpp imatrix data as tensor-name to float32 importance vectors.

    Supports the current GGUF imatrix format and the older binary format. For
    GGUF imatrix files, llama.cpp stores squared activation sums and counts;
    this returns the same averaged vectors consumed by llama-quantize.
    """

    path = Path(path)
    with path.open("rb") as f:
        if f.read(4) == GGUF_MAGIC:
            return _load_gguf_imatrix(f)
        f.seek(0)
        return _load_legacy_imatrix(f)


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("unexpected end of imatrix file")
    return data


def _read_struct(f: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, _read_exact(f, struct.calcsize("<" + fmt)))


def _read_string(f: BinaryIO) -> str:
    (size,) = _read_struct(f, "Q")
    return _read_exact(f, int(size)).decode("utf-8")


def _skip_value(f: BinaryIO, value_type: int) -> None:
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is not None:
        f.seek(struct.calcsize("<" + fmt), 1)
        return
    if value_type == GGUF_TYPE_STRING:
        (size,) = _read_struct(f, "Q")
        f.seek(int(size), 1)
        return
    if value_type == GGUF_TYPE_ARRAY:
        item_type, count = _read_struct(f, "IQ")
        for _ in range(int(count)):
            _skip_value(f, int(item_type))
        return
    raise ValueError(f"unsupported GGUF metadata value type {value_type}")


def _load_legacy_imatrix(f: BinaryIO) -> dict[str, np.ndarray]:
    (n_entries,) = _read_struct(f, "i")
    if n_entries < 1:
        raise ValueError("imatrix file contains no entries")

    result: dict[str, np.ndarray] = {}
    for _ in range(n_entries):
        (name_len,) = _read_struct(f, "i")
        name = _read_exact(f, int(name_len)).decode("utf-8")
        ncall, nval = _read_struct(f, "ii")
        if nval < 1:
            raise ValueError(f"imatrix entry {name!r} contains no values")
        values = np.frombuffer(_read_exact(f, int(nval) * 4), dtype="<f4").astype(np.float32)
        if ncall > 0:
            values /= np.float32(ncall)
        result[name] = np.ascontiguousarray(values, dtype=np.float32)

    return result


def _load_gguf_imatrix(f: BinaryIO) -> dict[str, np.ndarray]:
    version, tensor_count, kv_count = _read_struct(f, "IQQ")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")

    alignment = GGUF_DEFAULT_ALIGNMENT
    for _ in range(int(kv_count)):
        key = _read_string(f)
        (value_type,) = _read_struct(f, "I")
        if key == "general.alignment" and value_type == GGUF_TYPE_UINT32:
            (alignment,) = _read_struct(f, "I")
        else:
            _skip_value(f, int(value_type))
    if alignment < 1:
        raise ValueError(f"invalid GGUF alignment {alignment}")

    tensors = []
    for _ in range(int(tensor_count)):
        name = _read_string(f)
        n_dims, = _read_struct(f, "I")
        dims = _read_struct(f, "Q" * int(n_dims))
        tensor_type, offset = _read_struct(f, "IQ")
        tensors.append((name, tuple(int(dim) for dim in dims), int(tensor_type), int(offset)))

    data_offset = f.tell()
    padding = data_offset % alignment
    if padding:
        data_offset += alignment - padding

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for name, dims, tensor_type, offset in tensors:
        if tensor_type != GGML_TYPE_F32:
            continue
        n_values = int(np.prod(dims, dtype=np.int64))
        f.seek(data_offset + offset)
        values = np.frombuffer(_read_exact(f, n_values * 4), dtype="<f4").astype(np.float32)
        if name.endswith(".in_sum2"):
            sums[name[: -len(".in_sum2")]] = values.reshape(tuple(reversed(dims))).reshape(-1)
        elif name.endswith(".counts"):
            counts[name[: -len(".counts")]] = values.reshape(-1)

    result: dict[str, np.ndarray] = {}
    for name, sum_values in sums.items():
        count_values = counts.get(name)
        if count_values is None:
            raise ValueError(f"missing counts tensor for imatrix entry {name!r}")
        if count_values.size < 1:
            raise ValueError(f"imatrix counts tensor for {name!r} contains no values")
        if sum_values.size % count_values.size != 0:
            raise ValueError(f"imatrix sums/counts shape mismatch for {name!r}")

        row_size = sum_values.size // count_values.size
        averaged = sum_values.copy()
        for i, count in enumerate(count_values):
            start = i * row_size
            stop = start + row_size
            if count > 0.0:
                averaged[start:stop] /= count
            else:
                averaged[start:stop] = 1.0
        result[name] = np.ascontiguousarray(averaged, dtype=np.float32)

    if not result:
        raise ValueError("imatrix file contains no usable entries")
    return result
