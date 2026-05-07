from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

import libgguf


GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def _write_legacy_imatrix(path: Path, entries: list[tuple[str, int, list[float]]]) -> None:
    data = bytearray(struct.pack("<i", len(entries)))
    for name, ncall, values in entries:
        name_bytes = name.encode("utf-8")
        data += struct.pack("<i", len(name_bytes))
        data += name_bytes
        data += struct.pack("<ii", ncall, len(values))
        data += np.asarray(values, dtype="<f4").tobytes()
    path.write_bytes(data)


def _gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_gguf_imatrix(
    path: Path,
    tensors: list[tuple[str, tuple[int, ...], int, list[float]]],
) -> None:
    offset = 0
    infos = bytearray()
    payload = bytearray()

    for name, dims, tensor_type, values in tensors:
        values_array = np.asarray(values, dtype="<f4")
        assert values_array.size == np.prod(dims, dtype=np.int64)

        infos += _gguf_string(name)
        infos += struct.pack("<I", len(dims))
        infos += struct.pack("<" + ("Q" * len(dims)), *dims)
        infos += struct.pack("<IQ", tensor_type, offset)

        tensor_payload = values_array.tobytes()
        payload += tensor_payload
        offset += len(tensor_payload)

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", 3, len(tensors), 0)
    header += infos

    padding = (-len(header)) % GGUF_DEFAULT_ALIGNMENT
    path.write_bytes(bytes(header) + (b"\0" * padding) + bytes(payload))


def test_load_imatrix_legacy_divides_ncall_and_returns_contiguous_float32(tmp_path: Path) -> None:
    path = tmp_path / "legacy.imatrix"
    _write_legacy_imatrix(
        path,
        [
            ("blk.0.ffn_gate.weight", 2, [2.0, 4.0, 6.0]),
            ("blk.1.ffn_gate.weight", 4, [8.0, 12.0]),
        ],
    )

    result = libgguf.load_imatrix(path)

    assert set(result) == {"blk.0.ffn_gate.weight", "blk.1.ffn_gate.weight"}
    np.testing.assert_array_equal(
        result["blk.0.ffn_gate.weight"],
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result["blk.1.ffn_gate.weight"],
        np.array([2.0, 3.0], dtype=np.float32),
    )
    for values in result.values():
        assert values.dtype == np.float32
        assert values.flags.c_contiguous


def test_load_imatrix_legacy_leaves_values_unchanged_when_ncall_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "legacy-zero-call.imatrix"
    _write_legacy_imatrix(path, [("output.weight", 0, [0.25, 0.5, 1.5])])

    result = libgguf.load_imatrix(path)

    np.testing.assert_array_equal(
        result["output.weight"],
        np.array([0.25, 0.5, 1.5], dtype=np.float32),
    )


def test_load_imatrix_legacy_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty-legacy.imatrix"
    path.write_bytes(struct.pack("<i", 0))

    with pytest.raises(ValueError, match="contains no entries"):
        libgguf.load_imatrix(path)


def test_load_imatrix_gguf_averages_sum_rows_by_counts(tmp_path: Path) -> None:
    path = tmp_path / "imatrix.gguf"
    _write_gguf_imatrix(
        path,
        [
            ("blk.0.attn_q.weight.in_sum2", (2, 3), GGML_TYPE_F32, [2.0, 4.0, 9.0, 12.0, 8.0, 16.0]),
            ("blk.0.attn_q.weight.counts", (3,), GGML_TYPE_F32, [2.0, 0.0, 4.0]),
        ],
    )

    result = libgguf.load_imatrix(path)

    assert set(result) == {"blk.0.attn_q.weight"}
    np.testing.assert_array_equal(
        result["blk.0.attn_q.weight"],
        np.array([1.0, 2.0, 1.0, 1.0, 2.0, 4.0], dtype=np.float32),
    )
    assert result["blk.0.attn_q.weight"].dtype == np.float32
    assert result["blk.0.attn_q.weight"].flags.c_contiguous


def test_load_imatrix_gguf_raises_for_missing_counts(tmp_path: Path) -> None:
    path = tmp_path / "missing-counts.gguf"
    _write_gguf_imatrix(
        path,
        [("blk.0.ffn_up.weight.in_sum2", (2,), GGML_TYPE_F32, [3.0, 6.0])],
    )

    with pytest.raises(ValueError, match="missing counts tensor.*blk\\.0\\.ffn_up\\.weight"):
        libgguf.load_imatrix(path)


def test_load_imatrix_gguf_raises_for_no_usable_entries(tmp_path: Path) -> None:
    path = tmp_path / "no-usable.gguf"
    _write_gguf_imatrix(
        path,
        [
            ("blk.0.ffn_down.weight.counts", (1,), GGML_TYPE_F32, [1.0]),
            ("blk.0.ffn_down.weight.in_sum2", (1,), GGML_TYPE_F16, [2.0]),
        ],
    )

    with pytest.raises(ValueError, match="contains no usable entries"):
        libgguf.load_imatrix(path)
