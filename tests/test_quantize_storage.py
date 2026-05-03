import json
import struct

import numpy as np

from libgguf._metadata import GGMLQuantizationType
from libgguf.quantize import (
    _load_bf16_safetensors_tensor,
    _open_tensor_source,
    _read_safetensors_header,
    _to_numpy_for_qtype,
    _unquantized_tensor_data,
)


def test_unquantized_f32_and_f16_reuse_matching_arrays() -> None:
    f32 = np.arange(8, dtype=np.float32).reshape(2, 4)
    f16 = np.arange(8, dtype=np.float16).reshape(2, 4)

    assert _unquantized_tensor_data(f32, "F32", GGMLQuantizationType.F32) is f32
    assert _unquantized_tensor_data(f16, "F16", GGMLQuantizationType.F16) is f16


def test_unquantized_bf16_reuses_raw_uint16_array() -> None:
    bf16 = np.array([[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16)

    assert _unquantized_tensor_data(bf16, "BF16", GGMLQuantizationType.BF16) is bf16


def test_raw_bf16_uint16_decodes_as_float_when_conversion_is_needed() -> None:
    bf16 = np.array([[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16)

    decoded = _to_numpy_for_qtype(bf16, GGMLQuantizationType.F32, source_dtype="BF16")

    np.testing.assert_array_equal(decoded, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_bf16_safetensors_loader_returns_raw_uint16_memmap_view(tmp_path) -> None:
    bf16 = np.array([[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16)
    payload = bf16.tobytes()
    header = {"x": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(payload)]}}
    header_bytes = json.dumps(header).encode("utf-8")
    path = tmp_path / "x.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)

    data_start, parsed_header = _read_safetensors_header(path)
    file_bytes = np.memmap(path, dtype=np.uint8, mode="r")
    loaded = _load_bf16_safetensors_tensor(file_bytes, data_start, parsed_header, "x")

    assert loaded.dtype == np.dtype(np.uint16)
    assert np.shares_memory(loaded.view(np.uint8), file_bytes)
    np.testing.assert_array_equal(loaded, bf16)


def test_safetensors_source_uses_header_dtype_with_bf16_memmap_view(tmp_path) -> None:
    bf16 = np.array([[0x3F80, 0x4000], [0x4040, 0x4080]], dtype=np.uint16)
    payload = bf16.tobytes()
    header = {"x": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(payload)]}}
    header_bytes = json.dumps(header).encode("utf-8")
    path = tmp_path / "x.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)

    with _open_tensor_source(path) as (_, source):
        shape, dtype = source.tensor_meta("x")
        loaded = source.load_tensor("x")

    assert loaded.dtype == np.dtype(np.uint16)
    assert shape == (2, 2)
    assert dtype == "BF16"
    np.testing.assert_array_equal(loaded, bf16)
