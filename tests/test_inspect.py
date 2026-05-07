from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import libgguf


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _minimal_gguf(path: Path, *, shape: tuple[int, ...] = (256, 2)) -> int:
    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<IQQ", 3, 1, 4)
    data += _gguf_string("general.architecture")
    data += struct.pack("<I", 8)
    data += _gguf_string("test-arch")
    data += _gguf_string("general.quantization_version")
    data += struct.pack("<II", 4, 2)
    data += _gguf_string("general.alignment")
    data += struct.pack("<II", 4, 32)
    data += _gguf_string("tokenizer.ggml.tokens")
    data += struct.pack("<IIQ", 9, 8, 3)
    data += _gguf_string("a")
    data += _gguf_string("b")
    data += _gguf_string("c")
    data += _gguf_string("blocks.0.attn_v.weight")
    data += struct.pack("<I", len(shape))
    data += struct.pack("<" + "Q" * len(shape), *shape)
    data += struct.pack("<IQ", int(libgguf.GGMLQuantizationType.Q4_0), 64)
    header_end = len(data)
    data += b"\0" * ((32 - header_end % 32) % 32)
    data += b"\0" * (64 + 288)
    path.write_bytes(data)
    return header_end


def test_inspect_gguf_reads_metadata_and_tensor_descriptors_without_payload(tmp_path: Path) -> None:
    gguf_path = tmp_path / "minimal.gguf"
    header_end = _minimal_gguf(gguf_path)

    info = libgguf.inspect_gguf(gguf_path, max_array_values=2)

    assert info.version == 3
    assert info.tensor_count == 1
    assert info.metadata_kv_count == 4
    assert info.alignment == 32
    assert info.data_offset == header_end + ((32 - header_end % 32) % 32)
    assert info.metadata["general.architecture"].value == "test-arch"
    assert info.metadata["general.quantization_version"].value == 2
    tokens = info.metadata["tokenizer.ggml.tokens"]
    assert tokens.value == ["a", "b"]
    assert tokens.length == 3
    assert tokens.truncated

    assert info.tensors[0].to_dict() == {
        "name": "blocks.0.attn_v.weight",
        "shape": [256, 2],
        "qtype": "Q4_0",
        "qtype_value": int(libgguf.GGMLQuantizationType.Q4_0),
        "offset": 64,
        "data_offset": info.data_offset + 64,
        "nbytes": 288,
    }
    assert info.tensor_type_counts == {"Q4_0": 1}


def test_inspect_gguf_reports_unknown_nbytes_for_invalid_block_row_width(tmp_path: Path) -> None:
    gguf_path = tmp_path / "invalid-row-width.gguf"
    _minimal_gguf(gguf_path, shape=(16, 2))

    info = libgguf.inspect_gguf(gguf_path)

    assert info.tensors[0].shape == (16, 2)
    assert info.tensors[0].nbytes is None


def test_inspect_cli_json(tmp_path: Path) -> None:
    gguf_path = tmp_path / "minimal.gguf"
    _minimal_gguf(gguf_path)

    result = subprocess.run(
        [sys.executable, "-m", "libgguf.inspect", str(gguf_path), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["tensor_count"] == 1
    assert data["metadata"]["general.architecture"]["value"] == "test-arch"
    assert data["tensors"][0]["qtype"] == "Q4_0"
