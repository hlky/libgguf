from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import libgguf


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _tensor_nbytes(shape: tuple[int, ...], qtype_value: int) -> int | None:
    try:
        qtype = libgguf.GGMLQuantizationType(qtype_value)
        block_size, type_size = libgguf.GGML_QUANT_SIZES[qtype]
    except (KeyError, ValueError):
        return None

    n_per_row = shape[0] if shape else 1
    if n_per_row % block_size != 0:
        return None
    n_rows = math.prod(shape[1:]) if len(shape) > 1 else 1
    return n_rows * (n_per_row // block_size * type_size)


def _minimal_gguf(
    path: Path,
    *,
    shape: tuple[int, ...] = (256, 2),
    tensors: list[tuple[str, tuple[int, ...], int, int]] | None = None,
    payload_size: int | None = None,
) -> int:
    q4_0 = int(libgguf.GGMLQuantizationType.Q4_0)
    if tensors is None:
        tensors = [("blocks.0.attn_v.weight", shape, q4_0, 64)]

    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<IQQ", 3, len(tensors), 4)
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
    for name, tensor_shape, qtype_value, offset in tensors:
        data += _gguf_string(name)
        data += struct.pack("<I", len(tensor_shape))
        data += struct.pack("<" + "Q" * len(tensor_shape), *tensor_shape)
        data += struct.pack("<IQ", qtype_value, offset)
    header_end = len(data)
    data += b"\0" * ((32 - header_end % 32) % 32)
    if payload_size is None:
        payload_size = 0
        for _name, tensor_shape, qtype_value, offset in tensors:
            nbytes = _tensor_nbytes(tensor_shape, qtype_value)
            payload_size = max(payload_size, offset + (0 if nbytes is None else nbytes))
    data += b"\0" * payload_size
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


def test_validate_gguf_accepts_valid_file(tmp_path: Path) -> None:
    gguf_path = tmp_path / "minimal.gguf"
    _minimal_gguf(gguf_path)

    result = libgguf.validate_gguf(gguf_path)

    assert result.ok
    assert result.file is not None
    assert result.issues == ()
    assert result.to_dict()["valid"] is True


def test_validate_gguf_warns_for_invalid_block_row_width(tmp_path: Path) -> None:
    gguf_path = tmp_path / "invalid-row-width.gguf"
    _minimal_gguf(gguf_path, shape=(16, 2))

    result = libgguf.validate_gguf(gguf_path)

    assert result.ok
    assert [issue.code for issue in result.warnings] == ["qtype_row_width"]
    assert result.warnings[0].tensor_name == "blocks.0.attn_v.weight"


def test_validate_gguf_warns_for_unknown_qtype(tmp_path: Path) -> None:
    gguf_path = tmp_path / "unknown-qtype.gguf"
    _minimal_gguf(
        gguf_path,
        tensors=[("blocks.0.attn_v.weight", (256, 2), 9999, 64)],
    )

    result = libgguf.validate_gguf(gguf_path)

    assert result.ok
    assert [issue.code for issue in result.warnings] == ["qtype_unknown"]
    assert result.warnings[0].details == {"qtype_value": 9999}


def test_validate_gguf_errors_for_zero_tensor_dimension(tmp_path: Path) -> None:
    gguf_path = tmp_path / "zero-dimension.gguf"
    _minimal_gguf(gguf_path, shape=(0, 2))

    result = libgguf.validate_gguf(gguf_path)

    assert not result.ok
    assert result.file is not None
    assert result.file.tensors[0].shape == (0, 2)
    assert [issue.code for issue in result.errors] == ["tensor_shape_invalid"]
    assert result.errors[0].tensor_name == "blocks.0.attn_v.weight"
    assert result.errors[0].details == {"shape": [0, 2]}


def test_validate_gguf_errors_for_payload_range(tmp_path: Path) -> None:
    gguf_path = tmp_path / "truncated-payload.gguf"
    _minimal_gguf(gguf_path, payload_size=64 + 100)

    result = libgguf.validate_gguf(gguf_path)

    assert not result.ok
    assert [issue.code for issue in result.errors] == ["tensor_payload_range"]
    assert result.errors[0].tensor_name == "blocks.0.attn_v.weight"


def test_validate_gguf_errors_for_payload_overlap(tmp_path: Path) -> None:
    gguf_path = tmp_path / "overlap.gguf"
    q4_0 = int(libgguf.GGMLQuantizationType.Q4_0)
    _minimal_gguf(
        gguf_path,
        tensors=[
            ("first.weight", (256, 2), q4_0, 0),
            ("second.weight", (256, 2), q4_0, 100),
        ],
    )

    result = libgguf.validate_gguf(gguf_path)

    assert not result.ok
    assert [issue.code for issue in result.errors] == ["tensor_payload_overlap"]
    assert result.errors[0].tensor_name == "second.weight"


def test_validate_gguf_errors_for_duplicate_tensor_names(tmp_path: Path) -> None:
    gguf_path = tmp_path / "duplicate-tensor.gguf"
    q4_0 = int(libgguf.GGMLQuantizationType.Q4_0)
    _minimal_gguf(
        gguf_path,
        tensors=[
            ("duplicate.weight", (256, 2), q4_0, 0),
            ("duplicate.weight", (256, 2), q4_0, 288),
        ],
    )

    result = libgguf.validate_gguf(gguf_path)

    assert not result.ok
    assert [issue.code for issue in result.errors] == ["tensor_duplicate_name"]
    assert result.errors[0].tensor_name == "duplicate.weight"


def test_validate_cli_json(tmp_path: Path) -> None:
    gguf_path = tmp_path / "minimal.gguf"
    _minimal_gguf(gguf_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from libgguf.inspect import validate_main; validate_main()",
            str(gguf_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["valid"] is True
    assert data["error_count"] == 0
    assert data["warning_count"] == 0


def test_validate_cli_text_exits_zero_for_warnings(tmp_path: Path) -> None:
    gguf_path = tmp_path / "invalid-row-width.gguf"
    _minimal_gguf(gguf_path, shape=(16, 2))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from libgguf.inspect import validate_main; validate_main()",
            str(gguf_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "VALID:" in result.stdout
    assert "Warnings:" in result.stdout
    assert "[qtype_row_width]" in result.stdout


def test_validate_cli_text_exits_one_for_errors(tmp_path: Path) -> None:
    gguf_path = tmp_path / "truncated-payload.gguf"
    _minimal_gguf(gguf_path, payload_size=64 + 100)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from libgguf.inspect import validate_main; validate_main()",
            str(gguf_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "INVALID:" in result.stdout
    assert "Errors:" in result.stdout
    assert "[tensor_payload_range]" in result.stdout
