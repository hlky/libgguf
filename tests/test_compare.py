from __future__ import annotations

import hashlib
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
    architecture: str = "test-arch",
    tensors: list[tuple[str, tuple[int, ...], int, int]] | None = None,
    duplicate_architecture: bool = False,
    payloads: dict[str, bytes] | None = None,
    payload_size: int | None = None,
) -> None:
    q4_0 = int(libgguf.GGMLQuantizationType.Q4_0)
    if tensors is None:
        tensors = [("blocks.0.attn_v.weight", (256, 2), q4_0, 64)]
    metadata_kv_count = 5 if duplicate_architecture else 4

    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<IQQ", 3, len(tensors), metadata_kv_count)
    data += _gguf_string("general.architecture")
    data += struct.pack("<I", 8)
    data += _gguf_string(architecture)
    if duplicate_architecture:
        data += _gguf_string("general.architecture")
        data += struct.pack("<I", 8)
        data += _gguf_string(architecture)
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

    if payloads:
        info = libgguf.inspect_gguf(path)
        with path.open("r+b") as handle:
            for name, payload in payloads.items():
                tensor = info.get_tensor(name)
                assert tensor is not None
                handle.seek(tensor.data_offset)
                handle.write(payload)


def _run_compare(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "libgguf.compare", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_compare_identical_descriptors_pass(tmp_path: Path) -> None:
    left = tmp_path / "left.gguf"
    right = tmp_path / "right.gguf"
    _minimal_gguf(left)
    _minimal_gguf(right)

    result = _run_compare(str(left), str(right))

    assert result.returncode == 0, result.stderr
    assert "No differences found." in result.stdout


def test_compare_descriptor_shape_qtype_and_order_differences_fail(tmp_path: Path) -> None:
    q4_0 = int(libgguf.GGMLQuantizationType.Q4_0)
    q8_0 = int(libgguf.GGMLQuantizationType.Q8_0)
    left = tmp_path / "left.gguf"
    shape_diff = tmp_path / "shape-diff.gguf"
    qtype_diff = tmp_path / "qtype-diff.gguf"
    order_diff = tmp_path / "order-diff.gguf"
    _minimal_gguf(
        left,
        tensors=[
            ("a.weight", (256, 2), q4_0, 64),
            ("b.weight", (256, 2), q4_0, 384),
        ],
    )
    _minimal_gguf(
        shape_diff,
        tensors=[
            ("a.weight", (256, 3), q4_0, 64),
            ("b.weight", (256, 2), q4_0, 512),
        ],
    )
    _minimal_gguf(
        qtype_diff,
        tensors=[
            ("a.weight", (256, 2), q8_0, 64),
            ("b.weight", (256, 2), q4_0, 640),
        ],
    )
    _minimal_gguf(
        order_diff,
        tensors=[
            ("b.weight", (256, 2), q4_0, 64),
            ("a.weight", (256, 2), q4_0, 384),
        ],
    )

    shape_result = _run_compare(str(left), str(shape_diff))
    qtype_result = _run_compare(str(left), str(qtype_diff))
    order_result = _run_compare(str(left), str(order_diff))

    assert shape_result.returncode == 1
    assert "tensor_shape" in shape_result.stdout
    assert qtype_result.returncode == 1
    assert "tensor_qtype" in qtype_result.stdout
    assert order_result.returncode == 1
    assert "tensor_order" in order_result.stdout


def test_compare_metadata_catches_value_and_duplicate_count_differences(tmp_path: Path) -> None:
    left = tmp_path / "left.gguf"
    value_diff = tmp_path / "value-diff.gguf"
    duplicate_diff = tmp_path / "duplicate-diff.gguf"
    _minimal_gguf(left)
    _minimal_gguf(value_diff, architecture="other-arch")
    _minimal_gguf(duplicate_diff, duplicate_architecture=True)

    value_result = _run_compare(str(left), str(value_diff), "--metadata")
    duplicate_result = _run_compare(str(left), str(duplicate_diff), "--metadata")

    assert value_result.returncode == 1
    assert "metadata_value" in value_result.stdout
    assert "general.architecture" in value_result.stdout
    assert duplicate_result.returncode == 1
    assert "metadata_duplicate_count" in duplicate_result.stdout


def test_compare_tensor_bytes_catches_payload_differences_with_hashes(tmp_path: Path) -> None:
    name = "blocks.0.attn_v.weight"
    nbytes = _tensor_nbytes((256, 2), int(libgguf.GGMLQuantizationType.Q4_0))
    assert nbytes is not None
    left_payload = bytes([1]) * nbytes
    right_payload = bytes([2]) * nbytes
    left = tmp_path / "left.gguf"
    right = tmp_path / "right.gguf"
    _minimal_gguf(left, payloads={name: left_payload})
    _minimal_gguf(right, payloads={name: right_payload})

    text_result = _run_compare(str(left), str(right), "--tensor-bytes")
    result = _run_compare(str(left), str(right), "--tensor-bytes", "--json")

    assert text_result.returncode == 1
    assert hashlib.sha256(left_payload).hexdigest() in text_result.stdout
    assert hashlib.sha256(right_payload).hexdigest() in text_result.stdout
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["differences"][0]["kind"] == "tensor_bytes"
    assert data["differences"][0]["left"] == {
        "nbytes": nbytes,
        "sha256": hashlib.sha256(left_payload).hexdigest(),
    }
    assert data["differences"][0]["right"] == {
        "nbytes": nbytes,
        "sha256": hashlib.sha256(right_payload).hexdigest(),
    }


def test_compare_json_schema_and_exit_code_for_difference(tmp_path: Path) -> None:
    left = tmp_path / "left.gguf"
    right = tmp_path / "right.gguf"
    _minimal_gguf(left)
    _minimal_gguf(right, architecture="other-arch")

    result = _run_compare(str(left), str(right), "--metadata", "--json")

    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["ok"] is False
    assert data["paths"] == {"left": str(left), "right": str(right)}
    assert data["modes"] == {
        "descriptors": True,
        "metadata": True,
        "tensor_bytes": False,
    }
    assert data["differences"][0]["kind"] == "metadata_value"
