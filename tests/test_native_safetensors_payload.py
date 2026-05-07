from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType
from libgguf import _libgguf
from libgguf.quantize import convert_safetensors_to_gguf_native, convert_to_gguf


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> tuple[int, dict[str, object]]:
    payload = bytearray()
    header: dict[str, object] = {}
    for name, (dtype, shape, data) in tensors.items():
        begin = len(payload)
        payload += data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(payload)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    return 8 + len(header_bytes), header


def _payload_plan(
    header: dict[str, object],
    data_start: int,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    qtype: GGMLQuantizationType,
    nbytes: int,
    *,
    write_shape: tuple[int, ...] | None = None,
    imatrix: np.ndarray | None = None,
) -> dict[str, object]:
    info = header[name]
    assert isinstance(info, dict)
    begin, end = info["data_offsets"]
    return {
        "key": name,
        "source_dtype": dtype,
        "source_shape": shape,
        "write_shape": write_shape or shape,
        "qtype": int(qtype),
        "nbytes": nbytes,
        "data_begin": data_start + int(begin),
        "data_end": data_start + int(end),
        "imatrix": imatrix,
    }


def _write_native_payload(src: Path, plans: list[dict[str, object]], tmp_path: Path, *, alignment: int = 1) -> bytes:
    out = tmp_path / "payload.bin"
    with out.open("w+b") as f:
        _libgguf.write_safetensors_payload(str(src), f.fileno(), plans, alignment, scratch_bytes=4096)
        f.flush()
    return out.read_bytes()


def test_native_payload_direct_copies_f32_and_bf16(tmp_path: Path) -> None:
    f32 = np.arange(6, dtype=np.float32).reshape(2, 3)
    bf16 = np.array([[0x3F80, 0x4000, 0x4040]], dtype=np.uint16)
    src = tmp_path / "direct.safetensors"
    data_start, header = _write_safetensors(
        src,
        {
            "f32": ("F32", f32.shape, f32.tobytes()),
            "bf16": ("BF16", bf16.shape, bf16.tobytes()),
        },
    )

    payload = _write_native_payload(
        src,
        [
            _payload_plan(header, data_start, "f32", "F32", f32.shape, GGMLQuantizationType.F32, f32.nbytes),
            _payload_plan(header, data_start, "bf16", "BF16", bf16.shape, GGMLQuantizationType.BF16, bf16.nbytes),
        ],
        tmp_path,
    )

    assert payload == f32.tobytes() + bf16.tobytes()


def test_native_payload_converts_f16_to_f32(tmp_path: Path) -> None:
    f16 = np.array([[0.0, 1.5, -2.25], [3.5, -4.0, 8.0]], dtype=np.float16)
    src = tmp_path / "convert.safetensors"
    data_start, header = _write_safetensors(src, {"x": ("F16", f16.shape, f16.tobytes())})

    payload = _write_native_payload(
        src,
        [_payload_plan(header, data_start, "x", "F16", f16.shape, GGMLQuantizationType.F32, f16.size * 4)],
        tmp_path,
    )

    np.testing.assert_array_equal(np.frombuffer(payload, dtype=np.float32), f16.astype(np.float32).reshape(-1))


def test_native_payload_copies_scalar_f16(tmp_path: Path) -> None:
    scalar = np.array(0.99, dtype=np.float16)
    src = tmp_path / "scalar.safetensors"
    data_start, header = _write_safetensors(src, {"x": ("F16", (), scalar.tobytes())})

    payload = _write_native_payload(
        src,
        [_payload_plan(header, data_start, "x", "F16", (), GGMLQuantizationType.F16, scalar.nbytes)],
        tmp_path,
    )

    assert payload == scalar.tobytes()


def test_native_payload_converts_bf16_to_f32(tmp_path: Path) -> None:
    f32 = np.array([[0.0, 1.5, -2.25], [3.5, -4.0, 8.0]], dtype=np.float32)
    bf16 = (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    src = tmp_path / "convert_bf16.safetensors"
    data_start, header = _write_safetensors(src, {"x": ("BF16", bf16.shape, bf16.tobytes())})

    payload = _write_native_payload(
        src,
        [_payload_plan(header, data_start, "x", "BF16", bf16.shape, GGMLQuantizationType.F32, bf16.size * 4)],
        tmp_path,
    )

    expected = (bf16.astype(np.uint32) << np.uint32(16)).view(np.float32)
    np.testing.assert_array_equal(np.frombuffer(payload, dtype=np.float32), expected.reshape(-1))


def test_native_payload_quantizes_bf16_to_q8_0_without_scratch(tmp_path: Path) -> None:
    f32 = np.linspace(-3.0, 3.0, 128, dtype=np.float32).reshape(4, 32)
    bf16 = (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    bf16_as_f32 = (bf16.astype(np.uint32) << np.uint32(16)).view(np.float32)
    src = tmp_path / "bf16_q8.safetensors"
    data_start, header = _write_safetensors(src, {"x": ("BF16", bf16.shape, bf16.tobytes())})
    expected = libgguf.quantize_rows(bf16_as_f32, GGMLQuantizationType.Q8_0)

    payload = _write_native_payload(
        src,
        [_payload_plan(header, data_start, "x", "BF16", bf16.shape, GGMLQuantizationType.Q8_0, expected.nbytes)],
        tmp_path,
    )

    assert payload == expected.tobytes()


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K])
def test_native_payload_quantizes_bf16_with_generic_fused_path(tmp_path: Path, qtype: GGMLQuantizationType) -> None:
    width = 256 if qtype in {GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K} else 32
    f32 = np.linspace(-2.5, 2.5, width * 4, dtype=np.float32).reshape(4, width)
    bf16 = (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    bf16_as_f32 = (bf16.astype(np.uint32) << np.uint32(16)).view(np.float32)
    src = tmp_path / f"bf16_{qtype.name.lower()}.safetensors"
    data_start, header = _write_safetensors(src, {"x": ("BF16", bf16.shape, bf16.tobytes())})
    expected = libgguf.quantize_rows(bf16_as_f32, qtype)

    payload = _write_native_payload(
        src,
        [_payload_plan(header, data_start, "x", "BF16", bf16.shape, qtype, expected.nbytes)],
        tmp_path,
    )

    assert payload == expected.tobytes()


def test_native_payload_quantized_output_matches_python_api(tmp_path: Path) -> None:
    rows = np.linspace(-1.5, 1.5, 64, dtype=np.float32).reshape(2, 32)
    src = tmp_path / "quant.safetensors"
    data_start, header = _write_safetensors(src, {"x": ("F32", rows.shape, rows.tobytes())})
    expected = libgguf.quantize_rows(rows, GGMLQuantizationType.Q4_0)

    payload = _write_native_payload(
        src,
        [_payload_plan(header, data_start, "x", "F32", rows.shape, GGMLQuantizationType.Q4_0, expected.nbytes)],
        tmp_path,
    )

    assert payload == expected.tobytes()


def test_native_payload_rejects_truncated_source_range(tmp_path: Path) -> None:
    rows = np.arange(32, dtype=np.float32).reshape(1, 32)
    src = tmp_path / "bad.safetensors"
    data_start, header = _write_safetensors(src, {"x": ("F32", rows.shape, rows.tobytes())})
    plan = _payload_plan(header, data_start, "x", "F32", rows.shape, GGMLQuantizationType.F32, rows.nbytes)
    plan["data_begin"] = src.stat().st_size + 16
    plan["data_end"] = int(plan["data_begin"]) + rows.nbytes

    with pytest.raises(RuntimeError, match="exceeds file size"):
        _write_native_payload(src, [plan], tmp_path)


def test_native_converter_matches_python_converter_for_tiny_safetensors(tmp_path: Path) -> None:
    key = "double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(2, 32)
    src = tmp_path / "model.safetensors"
    _write_safetensors(src, {key: ("F32", rows.shape, rows.tobytes())})
    py_out = tmp_path / "python.gguf"
    native_out = tmp_path / "native.gguf"

    convert_to_gguf(src, py_out, "Q4_0", policy="uniform", overwrite=True)
    convert_safetensors_to_gguf_native(src, native_out, "Q4_0", policy="uniform", overwrite=True)

    assert native_out.read_bytes() == py_out.read_bytes()

def test_native_converter_allows_scalar_tensors_outside_selected_prefix(tmp_path: Path) -> None:
    key = "model.double_layers.3.modX.1.weight"
    rows = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(2, 32)
    scalar = np.array(1.0, dtype=np.float16)
    src = tmp_path / "model_with_scalar.safetensors"
    _write_safetensors(
        src,
        {
            key: ("F32", rows.shape, rows.tobytes()),
            "vae.model_ema.decay": ("F16", (), scalar.tobytes()),
        },
    )
    py_out = tmp_path / "python.gguf"
    native_out = tmp_path / "native.gguf"

    convert_to_gguf(src, py_out, "Q4_0", policy="uniform", overwrite=True)
    convert_safetensors_to_gguf_native(src, native_out, "Q4_0", policy="uniform", overwrite=True)

    assert native_out.read_bytes() == py_out.read_bytes()
