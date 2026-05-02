from __future__ import annotations

import ctypes
from pathlib import Path
import struct

import numpy as np
import pytest

import libgguf
from ggml_types import GGML_FORMAT_INFO, GGMLQuantizationType, quant_shape_to_byte_shape
from scripts.build_libgguf import build_shared_lib, default_output_path


REQUIRED_EXTENSION_FUNCTIONS = (
    "load_imatrix",
    "quantize_requires_imatrix",
    "quantize_rows_raw",
    "row_size",
    "type_name",
    "type_size",
)

REQUIRED_C_ABI_SYMBOLS = (
    "libgguf_row_size",
    "libgguf_type_size",
    "libgguf_type_name",
    "libgguf_quantize_chunk",
    "libgguf_quantize_requires_imatrix",
    "libgguf_quantize_free",
)

SUPPORTED_LIBGGUF_QTYPES = (
    GGMLQuantizationType.Q1_0,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
)


def test_libgguf_extension_imports() -> None:
    for name in REQUIRED_EXTENSION_FUNCTIONS:
        assert hasattr(libgguf, name), f"missing libgguf.{name}"


@pytest.fixture(scope="session")
def built_shared_libgguf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("libgguf-shared")
    output = root / default_output_path().name
    return build_shared_lib(output=output, build_dir=root / "build")


def test_shared_libgguf_exports_c_abi_symbols(built_shared_libgguf: Path) -> None:
    lib = ctypes.CDLL(str(built_shared_libgguf))

    for symbol in REQUIRED_C_ABI_SYMBOLS:
        assert hasattr(lib, symbol), f"missing exported symbol {symbol}"


@pytest.mark.parametrize("qtype", SUPPORTED_LIBGGUF_QTYPES)
def test_libgguf_row_size_matches_format_metadata(qtype: GGMLQuantizationType) -> None:
    info = GGML_FORMAT_INFO[qtype]
    n_per_row = info.block_size * 2

    assert libgguf.row_size(qtype, n_per_row) == info.type_size * 2
    assert libgguf.type_size(qtype) == info.type_size


@pytest.mark.parametrize("qtype", SUPPORTED_LIBGGUF_QTYPES)
def test_libgguf_raw_quantization_returns_expected_byte_count(qtype: GGMLQuantizationType) -> None:
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)

    if libgguf.quantize_requires_imatrix(qtype):
        imatrix = np.sum(rows * rows, axis=0, dtype=np.float32)
    else:
        imatrix = None

    raw = libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1], imatrix)

    assert isinstance(raw, bytes)
    assert len(raw) == rows.shape[0] * info.type_size


@pytest.mark.parametrize("qtype", SUPPORTED_LIBGGUF_QTYPES)
def test_libgguf_quantization_smoke(qtype: GGMLQuantizationType) -> None:
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)

    quantized = libgguf.quantize_rows(rows, qtype)

    assert quantized.shape == quant_shape_to_byte_shape(rows.shape, qtype)
    assert quantized.dtype == np.uint8
    assert np.any(quantized)


def test_libgguf_quantize_rows_accepts_explicit_imatrix() -> None:
    qtype = GGMLQuantizationType.IQ2_XXS
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)
    imatrix = np.linspace(0.5, 1.5, info.block_size, dtype=np.float32)

    quantized = libgguf.quantize_rows(rows, qtype, imatrix=imatrix)
    raw = libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1], imatrix)

    assert np.array_equal(quantized.reshape(-1), np.frombuffer(raw, dtype=np.uint8))


def test_libgguf_loads_legacy_imatrix(tmp_path: Path) -> None:
    path = tmp_path / "imatrix.dat"
    with path.open("wb") as f:
        f.write(struct.pack("<i", 1))
        name = b"blk.0.attn_q.weight"
        values = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        f.write(struct.pack("<i", len(name)))
        f.write(name)
        f.write(struct.pack("<ii", 2, values.size))
        f.write(values.tobytes())

    imatrix = libgguf.load_imatrix(path)

    assert set(imatrix) == {"blk.0.attn_q.weight"}
    np.testing.assert_array_equal(imatrix["blk.0.attn_q.weight"], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_libgguf_loads_gguf_imatrix(tmp_path: Path) -> None:
    path = tmp_path / "imatrix.gguf"
    tensors = [
        ("blk.0.attn_q.weight.in_sum2", (3, 2), np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0], dtype=np.float32)),
        ("blk.0.attn_q.weight.counts", (1, 2), np.array([2.0, 4.0], dtype=np.float32)),
    ]

    infos = bytearray()
    data = bytearray()
    for name, dims, values in tensors:
        encoded = name.encode("utf-8")
        offset = len(data)
        infos += struct.pack("<Q", len(encoded)) + encoded
        infos += struct.pack("<I", len(dims))
        infos += struct.pack("<" + "Q" * len(dims), *dims)
        infos += struct.pack("<IQ", 0, offset)
        data += values.tobytes()

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0)
    padding = (-len(header) - len(infos)) % 32
    path.write_bytes(header + infos + (b"\0" * padding) + data)

    imatrix = libgguf.load_imatrix(path)

    assert set(imatrix) == {"blk.0.attn_q.weight"}
    np.testing.assert_array_equal(
        imatrix["blk.0.attn_q.weight"],
        np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
