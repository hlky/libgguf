from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import struct
import sys

import numpy as np
import pytest

import libgguf
from libgguf import _libgguf
from ggml_types import GGML_FORMAT_INFO, GGMLQuantizationType, quant_shape_to_byte_shape
from scripts.build_libgguf import build_shared_lib, default_output_path


REQUIRED_EXTENSION_FUNCTIONS = (
    "dequantize_rows_raw",
    "dequantize_rows_into_raw",
    "load_imatrix",
    "quantize_requires_imatrix",
    "quantize_rows_raw",
    "quantize_rows_into_raw",
    "row_size",
    "store_rows",
    "type_name",
    "type_size",
)

REQUIRED_C_ABI_SYMBOLS = (
    "libgguf_dequantize_chunk",
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


def test_libgguf_store_rows_matches_plain_storage_formats() -> None:
    rows = np.array(
        [
            [0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 1.0e-8],
            [np.inf, -np.inf, np.nan, -3.5, 42.25, -1000.0, 3.1415927],
        ],
        dtype=np.float32,
    )

    stored_f32 = libgguf.store_rows(rows, 0)
    assert stored_f32.dtype == np.float32
    assert stored_f32.shape == rows.shape
    np.testing.assert_array_equal(stored_f32.view(np.uint32), rows.view(np.uint32))

    stored_f16 = libgguf.store_rows(rows, 1)
    assert stored_f16.dtype == np.float16
    assert stored_f16.shape == rows.shape
    np.testing.assert_array_equal(stored_f16.view(np.uint16), rows.astype(np.float16).view(np.uint16))

    bits = rows.view(np.uint32)
    high = bits >> 16
    expected_bf16 = np.where(
        (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000),
        high | np.uint32(64),
        (bits + (np.uint32(0x7FFF) + (high & np.uint32(1)))) >> np.uint32(16),
    ).astype(np.uint16)

    stored_bf16 = libgguf.store_rows(rows, 30)
    assert stored_bf16.dtype == np.uint8
    assert stored_bf16.shape == (*rows.shape[:-1], rows.shape[-1] * 2)
    np.testing.assert_array_equal(stored_bf16.view(np.uint16).reshape(rows.shape), expected_bf16)


def test_libgguf_storage_backends_match_reference() -> None:
    rows = np.array(
        [
            [0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 1.0e-8, np.inf, -np.inf],
            [np.nan, -3.5, 42.25, -1000.0, 3.1415927, 1.5, -7.25, 2.0e-4, -2.0e-4],
        ],
        dtype=np.float32,
    )

    try:
        _libgguf._storage_set_backend("ref")
        expected = libgguf.store_rows(rows, 30).tobytes()

        for backend in ("sse2", "sse4_1", "avx2"):
            if not _libgguf._storage_cpu_supports_backend(backend):
                continue
            _libgguf._storage_set_backend(backend)
            assert _libgguf._storage_backend() == backend
            actual = libgguf.store_rows(rows, 30).tobytes()
            assert actual == expected, backend
    finally:
        _libgguf._storage_set_backend("auto")


@pytest.fixture(scope="session")
def built_shared_libgguf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("libgguf-shared")
    output = root / default_output_path().name
    return build_shared_lib(output=output, build_dir=root / "build")


def test_shared_libgguf_exports_c_abi_symbols(built_shared_libgguf: Path) -> None:
    lib = ctypes.CDLL(str(built_shared_libgguf))

    for symbol in REQUIRED_C_ABI_SYMBOLS:
        assert hasattr(lib, symbol), f"missing exported symbol {symbol}"


def test_shared_libgguf_parallel_quantization_matches_serial(
    built_shared_libgguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = ctypes.CDLL(str(built_shared_libgguf))
    lib.libgguf_row_size.argtypes = [ctypes.c_int, ctypes.c_longlong]
    lib.libgguf_row_size.restype = ctypes.c_size_t
    lib.libgguf_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.libgguf_quantize_chunk.restype = ctypes.c_size_t

    qtype = GGMLQuantizationType.Q4_K
    rows = np.linspace(-1.5, 1.5, 128 * 256, dtype=np.float32).reshape(128, 256)
    row_size = lib.libgguf_row_size(qtype, rows.shape[1])
    serial = np.empty(rows.shape[0] * row_size, dtype=np.uint8)
    parallel = np.empty_like(serial)
    src = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    monkeypatch.setenv("LIBGGUF_NUM_THREADS", "1")
    serial_written = lib.libgguf_quantize_chunk(qtype, src, serial.ctypes.data, 0, rows.shape[0], rows.shape[1], None)

    monkeypatch.setenv("LIBGGUF_NUM_THREADS", "4")
    parallel_written = lib.libgguf_quantize_chunk(qtype, src, parallel.ctypes.data, 0, rows.shape[0], rows.shape[1], None)

    assert serial_written == parallel_written == serial.nbytes
    np.testing.assert_array_equal(parallel, serial)


def test_shared_libgguf_dequantize_chunk_round_trips_q8_0(built_shared_libgguf: Path) -> None:
    lib = ctypes.CDLL(str(built_shared_libgguf))
    lib.libgguf_dequantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    lib.libgguf_dequantize_chunk.restype = ctypes.c_size_t

    qtype = GGMLQuantizationType.Q8_0
    rows = np.linspace(-1.0, 1.0, 2 * 32, dtype=np.float32).reshape(2, 32)
    quantized = libgguf.quantize_rows(rows, qtype)
    dequantized = np.empty_like(rows)

    written = lib.libgguf_dequantize_chunk(
        qtype,
        quantized.ctypes.data,
        dequantized.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0,
        rows.shape[0],
        rows.shape[1],
    )

    assert written == dequantized.nbytes
    np.testing.assert_allclose(dequantized, rows, atol=0.01)


def test_shared_libgguf_parallel_dequantization_matches_serial(
    built_shared_libgguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = ctypes.CDLL(str(built_shared_libgguf))
    lib.libgguf_dequantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    lib.libgguf_dequantize_chunk.restype = ctypes.c_size_t

    qtype = GGMLQuantizationType.Q4_K
    rows = np.linspace(-1.5, 1.5, 128 * 256, dtype=np.float32).reshape(128, 256)
    quantized = libgguf.quantize_rows(rows, qtype)
    serial = np.empty_like(rows)
    parallel = np.empty_like(rows)

    monkeypatch.setenv("LIBGGUF_NUM_THREADS", "1")
    serial_written = lib.libgguf_dequantize_chunk(
        qtype,
        quantized.ctypes.data,
        serial.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0,
        rows.shape[0],
        rows.shape[1],
    )

    monkeypatch.setenv("LIBGGUF_NUM_THREADS", "4")
    parallel_written = lib.libgguf_dequantize_chunk(
        qtype,
        quantized.ctypes.data,
        parallel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0,
        rows.shape[0],
        rows.shape[1],
    )

    assert serial_written == parallel_written == rows.nbytes
    np.testing.assert_array_equal(parallel, serial)


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


def test_libgguf_quantize_rows_into_raw_matches_allocating_raw() -> None:
    qtype = GGMLQuantizationType.Q8_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 5 * info.block_size, dtype=np.float32).reshape(5, info.block_size)
    dst = bytearray(rows.shape[0] * info.type_size)

    written = libgguf.quantize_rows_into_raw(qtype, rows, dst, rows.shape[0], rows.shape[1])
    raw = libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1])

    assert written == len(dst)
    assert bytes(dst) == raw


def test_libgguf_quantize_rows_into_raw_rejects_small_destination() -> None:
    qtype = GGMLQuantizationType.Q8_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)
    dst = bytearray(rows.shape[0] * info.type_size - 1)

    with pytest.raises(ValueError, match="dst buffer is smaller"):
        libgguf.quantize_rows_into_raw(qtype, rows, dst, rows.shape[0], rows.shape[1])


@pytest.mark.parametrize("qtype", SUPPORTED_LIBGGUF_QTYPES)
def test_libgguf_quantization_smoke(qtype: GGMLQuantizationType) -> None:
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)

    quantized = libgguf.quantize_rows(rows, qtype)

    assert quantized.shape == quant_shape_to_byte_shape(rows.shape, qtype)
    assert quantized.dtype == np.uint8
    assert np.any(quantized)


@pytest.mark.parametrize("qtype", SUPPORTED_LIBGGUF_QTYPES)
def test_libgguf_dequantization_smoke(qtype: GGMLQuantizationType) -> None:
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)

    quantized = libgguf.quantize_rows(rows, qtype)
    dequantized = libgguf.dequantize_rows(quantized, qtype)

    assert dequantized.shape == rows.shape
    assert dequantized.dtype == np.float32
    assert np.all(np.isfinite(dequantized))


def test_libgguf_dequantize_rows_raw_matches_array_wrapper() -> None:
    qtype = GGMLQuantizationType.Q4_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 4 * info.block_size, dtype=np.float32).reshape(4, info.block_size)
    quantized = libgguf.quantize_rows(rows, qtype)

    raw = libgguf.dequantize_rows_raw(qtype, quantized, rows.shape[0], rows.shape[1])
    dequantized = libgguf.dequantize_rows(quantized, qtype)

    np.testing.assert_array_equal(np.frombuffer(raw, dtype=np.float32).reshape(rows.shape), dequantized)


def test_libgguf_dequantize_rows_into_raw_rejects_small_destination() -> None:
    qtype = GGMLQuantizationType.Q8_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)
    quantized = libgguf.quantize_rows(rows, qtype)
    dst = bytearray(rows.nbytes - 1)

    with pytest.raises(ValueError, match="dst buffer is smaller"):
        libgguf.dequantize_rows_into_raw(qtype, quantized, dst, rows.shape[0], rows.shape[1])


@pytest.mark.parametrize("qtype", SUPPORTED_LIBGGUF_QTYPES)
def test_libgguf_dequant_supported_backends_match_reference(qtype: GGMLQuantizationType) -> None:
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 5 * info.block_size, dtype=np.float32).reshape(5, info.block_size)
    if libgguf.quantize_requires_imatrix(qtype):
        imatrix = np.sum(rows * rows, axis=0, dtype=np.float32)
    else:
        imatrix = None
    quantized = libgguf.quantize_rows(rows, qtype, imatrix=imatrix)

    expected = _libgguf._dequantize_for_backend(qtype, "ref", quantized, rows.shape[0], rows.shape[1])
    assert _libgguf._dequant_backend(qtype) in {"ref", "sse2", "sse4_1", "avx2"}

    for backend in ("sse2", "sse4_1", "avx2"):
        if _libgguf._dequant_cpu_supports_backend(backend):
            actual = _libgguf._dequantize_for_backend(qtype, backend, quantized, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_quantize_rows_accepts_explicit_imatrix() -> None:
    qtype = GGMLQuantizationType.IQ2_XXS
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.5, 1.5, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)
    imatrix = np.linspace(0.5, 1.5, info.block_size, dtype=np.float32)

    quantized = libgguf.quantize_rows(rows, qtype, imatrix=imatrix)
    raw = libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1], imatrix)

    assert np.array_equal(quantized.reshape(-1), np.frombuffer(raw, dtype=np.uint8))


def test_libgguf_q4_0_matches_scalar_reference() -> None:
    qtype = GGMLQuantizationType.Q4_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 5 * info.block_size, dtype=np.float32).reshape(5, info.block_size)

    expected = bytearray()
    for row in rows:
        amax = np.float32(0.0)
        max_value = np.float32(0.0)
        for value in row:
            if amax < np.abs(value):
                amax = np.abs(value)
                max_value = value
        d = np.float32(max_value / np.float32(-8.0))
        inv = np.float32(1.0 / d) if d != 0 else np.float32(0.0)
        lo = np.minimum(15, (row[:16] * inv + np.float32(8.5)).astype(np.int8)).astype(np.uint8)
        hi = np.minimum(15, (row[16:] * inv + np.float32(8.5)).astype(np.int8)).astype(np.uint8)
        expected += np.float16(d).tobytes()
        expected += (lo | (hi << 4)).tobytes()

    raw = libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1], None)

    assert raw == bytes(expected)


def test_libgguf_q4_0_runtime_backend_is_supported() -> None:
    backend = _libgguf._q4_0_backend()

    assert backend in {"ref", "sse2", "sse4_1", "avx2"}
    assert _libgguf._q4_0_cpu_supports_backend(backend)


def test_libgguf_q4_0_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.Q4_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 7 * info.block_size, dtype=np.float32).reshape(7, info.block_size)
    rows[0] = 0.0
    rows[0, 1] = 3.0
    rows[0, 4] = -3.0
    expected = _libgguf._quantize_q4_0_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    for backend in ("sse2", "sse4_1", "avx2"):
        if _libgguf._q4_0_cpu_supports_backend(backend):
            actual = _libgguf._quantize_q4_0_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_q8_0_matches_scalar_reference() -> None:
    qtype = GGMLQuantizationType.Q8_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 5 * info.block_size, dtype=np.float32).reshape(5, info.block_size)

    expected = bytearray()
    for row in rows:
        amax = np.max(np.abs(row), initial=np.float32(0.0)).astype(np.float32)
        d = np.float32(amax / ((1 << 7) - 1))
        inv = np.float32(1.0 / d) if d != 0 else np.float32(0.0)
        scaled = row * inv
        quants = np.trunc(np.abs(scaled) + np.float32(0.5)).astype(np.int32)
        quants = np.where(scaled < 0, -quants, quants).astype(np.int8)
        expected += np.float16(d).tobytes()
        expected += quants.tobytes()

    raw = libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1], None)

    assert raw == bytes(expected)


def test_libgguf_q8_0_runtime_backend_is_supported() -> None:
    backend = _libgguf._q8_0_backend()

    assert backend in {"ref", "sse2", "sse4_1", "avx2"}
    assert _libgguf._q8_0_cpu_supports_backend(backend)


def test_libgguf_q8_0_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.Q8_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 7 * info.block_size, dtype=np.float32).reshape(7, info.block_size)
    expected = _libgguf._quantize_q8_0_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    for backend in ("sse2", "sse4_1", "avx2"):
        if _libgguf._q8_0_cpu_supports_backend(backend):
            actual = _libgguf._quantize_q8_0_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_simple_quant_supported_backends_match_reference() -> None:
    cases = [
        (
            GGMLQuantizationType.Q1_0,
            _libgguf._q1_0_cpu_supports_backend,
            _libgguf._quantize_q1_0_for_backend,
        ),
        (
            GGMLQuantizationType.Q4_1,
            _libgguf._q4_1_cpu_supports_backend,
            _libgguf._quantize_q4_1_for_backend,
        ),
        (
            GGMLQuantizationType.Q5_0,
            _libgguf._q5_0_cpu_supports_backend,
            _libgguf._quantize_q5_0_for_backend,
        ),
        (
            GGMLQuantizationType.Q5_1,
            _libgguf._q5_1_cpu_supports_backend,
            _libgguf._quantize_q5_1_for_backend,
        ),
        (
            GGMLQuantizationType.MXFP4,
            _libgguf._mxfp4_cpu_supports_backend,
            _libgguf._quantize_mxfp4_for_backend,
        ),
        (
            GGMLQuantizationType.NVFP4,
            _libgguf._nvfp4_cpu_supports_backend,
            _libgguf._quantize_nvfp4_for_backend,
        ),
    ]

    for qtype, supports_backend, quantize_for_backend in cases:
        info = GGML_FORMAT_INFO[qtype]
        rows = np.linspace(-2.0, 2.0, 5 * info.block_size, dtype=np.float32).reshape(5, info.block_size)
        rows[0] = 0.0
        rows[1, ::3] *= np.float32(-1.75)
        rows[2, 1::5] += np.float32(0.125)
        expected = quantize_for_backend("ref", rows, rows.shape[0], rows.shape[1])

        for backend in ("sse2", "sse4_1", "avx2"):
            if supports_backend(backend):
                actual = quantize_for_backend(backend, rows, rows.shape[0], rows.shape[1])
                assert actual == expected, f"{qtype.name}/{backend}"


def test_libgguf_q4_k_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.Q4_K
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)
    rows[0] = 0.0
    rows[1, ::17] *= -3.0
    expected = _libgguf._quantize_q4_k_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    for backend in ("sse2", "sse4_1", "avx2"):
        if _libgguf._q4_k_cpu_supports_backend(backend):
            actual = _libgguf._quantize_q4_k_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_k_quant_supported_backends_match_reference() -> None:
    cases = [
        (GGMLQuantizationType.Q2_K, _libgguf._q2_k_cpu_supports_backend, _libgguf._quantize_q2_k_for_backend),
        (GGMLQuantizationType.Q3_K, _libgguf._q3_k_cpu_supports_backend, _libgguf._quantize_q3_k_for_backend),
        (GGMLQuantizationType.Q5_K, _libgguf._q5_k_cpu_supports_backend, _libgguf._quantize_q5_k_for_backend),
        (GGMLQuantizationType.Q6_K, _libgguf._q6_k_cpu_supports_backend, _libgguf._quantize_q6_k_for_backend),
    ]
    for qtype, supports_backend, quantize_for_backend in cases:
        info = GGML_FORMAT_INFO[qtype]
        rows = np.linspace(-2.0, 2.0, 4 * info.block_size, dtype=np.float32).reshape(4, info.block_size)
        rows[0] = 0.0
        rows[1, ::7] *= np.float32(-1.75)
        rows[2, 3::11] += np.float32(0.25)
        expected = quantize_for_backend("ref", rows, rows.shape[0], rows.shape[1])

        for backend in ("sse2", "sse4_1", "avx2"):
            if not supports_backend(backend):
                continue
            actual = quantize_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected, f"{qtype.name}/{backend}"


def test_libgguf_tq2_0_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.TQ2_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-1.0, 1.0, 3 * info.block_size, dtype=np.float32).reshape(3, info.block_size)
    expected = _libgguf._quantize_tq2_0_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    assert _libgguf._tq2_0_backend() in {"ref", "sse2", "sse4_1", "avx2"}
    for backend in ("sse2", "sse4_1", "avx2"):
        if _libgguf._tq2_0_cpu_supports_backend(backend):
            actual = _libgguf._quantize_tq2_0_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_tq1_0_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.TQ1_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 5 * info.block_size, dtype=np.float32).reshape(5, info.block_size)
    rows[0] = 0.0
    rows[1, ::3] *= -1.0
    rows[2, 5::17] += np.float32(0.5)
    expected = _libgguf._quantize_tq1_0_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    assert _libgguf._tq1_0_backend() in {"ref", "sse2", "sse4_1", "avx2"}
    for backend in ("sse2", "sse4_1", "avx2"):
        if not _libgguf._tq1_0_cpu_supports_backend(backend):
            continue
        actual = _libgguf._quantize_tq1_0_for_backend(backend, rows, rows.shape[0], rows.shape[1])
        assert actual == expected


def test_libgguf_iq4_nl_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.IQ4_NL
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 5 * info.block_size, dtype=np.float32).reshape(5, info.block_size)
    rows[0] = 0.0
    rows[0, 0] = 0.75
    expected = _libgguf._quantize_iq4_nl_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    for backend in ("sse2", "sse4_1", "avx2"):
        if _libgguf._iq4_nl_cpu_supports_backend(backend):
            actual = _libgguf._quantize_iq4_nl_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_iq4_xs_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.IQ4_XS
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 4 * info.block_size, dtype=np.float32).reshape(4, info.block_size)
    rows[0] = 0.0
    rows[1, ::7] *= -1.5
    rows[2, 3::11] += np.float32(0.25)
    expected = _libgguf._quantize_iq4_xs_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    for backend in ("sse2", "sse4_1", "avx2"):
        if not _libgguf._iq4_xs_cpu_supports_backend(backend):
            continue
        actual = _libgguf._quantize_iq4_xs_for_backend(backend, rows, rows.shape[0], rows.shape[1])
        assert actual == expected


def test_libgguf_common_quant_supported_backends_match_reference() -> None:
    cases = [
        (2, 32, False),
        (2, 32, True),
        (3, 32, True),
        (6, 32, True),
        (7, 32, True),
        (10, 256, False),
        (10, 256, True),
        (11, 256, False),
        (11, 256, True),
        (12, 256, False),
        (12, 256, True),
        (13, 256, False),
        (13, 256, True),
        (14, 256, False),
        (14, 256, True),
    ]

    def run_backend(backend: str) -> list[bytes]:
        _libgguf._common_quant_set_backend(backend)
        assert _libgguf._common_quant_backend() == backend
        parts = []
        for index, (qtype, block, weighted) in enumerate(cases):
            rows = np.linspace(-2.5, 2.5, 3 * block, dtype=np.float32).reshape(3, block)
            rows[0] = 0.0
            rows[0, index % block] = np.float32(1.25)
            rows[1, ::7] *= np.float32(-3.0)
            rows[2, 1::5] += np.float32(0.125)
            imatrix = np.linspace(0.25, 1.75, block, dtype=np.float32) if weighted else None
            parts.append(libgguf.quantize_rows_raw(qtype, rows, rows.shape[0], rows.shape[1], imatrix))
        return parts

    try:
        expected = run_backend("ref")
        for backend in ("sse2", "sse4_1", "avx2"):
            if _libgguf._common_quant_cpu_supports_backend(backend):
                assert run_backend(backend) == expected
    finally:
        _libgguf._common_quant_set_backend("ref")


def test_libgguf_common_small_helpers_match_reference() -> None:
    expected = _libgguf._common_quant_probe_for_backend("ref")

    for backend in ("sse2", "sse4_1", "avx2"):
        if _libgguf._common_quant_cpu_supports_backend(backend):
            assert _libgguf._common_quant_probe_for_backend(backend) == expected


def test_libgguf_builds_do_not_use_global_avx2_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    setup_py = (root / "setup.py").read_text(encoding="utf-8")
    shared_build = (root / "scripts" / "build_libgguf.py").read_text(encoding="utf-8")
    native_sources = (root / "scripts" / "native_sources.py").read_text(encoding="utf-8")

    assert "LIBGGUF_AVX2" not in setup_py
    assert "LIBGGUF_AVX2" not in shared_build
    assert 'name.endswith("_avx2.cpp")' in setup_py
    assert 'name.endswith("_sse2.cpp")' in setup_py
    assert 'name.endswith("_sse4_1.cpp")' in setup_py
    assert 'source_name.endswith("_avx2.cpp")' in shared_build
    assert 'source_name.endswith("_sse2.cpp")' in shared_build
    assert 'source_name.endswith("_sse4_1.cpp")' in shared_build
    assert "DEQUANT_BACKEND_SOURCES" in native_sources
    assert "dequant_generic_sse2.cpp" not in native_sources
    assert "dequant_generic_sse4_1.cpp" not in native_sources
    assert "dequant_generic_avx2.cpp" not in native_sources


def test_libgguf_completed_dequant_simd_sources_do_not_delegate_to_ref() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = (
        "q1_0",
        "q4_1",
        "q5_0",
        "q5_1",
        "q2_k",
        "q3_k",
        "q4_k",
        "q5_k",
        "q6_k",
        "tq1_0",
        "tq2_0",
        "iq2_xxs",
        "iq2_xs",
        "iq2_s",
        "iq3_xxs",
        "iq3_s",
        "iq1_s",
        "iq1_m",
        "iq4_nl",
        "iq4_xs",
        "mxfp4",
        "nvfp4",
    )
    forbidden = (
        "dequant_backend_fallback.h",
        "dequant_simd_simple",
        "LIBGGUF_DEQUANT_DEFINE_BACKEND",
        "_ref(",
    )

    for qtype in completed:
        for backend in ("sse2", "sse4_1", "avx2"):
            source = (root / "csrc" / "quant" / f"dequant_{qtype}_{backend}.cpp").read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in source, f"{qtype}/{backend} delegates instead of using a real backend"


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
