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


@pytest.mark.parametrize(
    ("qtype", "env_name"),
    [
        (GGMLQuantizationType.Q4_0, "LIBGGUF_DEQUANT_Q4_0_BACKEND"),
        (GGMLQuantizationType.Q8_0, "LIBGGUF_DEQUANT_Q8_0_BACKEND"),
    ],
)
def test_libgguf_dequant_simd_backends_match_reference(qtype: GGMLQuantizationType, env_name: str) -> None:
    child = f"""
import hashlib
import numpy as np
import libgguf

qtype = {int(qtype)}
rows = np.linspace(-2.0, 2.0, 257 * 32, dtype=np.float32).reshape(257, 32)
quantized = libgguf.quantize_rows(rows, qtype)
dequantized = libgguf.dequantize_rows(quantized, qtype)
print(hashlib.sha256(dequantized.tobytes()).hexdigest())
"""
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env[env_name] = "ref"
    expected = subprocess.run(
        [sys.executable, "-c", child],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()

    for backend in ("sse2", "avx2"):
        env[env_name] = backend
        actual = subprocess.run(
            [sys.executable, "-c", child],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
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

    assert backend in {"ref", "sse2", "avx2"}
    assert _libgguf._q4_0_cpu_supports_backend(backend)


def test_libgguf_q4_0_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.Q4_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 7 * info.block_size, dtype=np.float32).reshape(7, info.block_size)
    rows[0] = 0.0
    rows[0, 1] = 3.0
    rows[0, 4] = -3.0
    expected = _libgguf._quantize_q4_0_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    for backend in ("sse2", "avx2"):
        if _libgguf._q4_0_cpu_supports_backend(backend):
            actual = _libgguf._quantize_q4_0_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_q4_0_forced_ref_backend_via_environment() -> None:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["LIBGGUF_Q4_0_BACKEND"] = "ref"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from libgguf import _libgguf; print(_libgguf._q4_0_backend())",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "ref"


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

    assert backend in {"ref", "sse2", "avx2"}
    assert _libgguf._q8_0_cpu_supports_backend(backend)


def test_libgguf_q8_0_supported_backends_match_reference() -> None:
    qtype = GGMLQuantizationType.Q8_0
    info = GGML_FORMAT_INFO[qtype]
    rows = np.linspace(-2.0, 2.0, 7 * info.block_size, dtype=np.float32).reshape(7, info.block_size)
    expected = _libgguf._quantize_q8_0_for_backend("ref", rows, rows.shape[0], rows.shape[1])

    for backend in ("sse2", "avx2"):
        if _libgguf._q8_0_cpu_supports_backend(backend):
            actual = _libgguf._quantize_q8_0_for_backend(backend, rows, rows.shape[0], rows.shape[1])
            assert actual == expected


def test_libgguf_q8_0_forced_ref_backend_via_environment() -> None:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["LIBGGUF_Q8_0_BACKEND"] = "ref"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from libgguf import _libgguf; print(_libgguf._q8_0_backend())",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "ref"


def test_libgguf_builds_do_not_use_global_avx2_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    setup_py = (root / "setup.py").read_text(encoding="utf-8")
    shared_build = (root / "scripts" / "build_libgguf.py").read_text(encoding="utf-8")

    assert "LIBGGUF_AVX2" not in setup_py
    assert "LIBGGUF_AVX2" not in shared_build
    for source in (
        "dequant_q4_0_avx2.cpp",
        "dequant_q8_0_avx2.cpp",
        "quant_q4_0_avx2.cpp",
        "quant_q8_0_avx2.cpp",
        "dequant_q4_0_sse2.cpp",
        "dequant_q8_0_sse2.cpp",
        "quant_q4_0_sse2.cpp",
        "quant_q8_0_sse2.cpp",
    ):
        assert source in setup_py
        assert source in shared_build


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
