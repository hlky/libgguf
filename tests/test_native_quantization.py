"""Native CPU bytes are checked against an independent Python encoder."""

import ctypes as ct
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libgguf import native_numpy as reference
from libgguf.native_formats import NATIVE_QUANT_SIZES, NativeQuantizationType as Format


@pytest.fixture(scope="module")
def library():
    path = os.environ.get("LIBGGUF_NATIVE_TEST_LIBRARY")
    if not path:
        pytest.skip("set LIBGGUF_NATIVE_TEST_LIBRARY to the tested native library")
    lib = ct.CDLL(path)
    lib.libgguf_native_encode_rows.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t,
        ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    lib.libgguf_native_decode_rows.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t,
        ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    lib.libgguf_native_validate_rows.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t,
        ct.c_size_t, ct.c_size_t]
    return lib


@pytest.mark.parametrize("format", list(Format))
@pytest.mark.parametrize("weighted", [False, True])
def test_native_bytes_match_independent_encoder(library, format, weighted):
    rng = np.random.default_rng(581)
    rows = rng.normal(size=(3, 512)).astype(np.float32)
    rows[0, :32] = 0
    rows[0, 32:64] *= 0.00001
    rows[0, 64:96] *= 4096
    rows[1, :32] = np.resize(np.array([-8, -7, -1, 0, 1, 7], np.float32), 32)
    rows[2] *= np.float32(2**-24)
    weights = rng.uniform(size=(512,)).astype(np.float32) if weighted else None
    if weights is not None:
        weights[:64] = 0
    expected = reference.quantize_rows(rows, format, weights)
    actual = np.full_like(expected, 0xCC)
    status = library.libgguf_native_encode_rows(format, rows.ctypes.data, rows.nbytes,
        *rows.shape, None if weights is None else weights.ctypes.data,
        0 if weights is None else weights.nbytes, actual.ctypes.data, actual.nbytes)
    assert status == 0
    np.testing.assert_array_equal(actual, expected)
    decoded = np.empty_like(rows)
    status = library.libgguf_native_decode_rows(format, actual.ctypes.data, actual.nbytes,
        *rows.shape, decoded.ctypes.data, decoded.nbytes)
    assert status == 0
    np.testing.assert_array_equal(decoded, reference.dequantize_rows(expected, format, 512))


@pytest.mark.parametrize("format", list(Format))
def test_invalid_scale_or_payload_is_rejected_before_output_write(library, format):
    values, size, _ = NATIVE_QUANT_SIZES[format]
    encoded = np.zeros((1, size), np.uint8)
    encoded[0, -1] = 1  # Nonzero quant under zero scale.
    output = np.full((1, values), 77, np.float32)
    status = library.libgguf_native_decode_rows(format, encoded.ctypes.data, size,
        1, values, output.ctypes.data, output.nbytes)
    assert status == 4
    assert (output == 77).all()
    encoded.fill(0)
    encoded[0, :2] = np.frombuffer(struct.pack("<H", 0x8000), np.uint8)
    assert library.libgguf_native_validate_rows(format, encoded.ctypes.data, size, 1, values) == 4


def test_invalid_input_and_alias_do_not_mutate_output(library):
    rows = np.zeros((1, 32), np.float32)
    rows[0, -1] = np.nan
    output = np.full((18,), 91, np.uint8)
    args = (Format.S4G32, rows.ctypes.data, rows.nbytes, 1, 32, None, 0)
    assert library.libgguf_native_encode_rows(*args, output.ctypes.data, output.nbytes) == 4
    assert (output == 91).all()
    rows.fill(1)
    assert library.libgguf_native_encode_rows(*args, rows.ctypes.data, 18) == 3


def test_hierarchy_zero_codes_and_s8_negative_128_are_invalid(library):
    block = np.zeros((138,), np.uint8)
    block[2] = 1
    assert library.libgguf_native_validate_rows(Format.S4H256, block.ctypes.data, 138, 1, 256) == 4
    block = np.zeros((34,), np.uint8)
    block[:2] = np.frombuffer(struct.pack("<e", 1), np.uint8)
    block[2] = 128
    assert library.libgguf_native_validate_rows(Format.S8G32, block.ctypes.data, 34, 1, 32) == 4


def test_hierarchy_all_nibbles_decode_independently(library):
    block = np.zeros((1, 138), np.uint8)
    block[0, :2] = np.frombuffer(struct.pack("<e", 0.25), np.uint8)
    block[0, 2:10] = [1, 2, 3, 4, 8, 16, 128, 255]
    block[0, 10:] = np.resize(np.arange(8, dtype=np.uint8) * 34 + 16, 128)
    actual = np.empty((1, 256), np.float32)
    assert library.libgguf_native_decode_rows(Format.S4H256, block.ctypes.data, 138,
        1, 256, actual.ctypes.data, actual.nbytes) == 0
    np.testing.assert_array_equal(actual, reference.dequantize_rows(block, Format.S4H256, 256))
