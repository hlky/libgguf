from __future__ import annotations

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES


STORAGE_QTYPES = (
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
)

SUPPORTED_QUANTIZED_QTYPES = (
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

UNSUPPORTED_ROW_QTYPES = (
    GGMLQuantizationType.Q8_1,
    GGMLQuantizationType.Q8_K,
    GGMLQuantizationType.I8,
    GGMLQuantizationType.I16,
    GGMLQuantizationType.I32,
    GGMLQuantizationType.I64,
    GGMLQuantizationType.F64,
)

NON_STORAGE_QTYPES = (*SUPPORTED_QUANTIZED_QTYPES, *UNSUPPORTED_ROW_QTYPES)
NON_QUANTIZED_QTYPES = (*STORAGE_QTYPES, *UNSUPPORTED_ROW_QTYPES)


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def build_rows(qtype: GGMLQuantizationType) -> np.ndarray:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    width = block_size * (2 if block_size < 256 else 1)
    rng = np.random.default_rng(qtype.value)
    rows = np.stack(
        [
            np.linspace(-1.5, 1.5, width, dtype=np.float32),
            rng.normal(0.0, 0.75, width).astype(np.float32),
            np.zeros(width, dtype=np.float32),
            np.full(width, 0.25, dtype=np.float32),
            ((np.arange(width, dtype=np.float32) % 17.0) - 8.0) / np.float32(3.0),
            (rng.normal(0.0, 1.0, width) * np.float32(8.0)).astype(np.float32),
        ]
    )
    return rows.reshape(2, 3, width)


def expected_bf16_bits(rows: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(rows, dtype=np.float32).view(np.uint32)
    high = bits >> np.uint32(16)
    rounded = np.where(
        (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000),
        high | np.uint32(64),
        (bits + (np.uint32(0x7FFF) + (high & np.uint32(1)))) >> np.uint32(16),
    )
    return rounded.astype(np.uint16)


def test_qtype_groups_cover_all_public_quantization_types() -> None:
    groups = (
        set(STORAGE_QTYPES),
        set(SUPPORTED_QUANTIZED_QTYPES),
        set(UNSUPPORTED_ROW_QTYPES),
    )

    assert all(left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1 :])
    assert set().union(*groups) == set(GGMLQuantizationType)


@pytest.mark.parametrize("qtype", SUPPORTED_QUANTIZED_QTYPES, ids=qtype_id)
def test_quantize_and_dequantize_rows_cover_supported_quantized_types(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype)
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    expected_byte_width = rows.shape[-1] // block_size * type_size

    quantized = libgguf.quantize_rows(rows, qtype)

    assert quantized.dtype == np.uint8
    assert quantized.shape == (*rows.shape[:-1], expected_byte_width)

    implicit = libgguf.dequantize_rows(quantized, qtype)
    explicit = libgguf.dequantize_rows(quantized, qtype, n_per_row=rows.shape[-1])

    assert implicit.dtype == np.float32
    assert implicit.shape == rows.shape
    assert np.all(np.isfinite(implicit))
    np.testing.assert_array_equal(explicit, implicit)


@pytest.mark.parametrize("qtype", SUPPORTED_QUANTIZED_QTYPES, ids=qtype_id)
def test_quantize_rows_auto_imatrix_matches_explicit_computed_weights(qtype: GGMLQuantizationType) -> None:
    if not libgguf.quantize_requires_imatrix(qtype):
        pytest.skip(f"{qtype.name} does not require an imatrix")

    rows = build_rows(qtype)
    weights = np.sum(rows.reshape((-1, rows.shape[-1])) ** np.float32(2.0), axis=0, dtype=np.float32)

    implicit = libgguf.quantize_rows(rows, qtype)
    explicit = libgguf.quantize_rows(rows, qtype, imatrix=weights)

    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize("qtype", STORAGE_QTYPES, ids=qtype_id)
def test_store_rows_covers_storage_types(qtype: GGMLQuantizationType) -> None:
    rows = np.array(
        [
            [0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 1.0e-8],
            [np.inf, -np.inf, np.nan, -3.5, 42.25, -1000.0, 3.1415927],
        ],
        dtype=np.float32,
    )

    stored = libgguf.store_rows(rows, qtype)

    if qtype == GGMLQuantizationType.F32:
        assert stored.dtype == np.float32
        assert stored.shape == rows.shape
        np.testing.assert_array_equal(stored.view(np.uint32), rows.view(np.uint32))
    elif qtype == GGMLQuantizationType.F16:
        assert stored.dtype == np.float16
        assert stored.shape == rows.shape
        np.testing.assert_array_equal(stored.view(np.uint16), rows.astype(np.float16).view(np.uint16))
    else:
        assert stored.dtype == np.uint8
        assert stored.shape == (*rows.shape[:-1], rows.shape[-1] * 2)
        np.testing.assert_array_equal(stored.view(np.uint16).reshape(rows.shape), expected_bf16_bits(rows))

    quantized = libgguf.quantize_rows(rows, qtype)
    np.testing.assert_array_equal(quantized, stored)


@pytest.mark.parametrize("qtype", NON_STORAGE_QTYPES, ids=qtype_id)
def test_store_rows_rejects_non_storage_types(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype)

    with pytest.raises(ValueError, match="store_rows only supports"):
        libgguf.store_rows(rows, qtype)


@pytest.mark.parametrize("qtype", UNSUPPORTED_ROW_QTYPES, ids=qtype_id)
def test_quantize_rows_rejects_unsupported_row_types(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype)

    with pytest.raises(ValueError, match="unsupported quantization type or row width"):
        libgguf.quantize_rows(rows, qtype)


@pytest.mark.parametrize("qtype", NON_QUANTIZED_QTYPES, ids=qtype_id)
def test_dequantize_rows_without_n_per_row_rejects_non_quantized_types(qtype: GGMLQuantizationType) -> None:
    _, type_size = GGML_QUANT_SIZES[qtype]
    encoded = np.zeros((2, type_size), dtype=np.uint8)

    with pytest.raises(ValueError):
        libgguf.dequantize_rows(encoded, qtype)
