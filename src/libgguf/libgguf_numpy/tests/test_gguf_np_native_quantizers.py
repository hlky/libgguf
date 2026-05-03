from __future__ import annotations

import numpy as np
import pytest

from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES, quant_shape_to_byte_shape, quantize_rows
from ..gguf_np import _type_traits, quantize


NATIVE_REFERENCE_QTYPES = (
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
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
)


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def test_native_reference_qtypes_cover_all_libgguf_supported_native_quantizers() -> None:
    expected = set(_type_traits) - {GGMLQuantizationType.BF16}

    assert set(NATIVE_REFERENCE_QTYPES) == expected


def build_reference_rows(qtype: GGMLQuantizationType) -> np.ndarray:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    width = block_size * (2 if block_size < 256 else 1)
    rng = np.random.default_rng(0)
    return np.vstack(
        [
            np.linspace(-1.5, 1.5, width, dtype=np.float32),
            rng.normal(0.0, 1.0, width).astype(np.float32),
            (rng.normal(0.0, 1.0, width) * 50.0).astype(np.float32),
            np.zeros(width, dtype=np.float32),
            np.full(width, 0.25, dtype=np.float32),
            ((np.arange(width, dtype=np.float32) % 17.0) - 8.0) / 3.0,
        ]
    )


@pytest.mark.parametrize("qtype", NATIVE_REFERENCE_QTYPES, ids=qtype_id)
def test_native_quantizers_match_libgguf_reference(
    qtype: GGMLQuantizationType,
) -> None:
    rows = build_reference_rows(qtype)

    expected = quantize_rows(rows, qtype)
    quantized = quantize(rows, qtype)

    assert np.array_equal(quantized, expected)
    assert quantized.shape == quant_shape_to_byte_shape(rows.shape, qtype)
