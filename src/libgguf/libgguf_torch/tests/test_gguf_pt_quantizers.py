from __future__ import annotations

import numpy as np
import pytest
import torch

from .accuracy_utils import build_roundtrip_cases

from libgguf import GGMLQuantizationType
from libgguf.libgguf_numpy.gguf_np import _type_traits, quantize as quantize_np, dequantize
from ..gguf_pt import quantize_functions, quantize

NATIVE_QUANTIZE_QTYPES = (
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.Q1_0,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
)

K_QUANTIZE_QTYPES = (
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
)

IQ3_QUANTIZE_QTYPES = (
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
)


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def test_torch_quantize_qtypes_cover_all_supported_formats() -> None:
    expected = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        *_type_traits.keys(),
    }
    actual = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        *quantize_functions.keys(),
    }

    assert actual == expected


@pytest.mark.parametrize(("qtype", "tensor_name"), build_roundtrip_cases(NATIVE_QUANTIZE_QTYPES))
def test_torch_quantize_matches_numpy_reference(
    qtype: GGMLQuantizationType, tensor_name: str
) -> None:
    from .accuracy_utils import REFERENCE_TENSORS

    reference = REFERENCE_TENSORS[tensor_name]
    expected = quantize_np(reference, qtype)

    actual = quantize(torch.from_numpy(reference), qtype)

    assert actual.dtype == torch.uint8
    assert actual.shape == expected.shape
    assert np.array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("qtype", K_QUANTIZE_QTYPES, ids=qtype_id)
def test_torch_quantize_k_quants_match_numpy_reference(
    qtype: GGMLQuantizationType,
) -> None:
    base = torch.linspace(-0.75, 0.75, 256, dtype=torch.float16)
    wave = torch.sin(torch.arange(256, dtype=torch.float32)).to(torch.float16) * 0.01
    reference = (base + wave).reshape(1, 256)
    expected = quantize_np(reference.numpy(), qtype)

    actual = quantize(reference, qtype).numpy()

    assert actual.dtype == np.uint8
    assert actual.shape == expected.shape
    if qtype == GGMLQuantizationType.Q2_K:
        expected_dequant = dequantize(expected, qtype)
        actual_dequant = dequantize(actual, qtype)
        diff = np.abs(expected_dequant - actual_dequant)
        assert float(diff.mean()) <= 7.0e-4
        assert float(diff.max()) <= 1.7e-2
    else:
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("qtype", IQ3_QUANTIZE_QTYPES, ids=qtype_id)
def test_torch_quantize_native_iq_quants_match_numpy_reference(
    qtype: GGMLQuantizationType,
) -> None:
    base = torch.linspace(-0.6, 0.6, 256, dtype=torch.float16)
    wave = torch.cos(torch.arange(256, dtype=torch.float32) * 0.25).to(torch.float16) * 0.015
    reference = (base + wave).reshape(1, 256)
    expected = quantize_np(reference.numpy(), qtype)

    actual = quantize(reference, qtype).numpy()

    assert actual.dtype == np.uint8
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "qtype",
    (GGMLQuantizationType.F32, GGMLQuantizationType.F16),
    ids=qtype_id,
)
def test_torch_quantize_dense_formats(qtype: GGMLQuantizationType) -> None:
    data = torch.linspace(-1, 1, 32, dtype=torch.float16).reshape(1, 32)
    actual = quantize(data, qtype)
    expected_dtype = torch.float32 if qtype == GGMLQuantizationType.F32 else torch.float16

    assert actual.dtype == expected_dtype
    assert torch.equal(actual, data.to(expected_dtype))


def test_torch_quantize_unsupported_format_raises() -> None:
    data = torch.zeros((1, 256), dtype=torch.float16)

    with pytest.raises(NotImplementedError, match="Q8_K"):
        quantize(data, GGMLQuantizationType.Q8_K)
