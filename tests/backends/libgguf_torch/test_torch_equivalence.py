from __future__ import annotations

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES

torch = pytest.importorskip("torch")
libgguf_torch = pytest.importorskip("libgguf.libgguf_torch")


STORAGE_QTYPES = (
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
)

QUANTIZED_QTYPES = (
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

FAST_QUANTIZE_PARITY_QTYPES = (
    GGMLQuantizationType.BF16,
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
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
)


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def build_rows(qtype: GGMLQuantizationType, *, rows: int = 2) -> np.ndarray:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    width = block_size * (2 if block_size < 256 else 1)
    rng = np.random.default_rng(qtype.value * 23 + rows)
    data = np.stack(
        [
            rng.normal(0.0, 0.75, width).astype(np.float32),
            ((np.arange(width, dtype=np.float32) % 17.0) - 8.0) / np.float32(3.0),
        ]
    )
    return np.ascontiguousarray(data[:rows], dtype=np.float32)


def tensor_bytes(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy()


@pytest.mark.parametrize("qtype", STORAGE_QTYPES, ids=qtype_id)
def test_torch_quantize_storage_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = np.array(
        [
            [0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 1.0e-8],
            [np.inf, -np.inf, np.nan, -3.5, 42.25, -1000.0, 3.1415927],
        ],
        dtype=np.float32,
    )

    actual = libgguf_torch.quantize(torch.from_numpy(rows), qtype)
    expected = libgguf.quantize_rows(rows, qtype)

    np.testing.assert_array_equal(tensor_bytes(actual), expected.view(np.uint8))


@pytest.mark.parametrize("qtype", FAST_QUANTIZE_PARITY_QTYPES, ids=qtype_id)
def test_torch_quantize_bytes_match_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype)

    actual = libgguf_torch.quantize(torch.from_numpy(rows), qtype)
    expected = libgguf.quantize_rows(rows, qtype)

    assert tuple(actual.shape) == expected.shape
    np.testing.assert_array_equal(tensor_bytes(actual), expected.view(np.uint8))


@pytest.mark.parametrize("qtype", QUANTIZED_QTYPES, ids=qtype_id)
def test_torch_dequantize_native_bytes_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype, rows=1)
    encoded = libgguf.quantize_rows(rows, qtype)

    actual = libgguf_torch.dequantize(
        torch.from_numpy(encoded),
        qtype,
        rows.shape,
        dtype=torch.float32,
    )
    expected = libgguf.dequantize_rows(encoded, qtype, n_per_row=rows.shape[-1])

    assert tuple(actual.shape) == rows.shape
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0.0, atol=4.0e-3)


def test_dequantize_tensor_returns_torch_compatible_tensor_in_requested_dtype() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensor.tensor_type = GGMLQuantizationType.F32

    actual = libgguf_torch.dequantize_tensor(tensor, dtype=torch.float16)

    assert actual.dtype == torch.float16
    np.testing.assert_array_equal(actual.cpu().numpy(), tensor.cpu().numpy().astype(np.float16))
