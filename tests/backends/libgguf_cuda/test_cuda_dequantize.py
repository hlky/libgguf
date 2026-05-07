from __future__ import annotations

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES

torch = pytest.importorskip("torch")
libgguf_cuda = pytest.importorskip("libgguf.libgguf_cuda")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


CUDA_DEQUANT_QTYPES = (
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


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def build_rows(qtype: GGMLQuantizationType, *, rows: int = 2) -> np.ndarray:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    width = block_size * (2 if block_size < 256 else 1)
    rng = np.random.default_rng(qtype.value * 29 + rows)
    data = np.stack(
        [
            rng.normal(0.0, 0.75, width).astype(np.float32),
            ((np.arange(width, dtype=np.float32) % 17.0) - 8.0) / np.float32(3.0),
        ]
    )
    return np.ascontiguousarray(data[:rows], dtype=np.float32)


def expected_bf16_bits(rows: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(rows, dtype=np.float32).view(np.uint32)
    high = bits >> np.uint32(16)
    rounded = np.where(
        (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000),
        high | np.uint32(64),
        (bits + (np.uint32(0x7FFF) + (high & np.uint32(1)))) >> np.uint32(16),
    )
    return rounded.astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint32) << np.uint32(16)).view(np.float32)


def bf16_encoded_rows(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        expected_bf16_bits(rows).view(np.uint8).reshape(rows.shape[0], rows.shape[1] * 2)
    )


def require_cuda_extension() -> None:
    if not hasattr(torch.ops, "_C_gguf") or not hasattr(torch.ops._C_gguf, "dequantize"):
        pytest.skip("libgguf CUDA extension is not available")


@pytest.mark.parametrize("qtype", CUDA_DEQUANT_QTYPES, ids=qtype_id)
def test_cuda_dequantize_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    require_cuda_extension()

    rows = build_rows(qtype)
    encoded = libgguf.quantize_rows(rows, qtype)

    actual = libgguf_cuda.dequantize(
        torch.from_numpy(encoded).to("cuda"),
        int(qtype),
        rows.shape[0],
        rows.shape[1],
        torch.float32,
    )
    expected = libgguf.dequantize_rows(encoded, qtype, n_per_row=rows.shape[1])

    np.testing.assert_allclose(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32), ids=str)
def test_cuda_dequantize_returns_requested_dtype(dtype: torch.dtype) -> None:
    require_cuda_extension()

    rows = build_rows(GGMLQuantizationType.Q4_0, rows=1)
    encoded = libgguf.quantize_rows(rows, GGMLQuantizationType.Q4_0)

    actual = libgguf_cuda.dequantize(
        torch.from_numpy(encoded).to("cuda"),
        int(GGMLQuantizationType.Q4_0),
        rows.shape[0],
        rows.shape[1],
        dtype,
    )

    assert actual.device.type == "cuda"
    assert actual.dtype == dtype
    assert tuple(actual.shape) == rows.shape


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32), ids=str)
def test_cuda_dequantize_bf16_storage_returns_requested_dtype(dtype: torch.dtype) -> None:
    require_cuda_extension()

    rows = np.array(
        [
            [0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 1.0e-8],
            [np.inf, -np.inf, np.nan, -3.5, 42.25, -1000.0, 3.1415927],
        ],
        dtype=np.float32,
    )
    encoded = bf16_encoded_rows(rows)

    actual = libgguf_cuda.dequantize(
        torch.from_numpy(encoded).to("cuda"),
        int(GGMLQuantizationType.BF16),
        rows.shape[0],
        rows.shape[1],
        dtype,
    )
    expected = torch.from_numpy(bf16_bits_to_float32(expected_bf16_bits(rows))).to(dtype)

    assert actual.device.type == "cuda"
    assert actual.dtype == dtype
    assert tuple(actual.shape) == rows.shape
    np.testing.assert_allclose(actual.cpu().float().numpy(), expected.float().numpy(), equal_nan=True)


def test_cuda_dequantize_rejects_short_encoded_tensor() -> None:
    require_cuda_extension()

    rows = build_rows(GGMLQuantizationType.Q4_0, rows=1)
    encoded = libgguf.quantize_rows(rows, GGMLQuantizationType.Q4_0)

    with pytest.raises(RuntimeError, match="expected"):
        libgguf_cuda.dequantize(
            torch.from_numpy(encoded.reshape(-1)[:-1]).to("cuda"),
            int(GGMLQuantizationType.Q4_0),
            rows.shape[0],
            rows.shape[1],
            torch.float32,
        )


@pytest.mark.parametrize(
    ("m", "n", "match"),
    ((0, 32, "positive row count"), (1, 0, "positive row width"), (1, 31, "divisible")),
)
def test_cuda_dequantize_rejects_bad_dimensions(m: int, n: int, match: str) -> None:
    require_cuda_extension()

    encoded = torch.empty((18,), device="cuda", dtype=torch.uint8)

    with pytest.raises(RuntimeError, match=match):
        libgguf_cuda.dequantize(encoded, int(GGMLQuantizationType.Q4_0), m, n, torch.float32)


def test_cuda_dequantize_rejects_unsupported_qtype() -> None:
    require_cuda_extension()

    encoded = torch.empty((16,), device="cuda", dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="Unsupported GGML quantization type"):
        libgguf_cuda.dequantize(encoded, int(GGMLQuantizationType.F32), 1, 32, torch.float32)


def test_cuda_dequantize_rejects_unsupported_dtype() -> None:
    require_cuda_extension()

    rows = build_rows(GGMLQuantizationType.Q4_0, rows=1)
    encoded = libgguf.quantize_rows(rows, GGMLQuantizationType.Q4_0)

    with pytest.raises(RuntimeError, match="output dtype"):
        libgguf_cuda.dequantize(
            torch.from_numpy(encoded).to("cuda"),
            int(GGMLQuantizationType.Q4_0),
            rows.shape[0],
            rows.shape[1],
            torch.int32,
        )


def test_cuda_dequantize_rejects_non_contiguous_input() -> None:
    require_cuda_extension()

    row_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0][1]
    encoded = torch.empty((row_size, 2), device="cuda", dtype=torch.uint8)[:, 0]

    assert not encoded.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        libgguf_cuda.dequantize(encoded, int(GGMLQuantizationType.Q4_0), 1, 32, torch.float32)
