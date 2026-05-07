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
