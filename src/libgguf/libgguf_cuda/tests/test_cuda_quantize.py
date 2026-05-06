from __future__ import annotations

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES

torch = pytest.importorskip("torch")
libgguf_cuda = pytest.importorskip("libgguf.libgguf_cuda")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


CUDA_QUANT_QTYPES = (
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.MXFP4,
    GGMLQuantizationType.NVFP4,
    GGMLQuantizationType.Q1_0,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
)


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def build_rows(qtype: GGMLQuantizationType, *, rows: int = 4) -> np.ndarray:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    width = block_size * 4
    rng = np.random.default_rng(qtype.value * 31 + rows)
    data = rng.normal(0.0, 0.75, size=(rows, width)).astype(np.float32)
    data += ((np.arange(width, dtype=np.float32) % 17.0) - 8.0) / np.float32(6.0)
    return np.ascontiguousarray(data, dtype=np.float32)


def require_cuda_quantize() -> None:
    if not hasattr(torch.ops, "_C_gguf") or not hasattr(torch.ops._C_gguf, "quantize"):
        pytest.skip("libgguf CUDA quantize extension is not available")


def make_imatrix(rows: np.ndarray) -> np.ndarray | None:
    width = rows.shape[-1]
    qtype_rows = rows.reshape(-1, width)
    return np.ascontiguousarray(np.sum(qtype_rows * qtype_rows, axis=0, dtype=np.float32))


def quantize_pair(rows: np.ndarray, qtype: GGMLQuantizationType) -> tuple[torch.Tensor, np.ndarray]:
    imatrix_np = make_imatrix(rows) if libgguf.quantize_requires_imatrix(qtype) else None
    imatrix = torch.from_numpy(imatrix_np).to("cuda") if imatrix_np is not None else None
    actual = libgguf_cuda.quantize(torch.from_numpy(rows).to("cuda"), int(qtype), imatrix)
    expected = libgguf.quantize_rows(rows, qtype, imatrix=imatrix_np)
    return actual, expected


@pytest.mark.parametrize("qtype", CUDA_QUANT_QTYPES, ids=qtype_id)
def test_cuda_quantize_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    require_cuda_quantize()

    rows = build_rows(qtype)
    actual, expected = quantize_pair(rows, qtype)

    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def test_cuda_quantize_preserves_leading_shape() -> None:
    require_cuda_quantize()

    rows = build_rows(GGMLQuantizationType.Q8_0, rows=6).reshape(2, 3, -1)
    actual = libgguf_cuda.quantize(torch.from_numpy(rows).to("cuda"), int(GGMLQuantizationType.Q8_0))
    expected = libgguf.quantize_rows(rows, GGMLQuantizationType.Q8_0)

    assert actual.device.type == "cuda"
    assert actual.dtype == torch.uint8
    assert tuple(actual.shape) == expected.shape
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize(
    "qtype",
    (GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K),
)
@pytest.mark.parametrize("case", ("constant", "adversarial"))
def test_cuda_quantize_k_edge_cases_match_libgguf(qtype: GGMLQuantizationType, case: str) -> None:
    require_cuda_quantize()

    if case == "constant":
        rows = np.full((4, 1024), 0.125, dtype=np.float32)
    else:
        row = np.linspace(-8.0, 8.0, 1024, dtype=np.float32)
        row[::17] = -32.0
        row[5::19] = 32.0
        rows = np.ascontiguousarray(np.vstack([row, -row, row[::-1], -row[::-1]]), dtype=np.float32)
    actual, expected = quantize_pair(rows, qtype)

    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize(
    "qtype",
    (
        GGMLQuantizationType.IQ2_XXS,
        GGMLQuantizationType.IQ2_XS,
        GGMLQuantizationType.IQ2_S,
        GGMLQuantizationType.IQ3_XXS,
        GGMLQuantizationType.IQ3_S,
        GGMLQuantizationType.IQ1_S,
        GGMLQuantizationType.IQ1_M,
        GGMLQuantizationType.IQ4_NL,
        GGMLQuantizationType.IQ4_XS,
        GGMLQuantizationType.MXFP4,
        GGMLQuantizationType.NVFP4,
        GGMLQuantizationType.Q1_0,
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    ),
    ids=qtype_id,
)
def test_cuda_quantize_large_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    require_cuda_quantize()

    rng = np.random.default_rng(0)
    rows = rng.standard_normal((512, 4096), dtype=np.float32)
    actual, expected = quantize_pair(rows, qtype)

    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
