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


def cuda_rows(qtype: GGMLQuantizationType, *, rows: int = 1) -> torch.Tensor:
    return torch.from_numpy(build_rows(qtype, rows=rows)).to("cuda")


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


def test_cuda_quantize_rejects_bad_input_dtype() -> None:
    require_cuda_quantize()

    rows = cuda_rows(GGMLQuantizationType.Q4_0).half()

    with pytest.raises(RuntimeError, match="float32 input"):
        libgguf_cuda.quantize(rows, int(GGMLQuantizationType.Q4_0))


def test_cuda_quantize_rejects_cpu_input() -> None:
    require_cuda_quantize()

    rows = torch.from_numpy(build_rows(GGMLQuantizationType.Q4_0, rows=1))

    with pytest.raises(RuntimeError, match="CPU.*backend"):
        libgguf_cuda.quantize(rows, int(GGMLQuantizationType.Q4_0))


def test_cuda_quantize_rejects_scalar_input() -> None:
    require_cuda_quantize()

    scalar = torch.tensor(0.0, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="at least one dimension"):
        libgguf_cuda.quantize(scalar, int(GGMLQuantizationType.Q4_0))


def test_cuda_quantize_rejects_non_contiguous_input() -> None:
    require_cuda_quantize()

    block_size, _ = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]
    rows = torch.empty((1, block_size * 2, 2), device="cuda", dtype=torch.float32)[:, :, 0]

    assert not rows.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous input"):
        libgguf_cuda.quantize(rows, int(GGMLQuantizationType.Q4_0))


def test_cuda_quantize_rejects_bad_row_width() -> None:
    require_cuda_quantize()

    block_size, _ = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]
    rows = torch.empty((1, block_size + 1), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="divisible"):
        libgguf_cuda.quantize(rows, int(GGMLQuantizationType.Q4_0))


def test_cuda_quantize_rejects_unsupported_qtype() -> None:
    require_cuda_quantize()

    rows = cuda_rows(GGMLQuantizationType.Q4_0)

    with pytest.raises(RuntimeError, match="Unsupported GGML quantization type"):
        libgguf_cuda.quantize(rows, int(GGMLQuantizationType.F32))


def test_cuda_quantize_rejects_missing_required_imatrix() -> None:
    require_cuda_quantize()

    qtype = GGMLQuantizationType.IQ2_XXS
    assert libgguf.quantize_requires_imatrix(qtype)

    with pytest.raises(RuntimeError, match="requires imatrix"):
        libgguf_cuda.quantize(cuda_rows(qtype), int(qtype))


def test_cuda_quantize_rejects_bad_imatrix_dtype() -> None:
    require_cuda_quantize()

    qtype = GGMLQuantizationType.IQ2_XXS
    rows = cuda_rows(qtype)
    imatrix = torch.empty((rows.shape[-1],), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="imatrix must be float32"):
        libgguf_cuda.quantize(rows, int(qtype), imatrix)


def test_cuda_quantize_rejects_bad_imatrix_rank() -> None:
    require_cuda_quantize()

    qtype = GGMLQuantizationType.IQ2_XXS
    rows = cuda_rows(qtype)
    imatrix = torch.empty((1, rows.shape[-1]), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="one-dimensional"):
        libgguf_cuda.quantize(rows, int(qtype), imatrix)


def test_cuda_quantize_rejects_short_imatrix() -> None:
    require_cuda_quantize()

    qtype = GGMLQuantizationType.IQ2_XXS
    rows = cuda_rows(qtype)
    imatrix = torch.empty((rows.shape[-1] - 1,), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="at least input width"):
        libgguf_cuda.quantize(rows, int(qtype), imatrix)


def test_cuda_quantize_rejects_non_contiguous_imatrix() -> None:
    require_cuda_quantize()

    qtype = GGMLQuantizationType.IQ2_XXS
    rows = cuda_rows(qtype)
    imatrix = torch.empty((rows.shape[-1], 2), device="cuda", dtype=torch.float32)[:, 0]

    assert not imatrix.is_contiguous()
    with pytest.raises(RuntimeError, match="imatrix must be contiguous"):
        libgguf_cuda.quantize(rows, int(qtype), imatrix)


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
        GGMLQuantizationType.TQ1_0,
        GGMLQuantizationType.TQ2_0,
    ),
    ids=qtype_id,
)
def test_cuda_quantize_large_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    require_cuda_quantize()

    rng = np.random.default_rng(0)
    rows = rng.standard_normal((512, 4096), dtype=np.float32)
    actual, expected = quantize_pair(rows, qtype)

    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
