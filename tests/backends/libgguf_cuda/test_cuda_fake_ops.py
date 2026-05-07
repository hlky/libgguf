from __future__ import annotations

from contextlib import nullcontext

import pytest

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES

torch = pytest.importorskip("torch")
libgguf_cuda = pytest.importorskip("libgguf.libgguf_cuda")

try:
    from torch._subclasses.fake_tensor import FakeTensorMode
except ImportError:  # pragma: no cover - depends on torch version
    FakeTensorMode = None


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
CUDA_IMATRIX_QTYPES = tuple(
    qtype for qtype in CUDA_QUANT_QTYPES if libgguf.quantize_requires_imatrix(qtype)
)
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
    GGMLQuantizationType.BF16,
)


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def require_cuda_ops() -> None:
    if not hasattr(torch.ops, "_C_gguf"):
        pytest.skip("libgguf CUDA extension is not available")
    if not hasattr(torch.ops._C_gguf, "quantize"):
        pytest.skip("libgguf CUDA quantize extension is not available")
    if not hasattr(torch.ops._C_gguf, "dequantize"):
        pytest.skip("libgguf CUDA dequantize extension is not available")


def require_fake_tensor_mode() -> type[FakeTensorMode]:
    if FakeTensorMode is None:
        pytest.skip("torch FakeTensorMode is not available")
    return FakeTensorMode


def quantize_imatrix(
    qtype: GGMLQuantizationType, width: int, device: str
) -> torch.Tensor | None:
    if not libgguf.quantize_requires_imatrix(qtype):
        return None
    return torch.empty((width,), device=device, dtype=torch.float32)


def quantize_validation_context(mode: str):
    if mode == "meta":
        return nullcontext(), "meta"
    fake_tensor_mode = require_fake_tensor_mode()
    return fake_tensor_mode(), "cuda"


@pytest.mark.parametrize("qtype", CUDA_QUANT_QTYPES, ids=qtype_id)
def test_cuda_quantize_meta_shape_matches_quant_row_size(qtype: GGMLQuantizationType) -> None:
    require_cuda_ops()

    block_size, type_size = GGML_QUANT_SIZES[qtype]
    width = block_size * 3
    rows = torch.empty((2, 3, width), device="meta", dtype=torch.float32)
    imatrix = quantize_imatrix(qtype, width, "meta")

    actual = libgguf_cuda.quantize(rows, int(qtype), imatrix)

    assert actual.device.type == "meta"
    assert actual.dtype == torch.uint8
    assert tuple(actual.shape) == (2, 3, width * type_size // block_size)


def test_cuda_quantize_fake_preserves_cuda_device_dtype_and_shape() -> None:
    require_cuda_ops()
    fake_tensor_mode = require_fake_tensor_mode()

    qtype = GGMLQuantizationType.Q4_0
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    width = block_size * 2

    with fake_tensor_mode():
        rows = torch.empty((4, width), device="cuda", dtype=torch.float32)
        actual = libgguf_cuda.quantize(rows, int(qtype))

    assert actual.device.type == "cuda"
    assert actual.dtype == torch.uint8
    assert tuple(actual.shape) == (4, width * type_size // block_size)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_scalar_input(mode: str) -> None:
    require_cuda_ops()

    context, device = quantize_validation_context(mode)
    with context:
        scalar = torch.empty((), device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="at least one dimension"):
            libgguf_cuda.quantize(scalar, int(GGMLQuantizationType.Q4_0))


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_bad_input_dtype(mode: str) -> None:
    require_cuda_ops()

    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, 32), device=device, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="float32 input"):
            libgguf_cuda.quantize(rows, int(GGMLQuantizationType.Q4_0))


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_bad_row_width(mode: str) -> None:
    require_cuda_ops()

    block_size, _ = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]
    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, block_size + 1), device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="divisible"):
            libgguf_cuda.quantize(rows, int(GGMLQuantizationType.Q4_0))


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_unsupported_qtype(mode: str) -> None:
    require_cuda_ops()

    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, 32), device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Unsupported GGML quantization type"):
            libgguf_cuda.quantize(rows, int(GGMLQuantizationType.F32))


@pytest.mark.parametrize("qtype", CUDA_IMATRIX_QTYPES, ids=qtype_id)
@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_missing_required_imatrix(
    mode: str, qtype: GGMLQuantizationType
) -> None:
    require_cuda_ops()

    block_size, _ = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, block_size), device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="requires imatrix"):
            libgguf_cuda.quantize(rows, int(qtype))


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_required_cpu_imatrix(mode: str) -> None:
    require_cuda_ops()

    qtype = CUDA_IMATRIX_QTYPES[0]
    block_size, _ = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, block_size), device=device, dtype=torch.float32)
        imatrix = torch.empty((block_size,), device="cpu", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="imatrix must be a CUDA tensor"):
            libgguf_cuda.quantize(rows, int(qtype), imatrix)


def test_cuda_quantize_fake_rejects_required_imatrix_on_different_device() -> None:
    require_cuda_ops()
    fake_tensor_mode = require_fake_tensor_mode()

    qtype = CUDA_IMATRIX_QTYPES[0]
    block_size, _ = GGML_QUANT_SIZES[qtype]
    with fake_tensor_mode():
        rows = torch.empty((1, block_size), device="cuda", dtype=torch.float32)
        imatrix = torch.empty((block_size,), device="meta", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="same device as input"):
            libgguf_cuda.quantize(rows, int(qtype), imatrix)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_required_imatrix_bad_dtype(mode: str) -> None:
    require_cuda_ops()

    qtype = CUDA_IMATRIX_QTYPES[0]
    block_size, _ = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, block_size), device=device, dtype=torch.float32)
        imatrix = torch.empty((block_size,), device=device, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="imatrix must be float32"):
            libgguf_cuda.quantize(rows, int(qtype), imatrix)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_required_imatrix_bad_rank(mode: str) -> None:
    require_cuda_ops()

    qtype = CUDA_IMATRIX_QTYPES[0]
    block_size, _ = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, block_size), device=device, dtype=torch.float32)
        imatrix = torch.empty((1, block_size), device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="one-dimensional"):
            libgguf_cuda.quantize(rows, int(qtype), imatrix)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_required_imatrix_short_length(
    mode: str,
) -> None:
    require_cuda_ops()

    qtype = CUDA_IMATRIX_QTYPES[0]
    block_size, _ = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, block_size * 2), device=device, dtype=torch.float32)
        imatrix = torch.empty((block_size,), device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="at least input width elements"):
            libgguf_cuda.quantize(rows, int(qtype), imatrix)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_quantize_fake_meta_rejects_required_imatrix_non_contiguous(
    mode: str,
) -> None:
    require_cuda_ops()

    qtype = CUDA_IMATRIX_QTYPES[0]
    block_size, _ = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        rows = torch.empty((1, block_size), device=device, dtype=torch.float32)
        imatrix = torch.empty((block_size, 2), device=device, dtype=torch.float32)[:, 0]
        assert not imatrix.is_contiguous()
        with pytest.raises(RuntimeError, match="imatrix must be contiguous"):
            libgguf_cuda.quantize(rows, int(qtype), imatrix)


@pytest.mark.parametrize("qtype", CUDA_IMATRIX_QTYPES, ids=qtype_id)
def test_cuda_quantize_fake_accepts_required_imatrix(qtype: GGMLQuantizationType) -> None:
    require_cuda_ops()
    fake_tensor_mode = require_fake_tensor_mode()

    block_size, type_size = GGML_QUANT_SIZES[qtype]
    width = block_size * 2

    with fake_tensor_mode():
        rows = torch.empty((3, width), device="cuda", dtype=torch.float32)
        imatrix = torch.empty((width,), device="cuda", dtype=torch.float32)
        actual = libgguf_cuda.quantize(rows, int(qtype), imatrix)

    assert actual.device.type == "cuda"
    assert actual.dtype == torch.uint8
    assert tuple(actual.shape) == (3, width * type_size // block_size)


@pytest.mark.parametrize("qtype", CUDA_DEQUANT_QTYPES, ids=qtype_id)
@pytest.mark.parametrize("dtype", (None, torch.float32, torch.float16, torch.bfloat16))
def test_cuda_dequantize_meta_dtype_device_and_shape(
    dtype: torch.dtype | None, qtype: GGMLQuantizationType
) -> None:
    require_cuda_ops()

    block_size, row_size = GGML_QUANT_SIZES[qtype]
    encoded = torch.empty((5, row_size * 2), device="meta", dtype=torch.uint8)

    actual = libgguf_cuda.dequantize(encoded, int(qtype), 5, block_size * 2, dtype)

    assert actual.device.type == "meta"
    assert actual.dtype == (dtype or torch.float16)
    assert tuple(actual.shape) == (5, block_size * 2)


def test_cuda_dequantize_fake_defaults_dtype_to_float16() -> None:
    require_cuda_ops()
    fake_tensor_mode = require_fake_tensor_mode()

    qtype = GGMLQuantizationType.Q4_0
    _, row_size = GGML_QUANT_SIZES[qtype]

    with fake_tensor_mode():
        encoded = torch.empty((2, row_size), device="cuda", dtype=torch.uint8)
        actual = libgguf_cuda.dequantize(encoded, int(qtype), 2, 32, None)

    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float16
    assert tuple(actual.shape) == (2, 32)


def test_cuda_dequantize_fake_respects_explicit_dtype() -> None:
    require_cuda_ops()
    fake_tensor_mode = require_fake_tensor_mode()

    qtype = GGMLQuantizationType.Q4_0
    _, row_size = GGML_QUANT_SIZES[qtype]

    with fake_tensor_mode():
        encoded = torch.empty((2, row_size), device="cuda", dtype=torch.uint8)
        actual = libgguf_cuda.dequantize(encoded, int(qtype), 2, 32, torch.float32)

    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    assert tuple(actual.shape) == (2, 32)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_dequantize_fake_meta_rejects_bad_input_dtype(mode: str) -> None:
    require_cuda_ops()

    qtype = GGMLQuantizationType.Q4_0
    _, row_size = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        encoded = torch.empty((2, row_size), device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="uint8 input"):
            libgguf_cuda.dequantize(encoded, int(qtype), 2, 32, None)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_dequantize_fake_meta_rejects_bad_output_dtype(mode: str) -> None:
    require_cuda_ops()

    qtype = GGMLQuantizationType.Q4_0
    _, row_size = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        encoded = torch.empty((2, row_size), device=device, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="float16, bfloat16, or float32"):
            libgguf_cuda.dequantize(encoded, int(qtype), 2, 32, torch.int32)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_dequantize_fake_meta_rejects_unsupported_qtype(mode: str) -> None:
    require_cuda_ops()

    context, device = quantize_validation_context(mode)
    with context:
        encoded = torch.empty((2, 128), device=device, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="Unsupported GGML quantization type"):
            libgguf_cuda.dequantize(encoded, int(GGMLQuantizationType.F32), 2, 32, None)


@pytest.mark.parametrize(
    "m,n,match", ((0, 32, "positive row count"), (2, 0, "positive row width"))
)
@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_dequantize_fake_meta_rejects_invalid_shape_args(
    mode: str, m: int, n: int, match: str
) -> None:
    require_cuda_ops()

    qtype = GGMLQuantizationType.Q4_0
    _, row_size = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        encoded = torch.empty((2, row_size), device=device, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match=match):
            libgguf_cuda.dequantize(encoded, int(qtype), m, n, None)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_dequantize_fake_meta_rejects_bad_output_width(mode: str) -> None:
    require_cuda_ops()

    qtype = GGMLQuantizationType.Q4_0
    block_size, row_size = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        encoded = torch.empty((2, row_size), device=device, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="divisible"):
            libgguf_cuda.dequantize(encoded, int(qtype), 2, block_size + 1, None)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_dequantize_fake_meta_rejects_row_size_mismatch(mode: str) -> None:
    require_cuda_ops()

    qtype = GGMLQuantizationType.Q4_0
    _, row_size = GGML_QUANT_SIZES[qtype]
    context, device = quantize_validation_context(mode)
    with context:
        encoded = torch.empty((2, row_size + 1), device=device, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match=r"input has .* bytes, expected"):
            libgguf_cuda.dequantize(encoded, int(qtype), 2, 32, None)


@pytest.mark.parametrize("mode", ("meta", "fake"))
def test_cuda_dequantize_fake_meta_rejects_scalar_input(mode: str) -> None:
    require_cuda_ops()

    qtype = GGMLQuantizationType.BF16
    context, device = quantize_validation_context(mode)
    with context:
        encoded = torch.empty((), device=device, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match=r"input has .* bytes, expected"):
            libgguf_cuda.dequantize(encoded, int(qtype), 1, 1, None)


def test_cuda_dequantize_fake_rejects_cpu_input() -> None:
    require_cuda_ops()
    fake_tensor_mode = require_fake_tensor_mode()

    qtype = GGMLQuantizationType.Q4_0
    _, row_size = GGML_QUANT_SIZES[qtype]

    with fake_tensor_mode():
        encoded = torch.empty((2, row_size), device="cpu", dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="expects a CUDA tensor"):
            libgguf_cuda.dequantize(encoded, int(qtype), 2, 32, None)
