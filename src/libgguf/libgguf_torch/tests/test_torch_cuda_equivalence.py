from __future__ import annotations

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType

torch = pytest.importorskip("torch")
libgguf_torch = pytest.importorskip("libgguf.libgguf_torch")

from .test_torch_equivalence import (  # noqa: E402
    FAST_QUANTIZE_PARITY_QTYPES,
    QUANTIZED_QTYPES,
    STORAGE_QTYPES,
    build_rows,
    qtype_id,
    tensor_bytes,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

COMPILED_QUANTIZE_QTYPES = (
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.NVFP4,
)

COMPILED_DEQUANTIZE_QTYPES = (
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.TQ2_0,
    GGMLQuantizationType.NVFP4,
)


def compile_or_skip(fn):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")
    try:
        return torch.compile(fn, dynamic=False)
    except Exception as exc:
        pytest.skip(f"torch.compile failed: {exc}")


@pytest.mark.parametrize("qtype", STORAGE_QTYPES, ids=qtype_id)
def test_cuda_torch_quantize_storage_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = np.array(
        [
            [0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 1.0e-8],
            [np.inf, -np.inf, np.nan, -3.5, 42.25, -1000.0, 3.1415927],
        ],
        dtype=np.float32,
    )

    actual = libgguf_torch.quantize(torch.from_numpy(rows).to("cuda"), qtype)
    expected = libgguf.quantize_rows(rows, qtype)

    np.testing.assert_array_equal(tensor_bytes(actual), expected.view(np.uint8))


@pytest.mark.parametrize("qtype", FAST_QUANTIZE_PARITY_QTYPES, ids=qtype_id)
def test_cuda_torch_quantize_bytes_match_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype, rows=1)

    actual = libgguf_torch.quantize(torch.from_numpy(rows).to("cuda"), qtype)
    expected = libgguf.quantize_rows(rows, qtype)

    assert tuple(actual.shape) == expected.shape
    np.testing.assert_array_equal(tensor_bytes(actual), expected.view(np.uint8))


@pytest.mark.parametrize("qtype", COMPILED_QUANTIZE_QTYPES, ids=qtype_id)
def test_compiled_cuda_torch_quantize_bytes_match_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype, rows=1)
    quantize = compile_or_skip(lambda data: libgguf_torch.quantize(data, qtype))

    actual = quantize(torch.from_numpy(rows).to("cuda"))
    expected = libgguf.quantize_rows(rows, qtype)

    assert tuple(actual.shape) == expected.shape
    np.testing.assert_array_equal(tensor_bytes(actual), expected.view(np.uint8))


@pytest.mark.parametrize("qtype", QUANTIZED_QTYPES, ids=qtype_id)
def test_cuda_torch_dequantize_native_bytes_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype, rows=1)
    encoded = libgguf.quantize_rows(rows, qtype)

    actual = libgguf_torch.dequantize(
        torch.from_numpy(encoded).to("cuda"),
        qtype,
        rows.shape,
        dtype=torch.float32,
    )
    expected = libgguf.dequantize_rows(encoded, qtype, n_per_row=rows.shape[-1])

    assert tuple(actual.shape) == rows.shape
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0.0, atol=4.0e-3)


@pytest.mark.parametrize("qtype", COMPILED_DEQUANTIZE_QTYPES, ids=qtype_id)
def test_compiled_cuda_torch_dequantize_native_bytes_matches_libgguf(qtype: GGMLQuantizationType) -> None:
    rows = build_rows(qtype, rows=1)
    encoded = libgguf.quantize_rows(rows, qtype)
    dequantize = compile_or_skip(
        lambda data: libgguf_torch.dequantize(data, qtype, rows.shape, dtype=torch.float32)
    )

    actual = dequantize(torch.from_numpy(encoded).to("cuda"))
    expected = libgguf.dequantize_rows(encoded, qtype, n_per_row=rows.shape[-1])

    assert tuple(actual.shape) == rows.shape
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0.0, atol=4.0e-3)


def test_cuda_dequantize_tensor_returns_torch_compatible_tensor_in_requested_dtype() -> None:
    tensor = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    tensor.tensor_type = GGMLQuantizationType.F32

    actual = libgguf_torch.dequantize_tensor(tensor, dtype=torch.float16)

    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float16
    np.testing.assert_array_equal(actual.cpu().numpy(), tensor.cpu().numpy().astype(np.float16))
