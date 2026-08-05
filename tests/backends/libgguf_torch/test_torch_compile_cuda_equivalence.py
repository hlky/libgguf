from __future__ import annotations

import numpy as np
import pytest

import libgguf
import libgguf.libgguf_torch
from libgguf import GGMLQuantizationType
import torch

from torch.utils._triton import has_triton  # noqa: E402

from .test_torch_equivalence import build_rows, qtype_id, tensor_bytes  # noqa: E402


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available"),
    pytest.mark.skipif(not has_triton(), reason="Triton is not available"),
]

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


@pytest.mark.parametrize("qtype", COMPILED_QUANTIZE_QTYPES, ids=qtype_id)
def test_compiled_cuda_torch_quantize_bytes_match_libgguf(
    qtype: GGMLQuantizationType,
) -> None:
    rows = build_rows(qtype, rows=1)
    quantize = torch.compile(
        lambda data: libgguf.libgguf_torch.quantize(data, qtype), dynamic=False
    )

    actual = quantize(torch.from_numpy(rows).to("cuda"))
    expected = libgguf.quantize_rows(rows, qtype)

    assert tuple(actual.shape) == expected.shape
    np.testing.assert_array_equal(tensor_bytes(actual), expected.view(np.uint8))


@pytest.mark.parametrize("qtype", COMPILED_DEQUANTIZE_QTYPES, ids=qtype_id)
def test_compiled_cuda_torch_dequantize_native_bytes_matches_libgguf(
    qtype: GGMLQuantizationType,
) -> None:
    rows = build_rows(qtype, rows=1)
    encoded = libgguf.quantize_rows(rows, qtype)
    dequantize = torch.compile(
        lambda data: libgguf.libgguf_torch.dequantize(
            data, qtype, rows.shape, dtype=torch.float32
        ),
        dynamic=False,
    )

    actual = dequantize(torch.from_numpy(encoded).to("cuda"))
    expected = libgguf.dequantize_rows(encoded, qtype, n_per_row=rows.shape[-1])

    assert tuple(actual.shape) == rows.shape
    np.testing.assert_allclose(
        actual.cpu().numpy(), expected, rtol=0.0, atol=4.0e-3
    )
