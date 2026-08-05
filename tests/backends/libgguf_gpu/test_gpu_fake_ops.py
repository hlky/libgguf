from __future__ import annotations

import pytest

import libgguf
import libgguf.libgguf_gpu
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES
import torch


GPU_QUANT_QTYPES = tuple(
    qtype
    for qtype in GGMLQuantizationType
    if qtype
    not in {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.Q8_1,
        GGMLQuantizationType.Q8_K,
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
        GGMLQuantizationType.F64,
    }
)
GPU_DEQUANT_QTYPES = (*GPU_QUANT_QTYPES, GGMLQuantizationType.BF16)


def require_gpu_ops() -> None:
    if not all(hasattr(torch.ops._C_gguf, name) for name in ("quantize", "dequantize")):
        pytest.skip("libgguf GPU Torch operations are unavailable")


def qtype_id(qtype: GGMLQuantizationType) -> str:
    return qtype.name


def imatrix_for(qtype: GGMLQuantizationType, width: int) -> torch.Tensor | None:
    if not libgguf.quantize_requires_imatrix(qtype):
        return None
    return torch.ones(width, device="meta", dtype=torch.float32)


@pytest.mark.parametrize("qtype", GPU_QUANT_QTYPES, ids=qtype_id)
def test_gpu_quantize_meta_contract(qtype: GGMLQuantizationType) -> None:
    require_gpu_ops()
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    width = block_size * 2
    rows = torch.empty((2, 3, width), device="meta", dtype=torch.float32)

    result = libgguf.libgguf_gpu.quantize(rows, int(qtype), imatrix_for(qtype, width))

    assert result.device.type == "meta"
    assert result.dtype == torch.uint8
    assert result.shape == (2, 3, width * type_size // block_size)


@pytest.mark.parametrize("qtype", GPU_DEQUANT_QTYPES, ids=qtype_id)
@pytest.mark.parametrize("dtype", (None, torch.float32))
def test_gpu_dequantize_meta_contract(
    qtype: GGMLQuantizationType, dtype: torch.dtype | None
) -> None:
    require_gpu_ops()
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    width = block_size * 2
    encoded = torch.empty((3, width * type_size // block_size), device="meta", dtype=torch.uint8)

    result = libgguf.libgguf_gpu.dequantize(encoded, int(qtype), 3, width, dtype)

    assert result.device.type == "meta"
    assert result.dtype == (dtype or torch.float16)
    assert result.shape == (3, width)


@pytest.mark.parametrize(
    "rows,qtype",
    (
        (torch.empty((), device="meta", dtype=torch.float32), GGMLQuantizationType.Q4_0),
        (torch.empty((2, 32), device="meta", dtype=torch.float16), GGMLQuantizationType.Q4_0),
        (torch.empty((2, 33), device="meta", dtype=torch.float32), GGMLQuantizationType.Q4_0),
        (torch.empty((2, 32), device="meta", dtype=torch.float32), GGMLQuantizationType.F32),
    ),
)
def test_gpu_quantize_rejects_invalid_input(
    rows: torch.Tensor, qtype: GGMLQuantizationType
) -> None:
    require_gpu_ops()
    with pytest.raises(RuntimeError):
        libgguf.libgguf_gpu.quantize(rows, int(qtype))


def test_gpu_quantize_enforces_imatrix_contract() -> None:
    require_gpu_ops()
    qtype = GGMLQuantizationType.IQ2_XXS
    width = GGML_QUANT_SIZES[qtype][0]
    rows = torch.empty((2, width), device="meta", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        libgguf.libgguf_gpu.quantize(rows, int(qtype))
    with pytest.raises(RuntimeError):
        libgguf.libgguf_gpu.quantize(
            rows,
            int(qtype),
            torch.empty((1, width), device="meta", dtype=torch.float32),
        )

    result = libgguf.libgguf_gpu.quantize(
        rows,
        int(qtype),
        torch.ones(width, device="meta", dtype=torch.float32),
    )
    assert result.dtype == torch.uint8


@pytest.mark.parametrize(
    "encoded,qtype,m,n,dtype",
    (
        (torch.empty((2, 18), device="meta", dtype=torch.float32), GGMLQuantizationType.Q4_0, 2, 32, None),
        (torch.empty((2, 18), device="meta", dtype=torch.uint8), GGMLQuantizationType.F32, 2, 32, None),
        (torch.empty((2, 18), device="meta", dtype=torch.uint8), GGMLQuantizationType.Q4_0, 2, 33, None),
        (torch.empty((2, 17), device="meta", dtype=torch.uint8), GGMLQuantizationType.Q4_0, 2, 32, None),
        (torch.empty((2, 18), device="meta", dtype=torch.uint8), GGMLQuantizationType.Q4_0, 2, 32, torch.int32),
    ),
)
def test_gpu_dequantize_rejects_invalid_input(
    encoded: torch.Tensor,
    qtype: GGMLQuantizationType,
    m: int,
    n: int,
    dtype: torch.dtype | None,
) -> None:
    require_gpu_ops()
    with pytest.raises(RuntimeError):
        libgguf.libgguf_gpu.dequantize(encoded, int(qtype), m, n, dtype)
