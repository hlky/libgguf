# SPDX-License-Identifier: Apache-2.0

import torch
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES, quantize_requires_imatrix

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

try:
    from . import _C_gguf  # noqa: F401
except ImportError:
    _C_gguf = None


if hasattr(torch.ops, "_C_gguf") and hasattr(torch.ops._C_gguf, "dequantize"):

    @register_fake("_C_gguf::dequantize")
    def _dequantize_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if W.device.type not in ("cuda", "meta"):
            raise RuntimeError("CUDA dequantize expects a CUDA tensor")
        if W.dtype != torch.uint8:
            raise RuntimeError("CUDA dequantize expects uint8 input")
        if m <= 0:
            raise RuntimeError("CUDA dequantize expects positive row count")
        if n <= 0:
            raise RuntimeError("CUDA dequantize expects positive row width")

        dtype_ = dtype or torch.float16
        if dtype_ not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(
                "CUDA dequantize output dtype must be float16, bfloat16, or float32"
            )

        try:
            qtype = GGMLQuantizationType(quant_type)
        except ValueError:
            raise RuntimeError(
                f"Unsupported GGML quantization type for CUDA dequantize: {quant_type}"
            ) from None
        if qtype not in (
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
        ):
            raise RuntimeError(
                f"Unsupported GGML quantization type for CUDA dequantize: {quant_type}"
            )

        block_size, type_size = GGML_QUANT_SIZES[qtype]
        if n % block_size != 0:
            raise RuntimeError(
                "CUDA dequantize output width must be divisible by the quantization block size"
            )
        row_size = n * type_size // block_size
        expected_numel = m * row_size
        if W.numel() != expected_numel:
            raise RuntimeError(
                f"CUDA dequantize input has {W.numel()} bytes, expected {expected_numel}"
            )
        return torch.empty((m, n), dtype=dtype_, device=W.device)

if hasattr(torch.ops, "_C_gguf") and hasattr(torch.ops._C_gguf, "quantize"):

    @register_fake("_C_gguf::quantize")
    def _quantize_fake(
        W: torch.Tensor, quant_type: int, imatrix: torch.Tensor | None = None
    ) -> torch.Tensor:
        qtype = GGMLQuantizationType(quant_type)
        if W.dtype != torch.float32:
            raise RuntimeError("CUDA quantize expects float32 input")
        if W.dim() < 1:
            raise RuntimeError("CUDA quantize expects at least one dimension")
        if qtype not in (
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
        ):
            raise RuntimeError(
                f"Unsupported GGML quantization type for CUDA quantize: {quant_type}"
            )
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        if W.shape[-1] % block_size != 0:
            raise RuntimeError(
                "CUDA quantize input width must be divisible by the quantization block size"
            )
        if quantize_requires_imatrix(qtype) and imatrix is None:
            raise RuntimeError(
                f"CUDA quantize requires imatrix for GGML quantization type: {quant_type}"
            )
        row_size = W.shape[-1] * type_size // block_size
        return torch.empty((*W.shape[:-1], row_size), dtype=torch.uint8, device=W.device)


def dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    return torch.ops._C_gguf.dequantize(W, quant_type, m, n, dtype)


def quantize(
    W: torch.Tensor, quant_type: int, imatrix: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.ops._C_gguf.quantize(W, quant_type, imatrix)
