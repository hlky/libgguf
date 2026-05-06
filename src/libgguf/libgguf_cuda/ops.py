# SPDX-License-Identifier: Apache-2.0

import torch
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES

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
        return torch.empty((m, n), dtype=dtype or torch.float16, device=W.device)

if hasattr(torch.ops, "_C_gguf") and hasattr(torch.ops._C_gguf, "quantize"):

    @register_fake("_C_gguf::quantize")
    def _quantize_fake(
        W: torch.Tensor, quant_type: int, imatrix: torch.Tensor | None = None
    ) -> torch.Tensor:
        qtype = GGMLQuantizationType(quant_type)
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
            raise NotImplementedError(f"CUDA quantize does not support {qtype.name}")
        block_size, type_size = GGML_QUANT_SIZES[qtype]
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
