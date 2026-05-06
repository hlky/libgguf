# SPDX-License-Identifier: Apache-2.0

import torch

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


def dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    return torch.ops._C_gguf.dequantize(W, quant_type, m, n, dtype)
