from __future__ import annotations

from os import PathLike
from typing import Any, Mapping, Sequence

import numpy as np

from .quantize_pt import (
    QuantResult,
    convert_to_gguf as _convert_to_gguf,
    detect_arch,
    load_state_dict,
    parse_qtype,
    parse_tensor_qtype,
    strip_prefix,
)


def _make_torch_quantized_tensor_data(device: str | Any, compile_quantize: bool):
    compiled_quantizers: dict[Any, Any] = {}

    def _torch_quantized_tensor_data(
        gguf: Any,
        libgguf: Any,
        torch: Any,
        tensor: Any,
        qtype: Any,
        imatrix: np.ndarray | None,
    ) -> np.ndarray:
        if imatrix is not None:
            raise ValueError("The torch quantization backend does not support explicit imatrix weights")

        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        import libgguf.libgguf_torch as libgguf_torch

        data = tensor.to(device=target_device, dtype=torch.float32)
        quantize_fn = libgguf_torch.quantize
        if compile_quantize:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available in this torch installation")
            if qtype not in compiled_quantizers:
                compiled_quantizers[qtype] = torch.compile(
                    lambda input_data, qtype=qtype: libgguf_torch.quantize(input_data, qtype),
                    dynamic=False,
                )
            quantize_fn = compiled_quantizers[qtype]
        quantized = quantize_fn(data, qtype) if not compile_quantize else quantize_fn(data)
        return quantized.detach().cpu().contiguous().view(torch.uint8).numpy()

    return _torch_quantized_tensor_data


def convert_to_gguf(
    src: str | PathLike[str],
    dst: str | PathLike[str] | None = None,
    qtype: str | Any = "Q4_K_S",
    *,
    policy: str = "comfy",
    overwrite: bool = False,
    imatrix: str | PathLike[str] | Mapping[str, np.ndarray] | None = None,
    tensor_overrides: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    device: str | Any = "cpu",
    compile: bool = False,
) -> QuantResult:
    if imatrix is not None:
        raise ValueError("The torch quantization backend does not support explicit imatrix weights")

    return _convert_to_gguf(
        src,
        dst,
        qtype,
        policy=policy,
        overwrite=overwrite,
        imatrix=None,
        tensor_overrides=tensor_overrides,
        include=include,
        exclude=exclude,
        quantized_tensor_data=_make_torch_quantized_tensor_data(device, compile),
    )


__all__ = [
    "QuantResult",
    "convert_to_gguf",
    "detect_arch",
    "load_state_dict",
    "parse_qtype",
    "parse_tensor_qtype",
    "strip_prefix",
]
