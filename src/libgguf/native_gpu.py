"""Compiled, allocation-free native GPU out variants.

Each call writes a caller-owned one-element I32 status tensor asynchronously:
zero means success, four means malformed/nonfinite data. Invalid values leave
the complete output untouched. Inspect status after stream completion before
publishing its output. Metadata errors raise immediately. Operations use the
current Torch stream, record allocation lifetimes, and support graph capture.
"""

import torch
from torch.library import register_fake

from . import _C_gguf
from .native_formats import NATIVE_QUANT_SIZES, NativeQuantizationType


def _check(tensor, device, dtype):
    torch._check(tensor.device == device and tensor.device.type in ("cuda", "meta"), lambda: "native GPU tensors must share a CUDA device")
    torch._check(tensor.dtype == dtype and tensor.is_contiguous(), lambda: "native GPU tensor dtype/contiguity differs")


def _geometry(format, columns):
    block, size, _ = NATIVE_QUANT_SIZES[NativeQuantizationType(format)]
    torch._check(columns > 0, lambda: "native GPU width must be positive")
    torch._check(columns % block == 0, lambda: "native GPU width must contain whole canonical blocks")
    return columns // block * size


def _status(status, device):
    _check(status, device, torch.int32)
    torch._check(status.ndim == 1 and status.numel() == 1, lambda: "native GPU status must be one-element I32")


@register_fake("_C_gguf::native_quantize_into")
def _quantize_fake(source, format, output, status, importance=None):
    torch._check(source.ndim == 2, lambda: "native GPU encoder requires rank-two input")
    torch._check(source.shape[0] > 0, lambda: "native GPU rows must be positive")
    _check(source, source.device, torch.float32)
    _check(output, source.device, torch.uint8)
    _status(status, source.device)
    row_bytes = _geometry(format, source.shape[1])
    torch._check(output.ndim == 2, lambda: "native GPU output must be rank-two")
    torch._check(output.shape[0] == source.shape[0], lambda: "native GPU row counts differ")
    torch._check(output.shape[1] == row_bytes, lambda: "native GPU output row bytes differ")
    if importance is not None:
        _check(importance, source.device, torch.float32)
        torch._check(importance.ndim == 1 and importance.shape[0] == source.shape[1], lambda: "native GPU importance must be a K vector")


@register_fake("_C_gguf::native_dequantize_into")
def _dequantize_fake(encoded, format, columns, output, status):
    torch._check(encoded.ndim == 2, lambda: "native GPU decoder requires rank-two input")
    torch._check(encoded.shape[0] > 0, lambda: "native GPU rows must be positive")
    _check(encoded, encoded.device, torch.uint8)
    _check(output, encoded.device, torch.float32)
    _status(status, encoded.device)
    torch._check(encoded.shape[1] == _geometry(format, columns), lambda: "native GPU encoded row bytes differ")
    torch._check(output.ndim == 2, lambda: "native GPU decoded output must be rank-two")
    torch._check(output.shape[0] == encoded.shape[0], lambda: "native GPU decoded rows differ")
    torch._check(output.shape[1] == columns, lambda: "native GPU decoded width differs")


def quantize_into(source, format, output, status, importance=None):
    """Enqueue native encoding into output and status on the current stream."""
    torch.ops._C_gguf.native_quantize_into(source, int(format), output, status, importance)


def dequantize_into(encoded, format, columns, output, status):
    """Enqueue canonical F32 decoding into caller-owned output and status."""
    torch.ops._C_gguf.native_dequantize_into(encoded, int(format), columns, output, status)
