from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

import libgguf
from libgguf import GGMLQuantizationType

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

LIBGGUF_CUDA_STATUS_SUCCESS = 0
LIBGGUF_CUDA_DEQUANTIZE_DTYPE_F32 = 0
LIBGGUF_CUDA_DEQUANTIZE_DTYPE_F16 = 1


def require_native_dequantize_abi() -> ctypes.CDLL:
    extension_path = os.environ.get("LIBGGUF_CUDA_EXTENSION")
    if extension_path is None:
        cuda_ops = pytest.importorskip("libgguf.libgguf_cuda.ops")
        extension = getattr(cuda_ops, "_C_gguf", None)
        if extension is None:
            pytest.skip("libgguf CUDA extension is not available")
        extension_path = str(Path(extension.__file__))
    library = ctypes.CDLL(extension_path)
    try:
        launcher = library.libgguf_cuda_dequantize_rows_on_stream
    except AttributeError:
        pytest.fail("libgguf CUDA native dequantize ABI is not exported")
    launcher.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    launcher.restype = ctypes.c_int
    return library


def q4_0_rows() -> np.ndarray:
    row = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
    return np.ascontiguousarray(np.stack([row, row[::-1]]), dtype=np.float32)


@pytest.mark.parametrize(
    ("abi_dtype", "torch_dtype"),
    (
        (LIBGGUF_CUDA_DEQUANTIZE_DTYPE_F32, torch.float32),
        (LIBGGUF_CUDA_DEQUANTIZE_DTYPE_F16, torch.float16),
    ),
)
def test_native_dequantize_q4_0_writes_caller_output_on_stream(
    abi_dtype: int, torch_dtype: torch.dtype
) -> None:
    library = require_native_dequantize_abi()
    rows = q4_0_rows()
    encoded = torch.from_numpy(libgguf.quantize_rows(rows, GGMLQuantizationType.Q4_0)).to("cuda")
    actual = torch.empty(rows.shape, device=encoded.device, dtype=torch_dtype)
    stream = torch.cuda.Stream(device=encoded.device)
    event = torch.cuda.Event()

    with torch.cuda.device(encoded.device), torch.cuda.stream(stream):
        status = library.libgguf_cuda_dequantize_rows_on_stream(
            ctypes.c_void_p(encoded.data_ptr()),
            int(GGMLQuantizationType.Q4_0),
            rows.shape[0],
            rows.shape[1],
            abi_dtype,
            ctypes.c_void_p(actual.data_ptr()),
            ctypes.c_void_p(stream.cuda_stream),
        )
        event.record()

    assert status == LIBGGUF_CUDA_STATUS_SUCCESS
    torch.cuda.current_stream(encoded.device).wait_event(event)
    expected = torch.from_numpy(
        libgguf.dequantize_rows(encoded.cpu().numpy(), GGMLQuantizationType.Q4_0, n_per_row=rows.shape[1])
    ).to(torch_dtype)

    np.testing.assert_allclose(actual.cpu().float().numpy(), expected.float().numpy())
