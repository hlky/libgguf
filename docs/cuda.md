# CUDA

`libgguf.libgguf_cuda` is an optional Torch CUDA extension. It exposes direct quantization and dequantization kernels through `torch.ops._C_gguf` and a small Python wrapper:

```python
import torch
import libgguf
import libgguf.libgguf_cuda as gguf_cuda

rows, width = 4, 4096
data = torch.randn(rows, width, device="cuda", dtype=torch.float32)
qtype = libgguf.GGMLQuantizationType.Q4_K

encoded = gguf_cuda.quantize(data, int(qtype))
decoded = gguf_cuda.dequantize(encoded, int(qtype), rows, width, torch.float16)
```

Public wrapper functions:

- `libgguf.libgguf_cuda.quantize(W, quant_type, imatrix=None) -> torch.Tensor`
- `libgguf.libgguf_cuda.dequantize(W, quant_type, m, n, dtype) -> torch.Tensor`

The CUDA API is experimental. It expects CUDA tensors and qtypes passed as integer enum values.

## Build Requirements

The internal CUDA kernel target builds with `nvcc` and the CUDA toolkit. The
Python Torch extension builds when these are also available:

- importable `torch`;
- Torch CMake metadata;
- Python development-module headers.

CMake option:

```bash
-DLIBGGUF_BUILD_CUDA_KERNELS=AUTO
```

`AUTO` skips CUDA when `nvcc` or the CUDA toolkit is missing. `ON` turns missing
CUDA toolkit requirements into a build error. When CUDA is available but Torch
metadata is missing, CMake can still build the internal native CUDA target while
skipping the Python Torch extension. `OFF` disables CUDA targets.

CUDA compilation uses `--fmad=false` to help preserve byte-sensitive behavior.

When building through pip, PEP 517 build isolation may use a temporary Python environment where `torch` is not importable, causing `AUTO` to skip the extension. For a local editable CUDA build, install Torch and the build backend first, then disable build isolation:

```bash
python -m pip install scikit-build-core cmake ninja torch
python -m pip install -e ".[cuda]" --no-build-isolation \
  --config-settings=cmake.define.LIBGGUF_BUILD_CUDA_KERNELS=ON
```

## Qtype Coverage

CUDA quantize/dequantize kernels are present for:

- `Q1_0`
- `Q4_0`, `Q4_1`
- `Q5_0`, `Q5_1`
- `Q8_0`
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- `IQ1_S`, `IQ1_M`
- `IQ2_XXS`, `IQ2_XS`, `IQ2_S`
- `IQ3_XXS`, `IQ3_S`
- `IQ4_NL`, `IQ4_XS`
- `TQ1_0`, `TQ2_0`
- `MXFP4`, `NVFP4`

There is also a BF16 dequantization kernel in the extension source. Treat storage qtype CUDA behavior as experimental unless covered by a local test for your target path.

## Correctness Goal

CUDA quantization targets byte-exact output versus the native CPU reference path for the same input, qtype, shape, and imatrix. CUDA dequantization checks compare decoded values against the native CPU dequantization path for a fixed output dtype.

The test suite includes CUDA exactness checks when CUDA is available.

## Ecosystem Context

[Diffusers](https://huggingface.co/docs/diffusers/main/quantization/gguf) currently documents optional GGUF CUDA kernels through the external [kernels](https://github.com/huggingface/kernels) package, with a note that optimized kernels may introduce minor numerical or visual differences compared with the original GGUF implementation. libgguf's CUDA work is positioned differently: it targets byte-exact quantization where supported and decoded-output parity for dequantization. Diffusers is a potential optional backend/integration target, not currently a supported integration in this repository.

## Current Limitations

- CUDA is optional and depends on local CUDA toolkit discovery.
- The Python Torch extension additionally depends on local Torch build discovery.
- The native safetensors-to-GGUF converter does not yet route tensors through CUDA quantization.
- Source dtype GPU input paths for F16/BF16 conversion are planned.
- IQ quant kernels are exact on checked rows but remain an active performance optimization area.
