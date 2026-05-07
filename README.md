# libgguf

Standalone GGUF read/write, byte-exact quantization, and CUDA-accelerated conversion for C++, Python, NumPy, Torch, and CUDA.

libgguf vendors and adapts GGUF/GGML quantization kernels from llama.cpp into a reusable standalone library and toolkit. The goal is to make GGUF infrastructure available directly to conversion tools and downstream projects without requiring a two-stage route through llama.cpp binaries or partial Python/Torch-only implementations.

The repository currently contains native GGUF row kernels, Python bindings, NumPy and Torch backends, an optional CUDA Torch extension, safetensors-to-GGUF conversion paths, public GGUF inspection and structural validation tools, benchmark tools, and tensor planning policy for real image-model conversion workflows. First-class public GGUF reader and writer APIs are planned; today, GGUF read/write logic is present primarily through converter paths.

## Status

| Field | Value |
| --- | --- |
| Status | active development |
| License | Apache-2.0 |
| Python | >=3.10 |
| Version | 0.1.0 |
| CUDA | optional, experimental, broad qtype coverage |

## Features

- Standalone native C++ GGUF quantization and dequantization library.
- Python bindings for native CPU row kernels.
- Extended NumPy GGUF quantization/dequantization backend.
- Extended Torch GGUF quantization/dequantization backend.
- Optional CUDA quantization and dequantization kernels exposed through a Torch extension.
- Native low-memory safetensors-to-GGUF conversion executable.
- Experimental/internal Python conversion helper modules for native, NumPy-backed, Torch-loaded, and Torch-native workflows.
- Experimental GGUF metadata, tensor descriptor inspection, and structural validation API/CLI.
- Deterministic policy-based tensor planning for real image-model GGUF conversion.
- Benchmark suite for native, Torch, and CUDA paths.
- Planned first-class GGUF reader/writer APIs.

## Why libgguf

- Byte-exact quantization/dequantization against the native CPU reference path where supported.
- Broad CUDA quantization and dequantization qtype coverage.
- Stack-free near-roofline CUDA dequantization across tested qtypes.
- Very fast CUDA quantization for Q/K/TQ/MXFP4/NVFP4 families, with IQ kernels improved and still the active optimization frontier.
- SIMD/threaded native CPU backend.
- Low-memory native converter path for safetensors-to-GGUF conversion.
- Multiple backend implementations for parity testing and integration.

## Backends

| Backend | Purpose | Status |
| --- | --- | --- |
| native C++ CPU | Reference row quant/dequant kernels, SIMD/threaded CPU paths, shared library, C ABI | active |
| Python bindings | `libgguf` row APIs and native converter bridge | active |
| `libgguf_numpy` | NumPy quant/dequant implementation for parity testing and integration | active |
| `libgguf_torch` | Torch-native quant/dequant implementation for parity testing and integration | active |
| `libgguf_cuda` | Optional Torch CUDA extension with direct quant/dequant kernels | experimental |
| `libgguf_quantize_gguf` | Low-memory C++ safetensors-to-GGUF conversion executable | active, Q/K-focused |
| Python conversion helpers | Helper modules over native bindings, safetensors loaders, and Torch backends | experimental/internal |

## Installation

Editable development install:

```bash
python -m pip install -e .
```

Python conversion helper dependencies:

```bash
python -m pip install -e ".[quantize]"
```

CUDA extension dependencies:

```bash
python -m pip install -e ".[cuda]"
```

Core dependency: `numpy`. Optional extras: `cuda`, `quantize`, and `test`.

The build backend is `scikit-build-core`. Native builds require CMake >=3.18 and C++17. CUDA extension builds require importable `torch`, Torch CMake metadata, and `nvcc`.

Useful CMake options:

- `LIBGGUF_BUILD_CUDA_KERNELS=AUTO|ON|OFF`: optional CUDA Torch extension build, default `AUTO`.
- `LIBGGUF_BUILD_TOOLS=ON`: build native command-line tools, default `ON`.
- `LIBGGUF_BUILD_BENCHMARKS=OFF`: build native benchmark binaries, default `OFF`.

## Quick Start

Native Python row kernels:

```python
import numpy as np
import libgguf

x = np.random.default_rng(0).normal(size=(4, 4096)).astype(np.float32)
qtype = libgguf.GGMLQuantizationType.Q4_K

q = libgguf.quantize_rows(x, qtype)
y = libgguf.dequantize_rows(q, qtype, n_per_row=4096)
```

Experimental CUDA Torch extension:

```python
import torch
import libgguf
import libgguf.libgguf_cuda as gguf_cuda

rows, width = 4, 4096
tensor_cuda = torch.randn(rows, width, device="cuda", dtype=torch.float32)
qtype = libgguf.GGMLQuantizationType.Q4_K

q = gguf_cuda.quantize(tensor_cuda, int(qtype))
y = gguf_cuda.dequantize(q, int(qtype), rows, width, torch.float16)
```

## CLI Tools

Python entry points:

- `gguf-inspect`: GGUF metadata and tensor descriptor inspection.
- `gguf-validate`: structural GGUF validation without reading tensor payload bytes.

Native executable:

- `libgguf_quantize_gguf`: low-memory C++ safetensors-to-GGUF converter. The native executable is currently Q/K-focused; non-Q/K quantization families are not supported by this executable yet.

Common conversion shape:

```bash
libgguf_quantize_gguf --src model.safetensors --qtype Q4_K_M --dst model-Q4_K_M.gguf
libgguf_quantize_gguf --src model.safetensors --qtype Q4_K_M --dst model-Q4_K_M.gguf --scratch-bytes 33554432
```

Python conversion helper modules remain experimental/internal and require the `quantize` extra when used directly.

See [docs/cli.md](docs/cli.md) for implemented options.

## Quantization Policy

Conversion uses deterministic tensor planning, not magic. Current policies are:

- `uniform`: quantize eligible 2D weight tensors uniformly.
- `comfy`: use architecture-aware skip and high-precision patterns similar to image-model GGUF conversion workflows.
- `dynamic`: build on `comfy` with deterministic tensor-role and layer-position promotion logic, including ongoing investigation of Unsloth Dynamic-like behavior.

All policies support tensor overrides plus include/exclude patterns. See [docs/policy.md](docs/policy.md).

## Supported Qtypes

The public enum and row APIs cover these storage and quantization families:

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
- `F32`, `F16`, `BF16` storage

Exact support varies by backend and converter path. See [docs/support-matrix.md](docs/support-matrix.md).

## Benchmarks

Benchmarks are representative development results on an RTX 3090, not universal performance claims. For shape `11008x4096`, recent CUDA dequantization results show tested qtypes running stack-free at roughly `0.23-0.28 ms`, around `778-817 GB/s`, with low register counts and about `65x-98x` speedup versus the CPU default path for the sampled qtypes.

Representative CUDA dequant rows:

| qtype | ms | GB/s | speedup vs CPU default |
| --- | ---: | ---: | ---: |
| `Q1_0` | 0.233 | 799.6 | 93.3x |
| `Q8_0` | 0.279 | 817.1 | 65.5x |
| `Q4_K` | 0.254 | 811.1 | 78.8x |
| `Q5_K` | 0.259 | 814.8 | 79.5x |
| `Q6_K` | 0.267 | 814.5 | 75.6x |
| `IQ2_XS` | 0.246 | 786.6 | 98.3x |
| `IQ4_XS` | 0.254 | 803.6 | 72.5x |
| `TQ1_0` | 0.237 | 802.4 | 81.6x |
| `TQ2_0` | 0.239 | 802.9 | 87.9x |

CUDA quantization is strong for Q/K/TQ/MX/NV families, with IQ kernels improved significantly and still the active optimization frontier. IQ quant kernels are exact on checked rows and continue to be optimized.

See [docs/benchmarks.md](docs/benchmarks.md) for detailed tables and metrics.

## Correctness

The native CPU path is the reference path. CUDA, NumPy, and Torch implementations are tested for byte exactness where supported: same input, qtype, and shape should produce identical encoded bytes. Dequantization checks compare decoded output for a fixed destination dtype. Frozen golden fixtures are planned to supplement generated CPU-reference checks.

See [docs/correctness.md](docs/correctness.md).

## Ecosystem Context

libgguf is not an official llama.cpp project. It adapts GGUF/GGML reference behavior into a standalone infrastructure library and keeps compatibility as an engineering target where applicable.

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) are the upstream GGUF/GGML ecosystem references for format behavior, constants, Python writer/reader patterns, and reference quantization behavior.
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) is the existing community ComfyUI GGUF inference/custom-node integration. libgguf may replace or support parts of that stack with reusable native, Python, Torch, and CUDA backend infrastructure.
- [ComfyUI-GGUF tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools) show the current conversion workflow that routes through Python tooling plus patched llama.cpp quantization. libgguf's native and Python conversion tools aim to make that flow more direct and reusable.
- [Diffusers GGUF docs](https://huggingface.co/docs/diffusers/main/quantization/gguf) describe current Diffusers GGUF loading through `from_single_file` model classes, low-memory `torch.uint8` storage, dynamic dequantization during forward, and optional CUDA kernels through the [kernels](https://github.com/huggingface/kernels) package. Diffusers is a potential optional backend/integration target for libgguf, not currently claimed as supported here.
- Public model repositories such as [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf) are useful real-world compatibility targets for conversion and inference testing.
- [Unsloth Dynamic GGUF](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) is relevant policy background for tensor-level qtype decisions. libgguf's `dynamic` policy is deterministic planning work inspired by this class of approach, not a claim of matching Unsloth results.

See [docs/ecosystem.md](docs/ecosystem.md) for the fuller reference map.

## Roadmap

- First-class GGUF reader/writer API.
- Deeper GGUF validator coverage.
- CUDA integration into the native converter.
- Source dtype GPU input path for F16/BF16.
- Support matrix automation.
- Golden exactness fixtures.
- CUDA IQ quant polish.
- Packaging and wheels.
- Diffusers optional backend/integration exploration.
- ComfyUI-GGUF backend/tooling support or replacement exploration.

## Relationship To Upstream Projects

GGUF format behavior and quantization kernels are intended to stay compatible with llama.cpp/GGML/GGUF reference behavior where applicable. The NumPy backend extends gguf-py-style implementations, and the Torch backend extends ComfyUI-GGUF-style native Torch implementations. libgguf keeps those ideas in a standalone infrastructure package with native C++ and CUDA paths.

## License

Apache-2.0. Vendored or adapted code provenance should be documented in the relevant source files and expanded where appropriate.
