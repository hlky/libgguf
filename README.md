# libgguf

libgguf is a standalone GGUF library with native row kernels, Python bindings,
Torch and NumPy implementations, optional HIP/ROCm GPU kernels, and a native
safetensors-to-GGUF converter.

The native `REF` CPU backend is the correctness reference. Optimized CPU
backends are selected at compile time.

## Install

```bash
python -m pip install .
```

Editable development install:

```bash
python -m pip install --editable .
```

For a ROCm build with the Torch GPU wrapper:

```powershell
$env:CMAKE_ARGS='-DLIBGGUF_BUILD_GPU=ON -DLIBGGUF_ENABLE_ROCM=ON -DLIBGGUF_BUILD_TORCH=ON'
python -m pip install --editable ".[gpu]" --no-build-isolation
```

See [docs/installation.md](docs/installation.md) for build requirements and
CMake options.

## Python row API

```python
import numpy as np
import libgguf

rows = np.random.default_rng(0).normal(size=(4, 4096)).astype(np.float32)
qtype = libgguf.GGMLQuantizationType.Q4_K

encoded = libgguf.quantize_rows(rows, qtype)
decoded = libgguf.dequantize_rows(encoded, qtype, n_per_row=rows.shape[-1])
```

The top-level package also exposes qtype metadata, storage-row operations,
imatrix loading, lightweight GGUF inspection, raw tensor-byte reads, and
structural validation. See [docs/python-api.md](docs/python-api.md).

## Backends

- `libgguf`: native CPU row operations and GGUF metadata APIs.
- `libgguf.libgguf_numpy`: NumPy implementation.
- `libgguf.libgguf_torch`: Torch-native implementation.
- `libgguf.libgguf_gpu`: native Torch GPU operations.

GPU builds explicitly select either HIP/ROCm or CUDA. PyTorch exposes ROCm
devices through its `cuda` device namespace:

```python
import torch
import libgguf
import libgguf.libgguf_gpu

rows = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
qtype = libgguf.GGMLQuantizationType.Q4_K
encoded = libgguf.libgguf_gpu.quantize(rows, int(qtype))
```

### Runtime-free device fragments

`<libgguf/device/simple_q.h>` is a narrow header-only API for embedding Q4_0
and Q8_0 decoding inside downstream GPU kernels. It contains no HIP/CUDA
runtime, Torch, launcher, or lookup-table dependency and is also usable from
ordinary C++ reference tests. The API exposes exact 32-value/18-byte and
32-value/34-byte block layouts, scale and integer-quant accessors, scalar
decoding, and eight-value fragment decoding. Consumers should pin both the
libgguf commit and `simple_q_api_revision`.

## Converter and commands

The native `libgguf_quantize_gguf` executable converts safetensors models to
GGUF and supports CPU, GPU, and automatic backend selection.

```bash
libgguf_quantize_gguf \
  --src model.safetensors \
  --qtype Q4_K_M \
  --dst model-Q4_K_M.gguf \
  --backend auto
```

The Python package also installs `gguf-inspect`, `gguf-validate`, and
`gguf-compare`. See [docs/cli.md](docs/cli.md).

## Documentation

- [Qtypes](docs/qtypes.md)
- [Conversion policy](docs/policy.md)
- [Correctness](docs/correctness.md)
- [Documentation index](docs/index.md)

## License

Apache-2.0. Adapted source files retain their applicable provenance notices.
