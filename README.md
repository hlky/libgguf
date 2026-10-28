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

`<libgguf/gpu/dequantize/fragment.cuh>` is the lean header-only API for
embedding partial block decoding inside downstream GPU kernels. It contains no
HIP/CUDA runtime, Torch, launcher, or lookup-table dependency and is also
usable from ordinary C++ reference tests. `BlockFragmentDecoder<Format>`
currently exposes scalar and eight-value fragment decoding for Q4_0, Q4_1,
Q5_0, Q5_1, and Q8_0. Consumers should pin both the libgguf commit and
`block_fragment_decoder_api_revision`.

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

Original Mixtral safetensors names are supported, including the `model.`
prefix and the independent `lm_head.weight`. For uniform Q4_K_S conversion,
keep routers in BF16 with `--tensor-type '*.block_sparse_moe.gate.weight=BF16'`.
The converter accepts one safetensors file; merge sharded checkpoints without
renaming their tensors before conversion.

A standalone ROCm SDK can be selected with `-DLIBGGUF_ROCM_ROOT=/path/to/sdk`
when configuring a GPU build. On Windows, deploy the matching SDK runtime DLLs
beside the converter executable if an older system HIP runtime would otherwise
be loaded. The tested SDK required `amdhip64_7.dll`, `amd_comgr.dll`, and
`rocm_kpack.dll`; do not replace system DLLs.

## Documentation

- [Qtypes](docs/qtypes.md)
- [Conversion policy](docs/policy.md)
- [Correctness](docs/correctness.md)
- [GPU logical subgroups and compile-only gfx942 bring-up](docs/gpu-subgroups.md)
- [Documentation index](docs/index.md)

## License

Apache-2.0. Adapted source files retain their applicable provenance notices.
