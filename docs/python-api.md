# Python API

The stable-ish public Python surface lives in `libgguf`.

## Qtype Metadata

```python
import libgguf

qtype = libgguf.GGMLQuantizationType.Q4_K
name = libgguf.type_name(qtype)
block_bytes = libgguf.type_size(qtype)
bytes_per_row = libgguf.row_size(qtype, 4096)
```

Public metadata helpers:

- `row_size(qtype, n_per_row) -> int`
- `type_size(qtype) -> int`
- `type_name(qtype) -> str`
- `quantize_requires_imatrix(qtype) -> bool`
- `GGMLQuantizationType`
- `LlamaFileType`
- `GGML_QUANT_SIZES`
- `QK_K`
- `quant_shape_to_byte_shape(shape, quant_type)`
- `quant_shape_from_byte_shape(shape, quant_type)`

## Row Quantization

High-level NumPy row APIs:

```python
import numpy as np
import libgguf

x = np.random.default_rng(0).normal(size=(2, 4096)).astype(np.float32)
qtype = libgguf.GGMLQuantizationType.Q4_K

encoded = libgguf.quantize_rows(x, qtype)
decoded = libgguf.dequantize_rows(encoded, qtype, n_per_row=x.shape[-1])
```

Public row APIs:

- `quantize_rows(data, qtype, imatrix=None) -> np.ndarray`
- `dequantize_rows(data, qtype, n_per_row=None) -> np.ndarray`
- `store_rows(data, qtype) -> np.ndarray`
- `quantize_rows_raw(qtype, src, n_rows, n_per_row, imatrix=None) -> bytes`
- `quantize_rows_into_raw(qtype, src, dst, n_rows, n_per_row, imatrix=None) -> int`
- `dequantize_rows_raw(qtype, src, n_rows, n_per_row) -> bytes`
- `dequantize_rows_into_raw(qtype, src, dst, n_rows, n_per_row) -> int`

`quantize_rows` accepts arrays whose last dimension is the row width. For imatrix qtypes, passing `imatrix=None` computes weights from the input rows.

`store_rows` handles storage qtypes `F32`, `F16`, and `BF16`.

## Imatrix

```python
weights = libgguf.load_imatrix("imatrix.dat")
```

`load_imatrix(path)` reads llama.cpp imatrix data for conversion and row quantization paths.

## Conversion Helpers

```python
result = libgguf.convert_to_gguf(
    "model.safetensors",
    "model-Q4_K_M.gguf",
    "Q4_K_M",
    policy="comfy",
    overwrite=True,
)
```

Public conversion helpers:

- `convert_to_gguf(src, dst=None, qtype="Q4_K_S", *, policy="comfy", overwrite=False, imatrix=None, tensor_overrides=None, include=None, exclude=None) -> QuantResult`
- `convert_safetensors_to_gguf_native(src, dst=None, qtype="Q4_K_S", *, policy="comfy", overwrite=False, imatrix=None, tensor_overrides=None, include=None, exclude=None, scratch_bytes=33554432) -> QuantResult`
- `QuantResult`

The conversion helpers are useful but still experimental. The native converter helper is safetensors-only and writes tensor payloads through the native extension to reduce memory pressure.

## Backend Modules

- `libgguf.libgguf_numpy`: NumPy quant/dequant backend.
- `libgguf.libgguf_torch`: Torch-native quant/dequant backend.
- `libgguf.libgguf_cuda`: optional Torch CUDA extension wrapper.

Backend modules are primarily for integration and parity testing; their APIs may change faster than the top-level row helpers.
