# Python API

The top-level `libgguf` package owns format metadata and native CPU row APIs.

## Metadata

```python
import libgguf

qtype = libgguf.GGMLQuantizationType.Q4_K
block_bytes = libgguf.type_size(qtype)
row_bytes = libgguf.row_size(qtype, 4096)
```

The public metadata surface includes:

- `GGMLQuantizationType`
- `LlamaFileType`
- `GGML_QUANT_SIZES`
- `QK_K`
- `type_name(qtype)`
- `type_size(qtype)`
- `row_size(qtype, n_per_row)`
- `quantize_requires_imatrix(qtype)`
- `quant_shape_to_byte_shape(shape, qtype)`
- `quant_shape_from_byte_shape(shape, qtype)`

`row_size` returns zero for unsupported qtypes and invalid row widths.

## Row operations

```python
import numpy as np
import libgguf

rows = np.random.default_rng(0).normal(size=(2, 4096)).astype(np.float32)
encoded = libgguf.quantize_rows(rows, libgguf.GGMLQuantizationType.Q4_K)
decoded = libgguf.dequantize_rows(
    encoded,
    libgguf.GGMLQuantizationType.Q4_K,
    n_per_row=rows.shape[-1],
)
```

The high-level row functions are:

- `quantize_rows(data, qtype, imatrix=None)`
- `dequantize_rows(data, qtype, n_per_row=None)`
- `store_rows(data, qtype)` for `F32`, `F16`, and `BF16`

Raw buffer variants are available as `quantize_rows_raw`,
`quantize_rows_into_raw`, `dequantize_rows_raw`, and
`dequantize_rows_into_raw`.

Quantized row widths must be divisible by the format block size. When a qtype
requires an importance matrix, `quantize_rows` derives one from the input if it
is not supplied. `load_imatrix(path)` reads llama.cpp imatrix files.

## GGUF inspection

`open_gguf`, `inspect_gguf`, and `read_gguf_header` expose the same lightweight
reader. It parses metadata and tensor descriptors without decoding tensor
payloads. `validate_gguf` performs structural validation.

## Backend modules

- `libgguf.libgguf_numpy`: NumPy implementation.
- `libgguf.libgguf_torch`: Torch-native implementation.
- `libgguf.libgguf_gpu`: native Torch GPU operations.
