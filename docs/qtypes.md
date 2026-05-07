# Qtypes

GGUF quantized tensors are stored in fixed-size blocks. A qtype defines how many source values are encoded per block and how many bytes each encoded block uses. Row APIs require the last dimension to be divisible by the qtype block size.

At the file-format level, GGUF is a single-file binary container with header metadata, key-value metadata, tensor descriptors, and a tensor data blob. libgguf's current public surface includes qtype metadata, row kernels, converter paths, GGUF inspection, and minimal structural validation; first-class public reader/writer APIs and deeper validator coverage are planned.

Use the metadata helpers to inspect exact block and row sizes:

```python
import libgguf

qtype = libgguf.GGMLQuantizationType.Q4_K
block_size, block_bytes = libgguf.GGML_QUANT_SIZES[qtype]
row_bytes = libgguf.row_size(qtype, 4096)
```

## Families

| Family | Examples | Notes |
| --- | --- | --- |
| Storage | `F32`, `F16`, `BF16` | Stored rather than quantized. |
| Legacy Q | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` | 32-value block families compatible with GGML/GGUF behavior. |
| K quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` | 256-value block families with scale/min metadata layouts. |
| IQ quants | `IQ1_S`, `IQ1_M`, `IQ2_*`, `IQ3_*`, `IQ4_*` | Importance/lookup-oriented formats. Some quantizers require or compute imatrix-style weights. |
| TQ quants | `TQ1_0`, `TQ2_0` | Ternary quantization families. |
| FP4 variants | `MXFP4`, `NVFP4` | 4-bit floating-style formats with format-specific scaling. |
| New low-bit Q | `Q1_0` | 128-value block qtype. |

## Supported Public List

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

Exact backend support varies. See [support-matrix.md](support-matrix.md).

## References

- [llama.cpp gguf-py constants](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)
- [llama.cpp gguf-py writer](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_writer.py)
- [GGUF file-format overview](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format)
- [GGUF format concept page](https://www.mintlify.com/ggml-org/llama.cpp/concepts/gguf-format)
