# Qtypes

A GGUF qtype defines the number of values in a block and the encoded byte size
of that block. Quantized rows must contain a whole number of blocks.

```python
import libgguf

qtype = libgguf.GGMLQuantizationType.Q4_K
block_size, block_bytes = libgguf.GGML_QUANT_SIZES[qtype]
row_bytes = libgguf.row_size(qtype, 4096)
```

## Families

| Family | Formats |
| --- | --- |
| Storage | `F32`, `F16`, `BF16` |
| Legacy Q | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` |
| K quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` |
| IQ quants | `IQ1_S`, `IQ1_M`, `IQ2_*`, `IQ3_*`, `IQ4_*` |
| Ternary | `TQ1_0`, `TQ2_0` |
| FP4 | `MXFP4`, `NVFP4` |
| Low-bit Q | `Q1_0` |

Backend support is determined by the compiled kernel set. Use `row_size` and
`quantize_requires_imatrix` rather than maintaining a separate documentation
matrix.
