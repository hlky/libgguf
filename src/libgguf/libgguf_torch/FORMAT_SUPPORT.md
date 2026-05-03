# libgguf_torch Format Support

`libgguf_torch.gguf_pt` has native Torch CPU quantize and dequantize coverage matching `libgguf_numpy` for supported GGUF formats. The lower-level `dequantize()` helper covers quantized byte formats; `dequantize_tensor()` also treats dense `F32` and `F16` tensors as pass-through formats.

## Summary

- Native quantize formats: 27, including dense `F32` and `F16` pass-through conversion.
- Native dequantize byte formats: 25.
- Dense pass-through dequantize formats: `F32`, `F16`.
- Metadata-only unsupported formats: 7.

## Supported Formats

| Format | Enum | Block size | Type size | Quantize | Dequantize | Coverage |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `F32` | 0 | 1 | 4 | pass-through | pass-through | dense quantize |
| `F16` | 1 | 1 | 2 | pass-through | pass-through | dense quantize |
| `Q4_0` | 2 | 32 | 18 | native | native | accuracy, NumPy parity |
| `Q4_1` | 3 | 32 | 20 | native | native | accuracy, NumPy parity |
| `Q5_0` | 6 | 32 | 22 | native | native | accuracy, NumPy parity |
| `Q5_1` | 7 | 32 | 24 | native | native | accuracy, NumPy parity |
| `Q8_0` | 8 | 32 | 34 | native | native | accuracy, NumPy parity |
| `Q2_K` | 10 | 256 | 84 | native | native | accuracy, NumPy parity |
| `Q3_K` | 11 | 256 | 110 | native | native | accuracy, NumPy parity |
| `Q4_K` | 12 | 256 | 144 | native | native | accuracy, NumPy parity |
| `Q5_K` | 13 | 256 | 176 | native | native | accuracy, NumPy parity |
| `Q6_K` | 14 | 256 | 210 | native | native | accuracy, NumPy parity |
| `IQ2_XXS` | 16 | 256 | 66 | native | native | accuracy, NumPy parity |
| `IQ2_XS` | 17 | 256 | 74 | native | native | accuracy, NumPy parity |
| `IQ3_XXS` | 18 | 256 | 98 | native | native | accuracy, NumPy parity |
| `IQ1_S` | 19 | 256 | 50 | native | native | accuracy, NumPy parity |
| `IQ4_NL` | 20 | 32 | 18 | native | native | accuracy, NumPy parity |
| `IQ3_S` | 21 | 256 | 110 | native | native | accuracy, NumPy parity |
| `IQ2_S` | 22 | 256 | 82 | native | native | accuracy, NumPy parity |
| `IQ4_XS` | 23 | 256 | 136 | native | native | accuracy, NumPy parity |
| `IQ1_M` | 29 | 256 | 56 | native | native | accuracy, NumPy parity |
| `BF16` | 30 | 1 | 2 | native | native | accuracy, NumPy parity |
| `TQ1_0` | 34 | 256 | 54 | native | native | accuracy, NumPy parity |
| `TQ2_0` | 35 | 256 | 66 | native | native | accuracy, NumPy parity |
| `MXFP4` | 39 | 32 | 17 | native | native | accuracy, NumPy parity |
| `NVFP4` | 40 | 64 | 36 | native | native | accuracy, NumPy parity |
| `Q1_0` | 41 | 128 | 18 | native | native | accuracy, NumPy parity |

## Metadata-Only Formats

| Format | Enum | Block size | Type size | Status |
| --- | ---: | ---: | ---: | --- |
| `Q8_1` | 9 | 32 | 40 | unsupported |
| `Q8_K` | 15 | 256 | 292 | unsupported |
| `I8` | 24 | 1 | 1 | unsupported |
| `I16` | 25 | 1 | 2 | unsupported |
| `I32` | 26 | 1 | 4 | unsupported |
| `I64` | 27 | 1 | 8 | unsupported |
| `F64` | 28 | 1 | 8 | unsupported |

## Coverage Gates

- `libgguf_torch/tests/test_gguf_pt_accuracy.py` asserts every native Torch byte dequantizer is present in the accuracy matrix.
- `libgguf_torch/tests/test_gguf_pt_quantizers.py` asserts Torch quantize coverage matches the full NumPy-supported format set, including dense `F32` and `F16`.
