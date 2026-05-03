# libgguf_numpy Format Support

`libgguf_numpy.gguf_np` has native NumPy quantize and dequantize coverage for every GGUF format listed as supported below. The remaining formats are present in shared metadata only and intentionally raise `NotImplementedError`.

## Summary

- Native quantize/dequantize formats: 27.
- Metadata-only unsupported formats: 7.
- Dense `F32` and `F16` are handled as direct dtype conversions.
- `BF16` is native in NumPy, but is not included in the libgguf byte-parity quantizer test because `libgguf` does not expose a BF16 encoder path.

## Supported Formats

| Format | Enum | Block size | Type size | Quantize | Dequantize | Coverage |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `F32` | 0 | 1 | 4 | native | native | `test_gguf_np_accuracy` |
| `F16` | 1 | 1 | 2 | native | native | `test_gguf_np_accuracy` |
| `Q4_0` | 2 | 32 | 18 | native | native | accuracy, byte parity |
| `Q4_1` | 3 | 32 | 20 | native | native | accuracy, byte parity |
| `Q5_0` | 6 | 32 | 22 | native | native | accuracy, byte parity |
| `Q5_1` | 7 | 32 | 24 | native | native | accuracy, byte parity |
| `Q8_0` | 8 | 32 | 34 | native | native | accuracy, byte parity |
| `Q2_K` | 10 | 256 | 84 | native | native | accuracy, byte parity |
| `Q3_K` | 11 | 256 | 110 | native | native | accuracy, byte parity |
| `Q4_K` | 12 | 256 | 144 | native | native | accuracy, byte parity |
| `Q5_K` | 13 | 256 | 176 | native | native | accuracy, byte parity |
| `Q6_K` | 14 | 256 | 210 | native | native | accuracy, byte parity |
| `IQ2_XXS` | 16 | 256 | 66 | native | native | accuracy, byte parity |
| `IQ2_XS` | 17 | 256 | 74 | native | native | accuracy, byte parity |
| `IQ3_XXS` | 18 | 256 | 98 | native | native | accuracy, byte parity |
| `IQ1_S` | 19 | 256 | 50 | native | native | accuracy, byte parity |
| `IQ4_NL` | 20 | 32 | 18 | native | native | accuracy, byte parity |
| `IQ3_S` | 21 | 256 | 110 | native | native | accuracy, byte parity |
| `IQ2_S` | 22 | 256 | 82 | native | native | accuracy, byte parity |
| `IQ4_XS` | 23 | 256 | 136 | native | native | accuracy, byte parity |
| `IQ1_M` | 29 | 256 | 56 | native | native | accuracy, byte parity |
| `BF16` | 30 | 1 | 2 | native | native | `test_gguf_np_accuracy` |
| `TQ1_0` | 34 | 256 | 54 | native | native | accuracy, byte parity |
| `TQ2_0` | 35 | 256 | 66 | native | native | accuracy, byte parity |
| `MXFP4` | 39 | 32 | 17 | native | native | accuracy, byte parity |
| `NVFP4` | 40 | 64 | 36 | native | native | accuracy, byte parity |
| `Q1_0` | 41 | 128 | 18 | native | native | accuracy, byte parity |

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

- `libgguf_numpy/tests/test_gguf_np_accuracy.py` asserts every NumPy-supported format is present in the round-trip accuracy matrix.
- `libgguf_numpy/tests/test_gguf_np_native_quantizers.py` asserts every libgguf-supported native quantizer is covered by byte-parity tests.
