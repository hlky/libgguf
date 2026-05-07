# Support Matrix

This is a conservative initial matrix based on visible source and tests. It distinguishes row/backends from converter executables. `yes` means implemented in visible code and covered by the current style of tests or source layout. `experimental` means implemented but optional, young, or explicitly subject to change. `planned` means intended but not currently public. `unknown` means not claimed here.

| qtype | native CPU quant | native CPU dequant | NumPy quant | NumPy dequant | Torch quant | Torch dequant | CUDA quant | CUDA dequant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `F32` | yes (storage) | unknown | yes (storage) | yes (storage) | yes (storage) | yes (storage) | unknown | unknown |
| `F16` | yes (storage) | unknown | yes (storage) | yes (storage) | yes (storage) | yes (storage) | unknown | unknown |
| `BF16` | yes (storage) | unknown | yes (storage) | yes (storage) | yes (storage) | yes (storage) | unknown | experimental |
| `Q1_0` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q4_0` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q4_1` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q5_0` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q5_1` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q8_0` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q2_K` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q3_K` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q4_K` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q5_K` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `Q6_K` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ1_S` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ1_M` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ2_XXS` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ2_XS` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ2_S` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ3_XXS` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ3_S` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ4_NL` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `IQ4_XS` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `TQ1_0` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `TQ2_0` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `MXFP4` | yes | yes | yes | yes | yes | yes | experimental | experimental |
| `NVFP4` | yes | yes | yes | yes | yes | yes | experimental | experimental |

Notes:

- The native row APIs support the broad qtype list above.
- The native executable `libgguf_quantize_gguf` is currently Q/K-focused and rejects IQ/TQ/MXFP4/NVFP4 output families.
- `Q8_1`, `Q8_K`, integer storage types, and `F64` are present in enum metadata but are not claimed as supported row quantization targets here.
- CUDA is optional and depends on a successful Torch/CUDA extension build.
- A generated support matrix is planned so this file can be kept in sync with code and tests automatically.
