# libgguf_torch

`libgguf_torch` provides the Torch CPU GGUF quantization reference implementation. Its supported quantize surface matches `libgguf_numpy`; byte-format dequantizers are Torch-native, while dense `F32` and `F16` are handled as pass-through tensor conversions.

## Coverage

- Native quantize support covers 27 formats, including dense `F32` and `F16`.
- Native byte dequantize support covers 25 quantized formats.
- Dense `F32` and `F16` dequantize through the higher-level tensor path as pass-through formats.
- Metadata-only formats intentionally remain unsupported: `Q8_1`, `Q8_K`, `I8`, `I16`, `I32`, `I64`, and `F64`.

See [FORMAT_SUPPORT.md](FORMAT_SUPPORT.md) for the full per-format support table.

## Test Results

The full package coverage set was run together with the NumPy package coverage tests:

```powershell
.venv\Scripts\python.exe -m pytest libgguf_numpy\tests\test_gguf_np_accuracy.py libgguf_numpy\tests\test_gguf_np_native_quantizers.py libgguf_torch\tests\test_gguf_pt_accuracy.py libgguf_torch\tests\test_gguf_pt_quantizers.py -q
```

Result:

```text
413 passed in 1230.71s (0:20:30)
```

Torch-specific coverage gates:

- `tests/test_gguf_pt_accuracy.py` asserts every native Torch byte dequantizer is present in the accuracy matrix.
- `tests/test_gguf_pt_quantizers.py` asserts Torch quantize coverage matches the full NumPy-supported format set, including dense `F32` and `F16`.

