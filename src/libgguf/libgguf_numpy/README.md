# libgguf_numpy

`libgguf_numpy` provides the native NumPy GGUF quantization reference implementation used by this repository's accuracy and parity tests.

## Coverage

- Native quantize/dequantize support covers 27 formats.
- Dense `F32` and `F16` are handled as direct dtype conversions.
- Quantized formats include legacy GGML quants, K quants, IQ quants, ternary quants, `BF16`, `MXFP4`, `NVFP4`, and `Q1_0`.
- Metadata-only formats intentionally remain unsupported: `Q8_1`, `Q8_K`, `I8`, `I16`, `I32`, `I64`, and `F64`.

See [FORMAT_SUPPORT.md](FORMAT_SUPPORT.md) for the full per-format support table.

## Test Results

The full package coverage set was run together with the Torch package coverage tests:

```powershell
.venv\Scripts\python.exe -m pytest libgguf_numpy\tests\test_gguf_np_accuracy.py libgguf_numpy\tests\test_gguf_np_native_quantizers.py libgguf_torch\tests\test_gguf_pt_accuracy.py libgguf_torch\tests\test_gguf_pt_quantizers.py -q
```

Result:

```text
413 passed in 1230.71s (0:20:30)
```

NumPy-specific coverage gates:

- `tests/test_gguf_np_accuracy.py` asserts every NumPy-supported format is present in the round-trip accuracy matrix.
- `tests/test_gguf_np_native_quantizers.py` asserts every libgguf-supported native quantizer is covered by byte-parity tests.

