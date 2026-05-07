# CLI

libgguf exposes two Python inspection entry points plus one native conversion executable. The documented conversion CLI is the low-memory C++ `libgguf_quantize_gguf` executable.

## Python Entry Points

| Command | Purpose | Status |
| --- | --- | --- |
| `gguf-inspect` | GGUF metadata and tensor descriptor inspection | experimental |
| `gguf-validate` | Structural GGUF validation without tensor payload reads | experimental |

The Python conversion helper API remains experimental/internal. The old Python conversion wrapper modules are retired; use `libgguf_quantize_gguf` for command-line conversion.

## Native Conversion Executable

The native executable is built when `LIBGGUF_BUILD_TOOLS=ON`:

```bash
libgguf_quantize_gguf \
  --src model.safetensors \
  --qtype Q4_K_M \
  --dst model-Q4_K_M.gguf \
  --policy comfy \
  --backend cpu \
  --overwrite
```

Implemented native options:

- `--src PATH`: source `.safetensors` model.
- `--qtype QTYPE`: output file type, for example `Q4_K_S`, `Q4_K_M`, `Q4_K`, or `Q8_0`.
- `--dst PATH`: output GGUF path.
- `--overwrite`: allow replacing an existing output file.
- `--policy comfy|dynamic|uniform`: tensor selection policy.
- `--imatrix PATH`: accepted for CLI parity; Q/K quantizers do not require it.
- `--tensor-type PATTERN=TYPE`: override matching tensor storage/quant type.
- `--include PATTERN`: force matching 2D tensors into quantization when possible.
- `--exclude PATTERN`: keep matching tensors unquantized.
- `--scratch-bytes N`: native scratch and direct-copy buffer target in bytes.
- `--cpu-ram-bytes N`: alias for `--scratch-bytes`, for callers that want to phrase the CPU-side memory budget directly.
- `--threads N`: worker thread count.
- `--backend cpu|cuda`: backend for quantized tensor encoding. The default is `cpu`; safetensors reads and GGUF writes remain CPU-side for both backends.
- `--cuda-fallback cpu`: when `--backend cuda` is selected, encode unsupported CUDA qtypes on CPU instead of failing.
- `--verify-cuda-tensors N`: for the first `N` tensors encoded on CUDA, also encode with CPU and compare the encoded bytes.
- `--cuda-vram-bytes N`: when `--backend cuda` is selected, use `N` bytes as the CUDA device input/output chunk budget. The default `0` keeps the existing `--scratch-bytes` chunk sizing. Normal CUDA conversion uses two pinned host staging slots, so host RAM can temporarily use up to about `2 * N` in addition to source read buffers.
- `--timings`: print conversion timing breakdown to stderr.
- `--help`: show native help.

The native executable currently supports Q/K output families and storage overrides. Non-Q/K quantization families are rejected by the native executable even though the row APIs support broader qtype coverage.

The CUDA converter backend is experimental and is linked only when the native CUDA target is built. A CPU-only executable accepts the CLI flags but fails clearly if `--backend cuda` is requested. The current CUDA converter qtype set is `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K`; other planned or kernel-level qtypes require `--cuda-fallback cpu` or fail.

With `--timings`, the native converter reports `read`, `cpu_convert`, `h2d`, `cuda_quant`, `d2h`, `write`, and `total` buckets. The `metadata` field is also printed to separate GGUF descriptor writes from tensor payload writes. For CUDA runs, the timing line also includes the configured CUDA VRAM budget plus the largest device input/output buffers used.

## GGUF Inspection

```bash
gguf-inspect model.gguf --metadata --tensors
gguf-inspect model.gguf --json
```

Implemented inspection options:

- `PATH`: GGUF file to inspect.
- `--json`: emit JSON.
- `--metadata`: print metadata entries in text mode.
- `--tensors`: print tensor descriptors in text mode.
- `--limit N`: maximum metadata/tensor rows to print in text mode.
- `--max-array-values N`: maximum metadata array values to keep; use `-1` for full arrays.

`gguf-inspect` reads headers, metadata, and tensor descriptors only. It does not read tensor payload bytes.

## GGUF Validation

```bash
gguf-validate model.gguf
gguf-validate model.gguf --json
```

Implemented validation options:

- `PATH`: GGUF file to validate.
- `--json`: emit JSON.
- `--max-array-values N`: maximum metadata array values to keep while parsing; defaults to `0`, use `-1` for full arrays.

`gguf-validate` checks GGUF structure using the inspector and does not read tensor payload bytes. It reports errors for invalid format structure, duplicate tensor names, tensor payload ranges outside the file, and overlapping known tensor ranges. It reports warnings for missing common metadata, unknown qtypes, invalid qtype row widths, and non-monotonic known tensor offsets. The command exits `1` when errors are present and `0` when there are no errors; warnings alone still exit `0`.
