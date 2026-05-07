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
- `--scratch-bytes N`: native scratch buffer target in bytes.
- `--threads N`: worker thread count.
- `--timings`: print conversion timing breakdown to stderr.
- `--help`: show native help.

The native executable currently supports Q/K output families and storage overrides. Non-Q/K quantization families are rejected by the native executable even though the row APIs support broader qtype coverage.

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
