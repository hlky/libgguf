# CLI

libgguf exposes Python conversion and inspection entry points plus one native executable. The Python conversion tools are frontends; the native executable is a low-memory C++ safetensors-to-GGUF converter.

## Python Entry Points

| Command | Loader/backend | Status |
| --- | --- | --- |
| `quantize-gguf` | safetensors/ckpt loader with native bindings | experimental |
| `quantize-gguf-pt` | Torch safetensors/ckpt loader with native bindings | experimental |
| `quantize-gguf-native` | native safetensors metadata/payload path | experimental |
| `quantize-gguf-torch` | Torch loader plus `libgguf_torch` quantization | experimental |
| `gguf-inspect` | GGUF metadata and tensor descriptor inspection | experimental |

Common implemented options:

```bash
quantize-gguf \
  --src model.safetensors \
  --qtype Q4_K_M \
  --dst model-Q4_K_M.gguf \
  --policy comfy \
  --overwrite
```

Implemented Python CLI options shared by the conversion tools:

- `--src PATH`: source safetensors/ckpt model, except `quantize-gguf-native` which accepts only `.safetensors`.
- `--qtype QTYPE`: output file type, for example `Q4_K_S`, `Q4_K_M`, `Q4_K`, or `Q8_0`.
- `--dst PATH`: output GGUF path.
- `--overwrite`: allow replacing an existing output file.
- `--policy comfy|dynamic|uniform`: tensor selection policy.
- `--imatrix PATH`: llama.cpp imatrix file where supported.
- `--tensor-type PATTERN=QTYPE`: override matching tensor storage/quant type.
- `--include PATTERN`: force matching 2D tensors into quantization when possible.
- `--exclude PATTERN`: keep matching tensors unquantized.

Additional implemented options:

- `quantize-gguf-native --scratch-bytes N`: native scratch buffer target in bytes.
- `quantize-gguf-torch --device DEVICE`: Torch device used for quantization, for example `cpu` or `cuda`.
- `quantize-gguf-torch --compile`: wrap `libgguf_torch` quantization with `torch.compile`.

`quantize-gguf-torch --imatrix` is intentionally rejected by the CLI; use `quantize-gguf-pt` or `quantize-gguf-native` for imatrix workflows.

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

## Native Executable

The native executable is built when `LIBGGUF_BUILD_TOOLS=ON`:

```bash
libgguf_quantize_gguf --src model.safetensors --qtype Q4_K_M --dst model-Q4_K_M.gguf
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
