# CLI

## Python commands

The installed package provides:

- `gguf-inspect`: print GGUF metadata and tensor descriptors.
- `gguf-validate`: report structural GGUF errors and warnings.
- `gguf-compare`: compare descriptors and, optionally, metadata or tensor bytes.

Use `COMMAND --help` for the current arguments.

## Native converter

`libgguf_quantize_gguf` converts a safetensors model to GGUF:

```bash
libgguf_quantize_gguf \
  --src model.safetensors \
  --qtype Q4_K_M \
  --dst model-Q4_K_M.gguf \
  --policy comfy \
  --backend auto \
  --overwrite
```

Required arguments are `--src` and `--qtype`. The converter supports:

- `--policy comfy|dynamic|uniform`
- `--tensor-type PATTERN=TYPE`, `--include PATTERN`, and `--exclude PATTERN`
- `--scratch-bytes`, `--cpu-ram-bytes`, and `--threads`
- `--backend cpu|gpu|auto`
- `--gpu-fallback cpu`
- GPU verification selectors and GPU buffer/pipeline controls
- `--timings`

GPU flags require a build with `LIBGGUF_BUILD_GPU=ON`. Unsupported GPU tensor
formats fail for an explicit GPU backend unless CPU fallback is enabled. Run
`libgguf_quantize_gguf --help` for the complete option list.
