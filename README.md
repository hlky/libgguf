# libgguf

Standalone GGUF reference quantization bindings.

## Benchmarks

Run the torch quantization/dequantization benchmark suite with:

```powershell
python bench\torch_bench.py
```

Use `--qtypes`, `--rows`, `--blocks-per-row`, `--iterations`, `--warmup`, `--device`, `--compile`, and `--json` to configure longer runs or save results.
