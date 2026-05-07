# Benchmarks

Benchmark numbers here are representative development results, not universal performance claims.

Hardware:

- GPU: RTX 3090
- CUDA roofline used by `bench/cuda_quant_bench.py`: `936.2 GB/s`
- Representative large shape: `11008x4096`

## Commands

Native conversion benchmark:

```bash
python bench/conversion_bench.py \
  --src /models/FLUX.1-dev/flux1-dev.safetensors \
  --qtype Q4_K_M \
  --policy dynamic \
  --runs 3 \
  --converter ./build/cmake/ninja-release/libgguf_quantize_gguf
```

The conversion benchmark runs `libgguf_quantize_gguf` end-to-end with `--timings`,
writes one GGUF output per repeated run, and stores `summary.json` plus
`summary.csv` under `bench/results/<timestamp>/`. Reports include
native timing fields when printed by the converter, Python wall time, output
file size, tensor qtype counts, fallback counts, stdout/stderr, and the exact
command used for each run.

Use a local safetensors path for FLUX.1-dev or any other model. The benchmark
does not download model files automatically; place the file on local storage
first and point `--src` at it. Use `--run-name local_flux_q4km` when you want a
stable results directory name, and keep ad hoc machine-specific outputs local
unless they are intentionally promoted into curated benchmark artifacts.

For future CPU/CUDA converter comparisons, `--backend` is a report label:

```bash
python bench/conversion_bench.py \
  --src /models/model.safetensors \
  --qtype Q4_K_M \
  --backend native \
  --runs 3
```

When a CUDA-enabled conversion path exists, run the same command with the CUDA
converter/flags and `--backend cuda` so the JSON/CSV rows are comparable.

CUDA quantization benchmark:

```bash
python bench/cuda_quant_bench.py \
  --qtypes Q4_0,Q8_0,Q4_K,Q5_K,Q6_K,TQ1_0,TQ2_0,MXFP4,NVFP4 \
  --shapes 11008x4096 \
  --csv bench/results/local_cuda_quant.csv
```

CUDA benchmark options include qtype selection, shape selection, baseline CSV comparison, output CSV/JSON, `traffic_gb_s`, `roofline_pct`, `blocks_per_s`, and exactness checks for shapes up to `--exact-rows`.

Torch backend benchmark:

```bash
python bench/torch_bench.py --qtypes default --rows 1 --blocks-per-row 1
```

Useful Torch benchmark options include `--qtypes`, `--rows`, `--blocks-per-row`, `--iterations`, `--warmup`, `--device`, `--compile`, and `--json`.

## CUDA Dequantization

Representative latest development results on RTX 3090 for shape `11008x4096`:

- all listed qtypes are stack-free;
- register count is low, roughly 14-27 registers per thread;
- tested qtypes dequantize in about `0.23-0.28 ms`;
- throughput is about `778-817 GB/s`;
- speedup versus CPU default is about `65x-98x` for the sampled qtypes.

| qtype | ms | traffic GB/s | speedup vs CPU default |
| --- | ---: | ---: | ---: |
| `Q1_0` | 0.233 | 799.6 | 93.3x |
| `Q8_0` | 0.279 | 817.1 | 65.5x |
| `Q4_K` | 0.254 | 811.1 | 78.8x |
| `Q5_K` | 0.259 | 814.8 | 79.5x |
| `Q6_K` | 0.267 | 814.5 | 75.6x |
| `IQ2_XS` | 0.246 | 786.6 | 98.3x |
| `IQ4_XS` | 0.254 | 803.6 | 72.5x |
| `TQ1_0` | 0.237 | 802.4 | 81.6x |
| `TQ2_0` | 0.239 | 802.9 | 87.9x |

## CUDA Quantization

Representative latest development results on RTX 3090 for shape `11008x4096`:

| qtype | ms | traffic GB/s | note |
| --- | ---: | ---: | --- |
| `Q4_0` | ~0.349 | ~589 | very fast |
| `Q8_0` | ~0.349 | ~654 | very fast |
| `Q4_K` | ~1.534 | ~134 | strong K-family result |
| `Q5_K` | ~1.250 | ~169 | strong K-family result |
| `Q6_K` | ~1.007 | ~216 | strong K-family result |
| `Q3_K` | ~0.664 | ~301 | fast K-family result |
| `TQ1_0` | ~0.310 | ~612 | very fast |
| `TQ2_0` | ~0.336 | ~572 | very fast |
| `MXFP4` | ~0.723 | ~283 | strong |
| `NVFP4` | ~0.396 | ~519 | very fast |
| `IQ1_S` | ~14.778 | | exact on checked rows; optimization frontier |
| `IQ1_M` | ~25.260 | | exact on checked rows; optimization frontier |
| `IQ2_XXS` | ~31.463 | | exact on checked rows; optimization frontier |
| `IQ2_XS` | ~88.308 | | exact on checked rows; optimization frontier |
| `IQ3_XXS` | ~47.063 | | exact on checked rows; optimization frontier |
| `IQ3_S` | ~33.158 | | exact on checked rows; optimization frontier |

Q/K/TQ/MX/NV kernels are already fast. IQ quant kernels have improved significantly, are exact on checked rows, and remain the active CUDA optimization frontier.

## Metrics

| Metric | Meaning |
| --- | --- |
| `ms` | Average kernel/runtime duration in milliseconds. |
| `traffic_gb_s` | Estimated decoded plus encoded byte traffic per second for CUDA quant benchmarks. |
| `roofline_pct` | `traffic_gb_s / 936.2 GB/s * 100` for the RTX 3090 roofline used by the benchmark. |
| `blocks_per_s` | GGUF quantization blocks processed per second. |
| CPU default speedup | Ratio versus the build-configured native CPU backend in the benchmark run. |
| reg/stack | Register and local stack usage from compiler/disassembly tooling, when collected. |

Resource data such as register count and stack usage is typically collected from `ptxas`/CUDA binary inspection rather than from the Python benchmark itself.
