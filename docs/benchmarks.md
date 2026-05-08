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
`summary.csv` under `bench/results/<timestamp>/`. Use `--output-root /tmp` to
put generated GGUFs on a faster scratch filesystem while keeping summaries under
`bench/results`, and use `--delete-outputs` to remove each GGUF after its size
and timings are recorded. Reports include
native timing fields when printed by the converter (`read`, `cpu_convert`,
`h2d`, `cuda_quant`, `d2h`, `write`, and `total`), Python wall time, output file
size, tensor qtype counts, fallback counts, stdout/stderr, and the exact command
used for each run. CUDA converter timing summaries also record `cuda_chunks`,
`cuda_pipeline`, `cuda_vram_bytes`, `cuda_max_input_bytes`, and
`cuda_max_output_bytes` when those fields are printed by the native converter.

Use a local safetensors path for FLUX.1-dev or any other model. The benchmark
does not download model files automatically; place the file on local storage
first and point `--src` at it. Use `--run-name local_flux_q4km` when you want a
stable results directory name, and keep ad hoc machine-specific outputs local
unless they are intentionally promoted into curated benchmark artifacts.

For CPU/CUDA converter comparisons, `--backend` is a report label:

```bash
python bench/conversion_bench.py \
  --src /models/model.safetensors \
  --qtype Q4_K_M \
  --backend native \
  --runs 3
```

Run the same command with CUDA converter flags and `--backend cuda` so the
JSON/CSV rows are comparable.
Converter-specific flags can be placed after `--`, which is easier than
repeating `--converter-arg` for flags that also begin with `--`:

```bash
python bench/conversion_bench.py \
  --src /models/model.safetensors \
  --qtype Q4_K_M \
  --backend cuda \
  --runs 3 \
  -- --backend cuda --cuda-batch-mb 1024 --cuda-pipeline 1
```

Useful CUDA converter flags for benchmark and correctness sweeps:

- `--cuda-batch-mb N`: sets the CUDA planning budget in MiB; stderr timing
  output reports the derived `cuda_vram` byte budget plus the largest input and
  output staging buffers as `cuda_max_input` and `cuda_max_output`.
- `--cuda-pipeline 0|1`: selects the CUDA scheduling path; the benchmark records
  the emitted `cuda_pipeline` value beside `cuda_chunks` so pipeline variants can
  be compared directly.
- `--verify-cuda-tensors all`: verifies every CUDA-routed tensor against the CPU
  encoding path.
- `--verify-cuda-large-tensors N`: additionally verifies the `N` largest
  CUDA-routed tensors by encoded byte size.
- `--verify-cuda-random-tensors N --seed S`: additionally verifies a stable
  seeded random sample of CUDA-routed tensors.

When comparing CPU and CUDA aggregate JSON files with
`python bench/conversion_bench.py compare`, joined JSON includes CUDA chunk,
pipeline, VRAM budget, and maximum buffer fields when present. Markdown
comparison output adds a CUDA execution details table for rows that contain
those fields. Older aggregate files that do not contain these optional fields
remain valid inputs.

### Flux CUDA Pipeline Benchmark

The following full-model run was captured on 2026-05-07 with
`/workspace/models/flux1-dev/flux1-dev.safetensors`, `Q4_K_M`, dynamic policy,
`--backend cuda --cuda-fallback cpu --cuda-batch-mb 1024`, and one run per
pipeline mode. Outputs were written under `/tmp` and deleted after timing and
size capture. This benchmark isolates the value of the existing two-slot CUDA
host pipeline on a real conversion workload; it is a one-run development
snapshot, so storage cache/order can still affect totals.

Artifacts:

- `bench/results/flux1_dev_cuda_q4km_pipeline0_20260507T205848Z/summary.json`
- `bench/results/flux1_dev_cuda_q4km_pipeline1_20260507T205848Z/summary.json`

| mode | total s | wall s | read s | cpu convert s | h2d s | cuda quant s | d2h s | write s | cuda chunks | output GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `--cuda-pipeline 0` | 108.144 | 108.299 | 54.782 | 3.804 | 3.953 | 0.435 | 1.066 | 1.303 | 304 | 7.16 |
| `--cuda-pipeline 1` | 29.284 | 29.642 | 16.947 | 3.579 | 2.025 | 0.412 | 0.336 | 1.289 | 304 | 7.16 |

Pipeline-on speedup for this run was `3.69x` by native `total_s` and `3.65x`
by Python wall time. The result shows the current host pipeline is materially
worth keeping and benchmarking before deeper cross-tensor batching work.

## Recommended Converter Backend By Qtype

These recommendations are grounded in the curated Flux CPU-vs-CUDA conversion
artifact at
`bench/results/flux1_dev_cpu_vs_cuda_qtypes_20260507T1745Z/summary.md`.
The measured numbers are representative development results, not universal
performance claims; storage, model mix, hardware, and build options can shift
end-to-end totals.

`--backend auto` encodes these current recommendations. It falls back to CPU
when CUDA is unavailable, unsupported for the selected qtype, or not linked into
the native converter.

| qtype family | Recommended backend | Notes |
| --- | --- | --- |
| Simple Q: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` | CPU | Flux comparison shows CPU faster for sampled simple Q qtypes such as `Q4_0` and `Q8_0`. |
| K qtypes: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` | CUDA when available; otherwise `auto`/CPU fallback | Flux comparison shows CUDA encode wins for sampled K qtypes. File-type aliases such as `Q4_K_M` and `Q5_K_M` route through the K-family recommendation. |
| IQ, TQ, MX, NV families | Experimental / depends | CUDA kernels exist for several qtypes, but native converter support and policy coverage should be checked before treating CUDA as the default. |
| Storage types: `F16`, `BF16`, `F32` | CPU / direct storage | These are storage/direct-copy paths rather than CUDA quantization wins. |

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
