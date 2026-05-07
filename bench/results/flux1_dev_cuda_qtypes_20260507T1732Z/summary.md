# Flux1-dev CUDA Conversion Qtype Sweep

- Source: `/workspace/models/flux1-dev/flux1-dev.safetensors`
- Converter: `/workspace/libgguf/build/codex-worker-c-cuda-auto/libgguf_quantize_gguf`
- Backend/policy: CUDA with CPU fallback, dynamic policy
- Runs: 1 per qtype
- Output handling: GGUFs written under `/tmp/libgguf_conversion_outputs` and deleted after size/timing capture
- CUDA VRAM budget: `1073741824` bytes

One-run qtype comparisons are sensitive to storage cache state. `Q2_K` ran first and includes the coldest source-read path; later qtypes benefited from warmed reads. Use these as a saved broad sweep, not final rank ordering. `encode_s` is `cpu_convert_s + h2d_s + cuda_quant_s + d2h_s`.

| qtype | total s | wall s | read s | encode s | write s | output GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q2_K` | 71.283 | 71.622 | 53.729 | 6.372 | 0.837 | 4.2 |
| `Q3_K_M` | 29.641 | 29.974 | 17.598 | 6.285 | 1.053 | 5.58 |
| `Q4_0` | 30.087 | 30.424 | 17.772 | 6.093 | 1.256 | 6.8 |
| `Q4_K_M` | 30.083 | 30.431 | 17.426 | 6.447 | 1.336 | 7.16 |
| `Q5_K_M` | 31.587 | 31.938 | 18.17 | 6.694 | 1.71 | 8.74 |
| `Q6_K` | 35.096 | 35.478 | 18.068 | 6.745 | 5.192 | 10.2 |
| `Q8_0` | 34.354 | 34.738 | 17.865 | 6.63 | 4.438 | 12.71 |

Saved artifacts:

- `aggregate.json`
- `aggregate.csv`
- Per-qtype `summary.json` and `summary.csv` directories under this folder
