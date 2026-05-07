# Flux1-dev CPU vs CUDA Conversion Qtype Comparison

- Source: `/workspace/models/flux1-dev/flux1-dev.safetensors`
- Policy: dynamic
- Runs: 1 per qtype per backend
- Outputs: written under `/tmp` and deleted after size/timing capture
- Caveat: storage cache/order affects `read_s` and end-to-end totals; compare `encode_s` for converter work.

| qtype | CPU total s | CUDA total s | total speedup | CPU encode s | CUDA encode s | encode speedup | output GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q2_K` | 110.838 | 71.283 | 1.55x | 61.433 | 6.372 | 9.64x | 4.2 |
| `Q3_K_M` | 64.647 | 29.641 | 2.18x | 13.978 | 6.285 | 2.22x | 5.58 |
| `Q4_0` | 15.733 | 30.087 | 0.52x | 3.715 | 6.093 | 0.61x | 6.8 |
| `Q4_K_M` | 112.513 | 30.083 | 3.74x | 99.742 | 6.447 | 15.47x | 7.16 |
| `Q5_K_M` | 105.191 | 31.587 | 3.33x | 56.193 | 6.694 | 8.39x | 8.74 |
| `Q6_K` | 49.23 | 35.096 | 1.4x | 35.611 | 6.745 | 5.28x | 10.2 |
| `Q8_0` | 17.781 | 34.354 | 0.52x | 4.432 | 6.63 | 0.67x | 12.71 |

Saved artifacts:

- CPU aggregate: `../flux1_dev_cpu_qtypes_20260507T1745Z/aggregate.json` and `aggregate.csv`
- CUDA aggregate: `../flux1_dev_cuda_qtypes_20260507T1732Z/aggregate.json` and `aggregate.csv`
- Joined comparison: `aggregate.json`, `aggregate.csv`, and this `summary.md`
