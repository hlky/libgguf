# CUDA Q3_K warp quant benchmark

| qtype | shape | baseline ms | new ms | delta | input GB/s | exact |
|---|---:|---:|---:|---:|---:|---|
| Q3_K | 4096x4096 | 1.113190 | 0.376960 | -66.14% | 178.026662 | True |
| Q3_K | 11008x4096 | 2.453299 | 0.884654 | -63.94% | 203.870808 | True |

Baseline source: `bench/results/cuda_cpu_quant_dequant_20260506T160633Z/results.csv.`