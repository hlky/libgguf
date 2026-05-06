# CUDA Q warp quant benchmark 20260506T172008Z
## Machine
```
CPU:
CPU(s):                                  256
On-line CPU(s) list:                     0-255
Model name:                              AMD EPYC 7763 64-Core Processor
Thread(s) per core:                      2
Core(s) per socket:                      64
Socket(s):                               2
CPU(s) scaling MHz:                      49%
CPU max MHz:                             3530.4929
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-63,128-191
NUMA node1 CPU(s):                       64-127,192-255
GPU: NVIDIA GeForce RTX 3090, 580.126.20, 8.6, 24576 MiB
Torch CUDA: 12.8
```
## Results
| qtype | shape | baseline ms | new ms | delta | input GB/s |
|---|---:|---:|---:|---:|---:|
| Q5_0 | 4096x4096 | 0.296806 | 0.212958 | -28.25% | 315.127436 |
| Q5_1 | 4096x4096 | 0.217549 | 0.187733 | -13.71% | 357.469093 |
| Q8_0 | 4096x4096 | 0.405555 | 0.137567 | -66.08% | 487.827002 |
| Q5_0 | 11008x4096 | 0.754432 | 0.566682 | -24.89% | 318.265265 |
| Q5_1 | 11008x4096 | 0.548813 | 0.499221 | -9.04% | 361.272772 |
| Q8_0 | 11008x4096 | 1.053901 | 0.365158 | -65.35% | 493.909143 |

Baseline source: `bench/results/cuda_cpu_quant_dequant_20260506T160633Z/results.csv.`
