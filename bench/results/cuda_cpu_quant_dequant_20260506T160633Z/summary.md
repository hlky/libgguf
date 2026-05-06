# CUDA/CPU quant-dequant benchmark 20260506T160633Z
## Machine
```
CPU:
CPU(s): 256
On-line CPU(s) list: 0-255
Model name: AMD EPYC 7763 64-Core Processor
Thread(s) per core: 2
Core(s) per socket: 64
Socket(s): 2
CPU(s) scaling MHz: 51%
CPU max MHz: 3530.4929
NUMA node0 CPU(s): 0-63,128-191
NUMA node1 CPU(s): 64-127,192-255
GPU: NVIDIA GeForce RTX 3090, 580.126.20, 8.6, 24576 MiB
Torch CUDA: 12.8
```
## Selected real-world rows
| backend | op | threads | qtype | shape | ms | input GB/s | output GB/s | exact/status |
|---|---|---|---|---:|---:|---:|---:|---|
| cuda | quantize |  | IQ4_XS | 4096x4096 | 3.461581 | 19.386769 |  | ok |
| cuda | dequantize |  | IQ4_XS | 4096x4096 | 0.097075 | 91.814346 | 691.308016 | ok |
| libgguf_cpu | quantize | 1_thread | IQ4_XS | 4096x4096 | 4245.085799 | 0.015809 |  | ok |
| libgguf_cpu | dequantize | 1_thread | IQ4_XS | 4096x4096 | 26.113718 | 0.341311 | 2.569870 | ok |
| libgguf_cpu | quantize | default | IQ4_XS | 4096x4096 | 284.112019 | 0.236206 |  | ok |
| libgguf_cpu | dequantize | default | IQ4_XS | 4096x4096 | 25.587022 | 0.348337 | 2.622770 | ok |
| cuda | quantize |  | Q2_K | 4096x4096 | 0.724890 | 92.578046 |  | ok |
| cuda | dequantize |  | Q2_K | 4096x4096 | 0.091887 | 59.910847 | 730.341751 | ok |
| libgguf_cpu | quantize | 1_thread | Q2_K | 4096x4096 | 1095.780616 | 0.061243 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q2_K | 4096x4096 | 20.921776 | 0.263124 | 3.207608 | ok |
| libgguf_cpu | quantize | default | Q2_K | 4096x4096 | 25.000876 | 2.684260 |  | ok |
| libgguf_cpu | dequantize | default | Q2_K | 4096x4096 | 17.261993 | 0.318910 | 3.887666 | ok |
| cuda | quantize |  | Q3_K | 4096x4096 | 1.113190 | 60.285164 |  | ok |
| cuda | dequantize |  | Q3_K | 4096x4096 | 0.093116 | 77.419353 | 720.703799 | ok |
| libgguf_cpu | quantize | 1_thread | Q3_K | 4096x4096 | 104.824189 | 0.640204 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q3_K | 4096x4096 | 27.857065 | 0.258784 | 2.409043 | ok |
| libgguf_cpu | quantize | default | Q3_K | 4096x4096 | 16.383119 | 4.096220 |  | ok |
| libgguf_cpu | dequantize | default | Q3_K | 4096x4096 | 16.795526 | 0.429219 | 3.995639 | ok |
| cuda | quantize |  | Q4_0 | 4096x4096 | 0.156006 | 430.167389 |  | ok |
| cuda | dequantize |  | Q4_0 | 4096x4096 | 0.095710 | 98.602000 | 701.169775 | ok |
| libgguf_cpu | quantize | 1_thread | Q4_0 | 4096x4096 | 15.159789 | 4.426768 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q4_0 | 4096x4096 | 8.653356 | 1.090581 | 7.755241 | ok |
| libgguf_cpu | quantize | default | Q4_0 | 4096x4096 | 16.256470 | 4.128133 |  | ok |
| libgguf_cpu | dequantize | default | Q4_0 | 4096x4096 | 16.961386 | 0.556392 | 3.956567 | ok |
| cuda | quantize |  | Q4_K | 4096x4096 | 0.942989 | 71.166132 |  | ok |
| cuda | dequantize |  | Q4_K | 4096x4096 | 0.095642 | 98.672380 | 701.670257 | ok |
| libgguf_cpu | quantize | 1_thread | Q4_K | 4096x4096 | 1121.911024 | 0.059817 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q4_K | 4096x4096 | 9.396451 | 1.004335 | 7.141937 | ok |
| libgguf_cpu | quantize | default | Q4_K | 4096x4096 | 26.080088 | 2.573184 |  | ok |
| libgguf_cpu | dequantize | default | Q4_K | 4096x4096 | 19.696645 | 0.479126 | 3.407122 | ok |
| cuda | quantize |  | Q5_K | 4096x4096 | 1.161416 | 57.781937 |  | ok |
| cuda | dequantize |  | Q5_K | 4096x4096 | 0.097792 | 117.947647 | 686.240855 | ok |
| libgguf_cpu | quantize | 1_thread | Q5_K | 4096x4096 | 879.982735 | 0.076262 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q5_K | 4096x4096 | 9.699579 | 1.189158 | 6.918740 | ok |
| libgguf_cpu | quantize | default | Q5_K | 4096x4096 | 20.959716 | 3.201802 |  | ok |
| libgguf_cpu | dequantize | default | Q5_K | 4096x4096 | 17.199674 | 0.670614 | 3.901752 | ok |
| cuda | quantize |  | Q6_K | 4096x4096 | 0.838554 | 80.029306 |  | ok |
| cuda | dequantize |  | Q6_K | 4096x4096 | 0.100250 | 137.282941 | 669.417770 | ok |
| libgguf_cpu | quantize | 1_thread | Q6_K | 4096x4096 | 715.059108 | 0.093851 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q6_K | 4096x4096 | 25.494473 | 0.539825 | 2.632291 | ok |
| libgguf_cpu | quantize | default | Q6_K | 4096x4096 | 21.882629 | 3.066764 |  | ok |
| libgguf_cpu | dequantize | default | Q6_K | 4096x4096 | 19.416477 | 0.708808 | 3.456284 | ok |
| cuda | quantize |  | Q8_0 | 4096x4096 | 0.405555 | 165.474056 |  | ok |
| cuda | dequantize |  | Q8_0 | 4096x4096 | 0.104858 | 169.999994 | 639.999977 | ok |
| libgguf_cpu | quantize | 1_thread | Q8_0 | 4096x4096 | 9.447341 | 7.103466 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q8_0 | 4096x4096 | 8.222360 | 2.167965 | 8.161752 | ok |
| libgguf_cpu | quantize | default | Q8_0 | 4096x4096 | 19.854604 | 3.380015 |  | ok |
| libgguf_cpu | dequantize | default | Q8_0 | 4096x4096 | 16.984825 | 1.049513 | 3.951107 | ok |
| cuda | quantize |  | IQ4_XS | 11008x4096 | 8.388045 | 21.501444 |  | ok |
| cuda | dequantize |  | IQ4_XS | 11008x4096 | 0.252655 | 94.806809 | 713.839501 | ok |
| libgguf_cpu | quantize | 1_thread | IQ4_XS | 11008x4096 | 10374.922659 | 0.017384 |  | ok |
| libgguf_cpu | dequantize | 1_thread | IQ4_XS | 11008x4096 | 67.245825 | 0.356207 | 2.682026 | ok |
| libgguf_cpu | quantize | default | IQ4_XS | 11008x4096 | 792.986434 | 0.227438 |  | ok |
| libgguf_cpu | dequantize | default | IQ4_XS | 11008x4096 | 18.422664 | 1.300214 | 9.789847 | ok |
| cuda | quantize |  | Q2_K | 11008x4096 | 1.555456 | 115.949969 |  | ok |
| cuda | dequantize |  | Q2_K | 11008x4096 | 0.241801 | 61.185771 | 745.883689 | ok |
| libgguf_cpu | quantize | 1_thread | Q2_K | 11008x4096 | 2590.910772 | 0.069611 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q2_K | 11008x4096 | 54.770947 | 0.270120 | 3.292897 | ok |
| libgguf_cpu | quantize | default | Q2_K | 11008x4096 | 192.885421 | 0.935037 |  | ok |
| libgguf_cpu | dequantize | default | Q2_K | 11008x4096 | 19.543116 | 0.757031 | 9.228573 | ok |
| cuda | quantize |  | Q3_K | 11008x4096 | 2.453299 | 73.515320 |  | ok |
| cuda | dequantize |  | Q3_K | 11008x4096 | 0.245248 | 78.997912 | 735.398748 | ok |
| libgguf_cpu | quantize | 1_thread | Q3_K | 11008x4096 | 234.319805 | 0.769696 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q3_K | 11008x4096 | 71.661453 | 0.270356 | 2.516765 | ok |
| libgguf_cpu | quantize | default | Q3_K | 11008x4096 | 16.888166 | 10.679376 |  | ok |
| libgguf_cpu | dequantize | default | Q3_K | 11008x4096 | 17.869488 | 1.084199 | 10.092907 | ok |
| cuda | quantize |  | Q4_0 | 11008x4096 | 0.384355 | 469.240605 |  | ok |
| cuda | dequantize |  | Q4_0 | 11008x4096 | 0.252245 | 100.546686 | 714.998654 | ok |
| libgguf_cpu | quantize | 1_thread | Q4_0 | 11008x4096 | 33.343874 | 5.408942 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q4_0 | 11008x4096 | 22.599814 | 1.122241 | 7.980379 | ok |
| libgguf_cpu | quantize | default | Q4_0 | 11008x4096 | 16.786186 | 10.744256 |  | ok |
| libgguf_cpu | dequantize | default | Q4_0 | 11008x4096 | 18.557994 | 1.366658 | 9.718457 | ok |
| cuda | quantize |  | Q4_K | 11008x4096 | 2.161454 | 83.441535 |  | ok |
| cuda | dequantize |  | Q4_K | 11008x4096 | 0.252041 | 100.628387 | 715.579643 | ok |
| libgguf_cpu | quantize | 1_thread | Q4_K | 11008x4096 | 2734.307748 | 0.065960 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q4_K | 11008x4096 | 26.008948 | 0.975143 | 6.934347 | ok |
| libgguf_cpu | quantize | default | Q4_K | 11008x4096 | 199.304073 | 0.904924 |  | ok |
| libgguf_cpu | dequantize | default | Q4_K | 11008x4096 | 19.976993 | 1.269582 | 9.028139 | ok |
| cuda | quantize |  | Q5_K | 11008x4096 | 2.590370 | 69.625227 |  | ok |
| cuda | dequantize |  | Q5_K | 11008x4096 | 0.258799 | 119.778423 | 696.892641 | ok |
| libgguf_cpu | quantize | 1_thread | Q5_K | 11008x4096 | 2136.186709 | 0.084429 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q5_K | 11008x4096 | 24.576309 | 1.261317 | 7.338574 | ok |
| libgguf_cpu | quantize | default | Q5_K | 11008x4096 | 105.924880 | 1.702670 |  | ok |
| libgguf_cpu | dequantize | default | Q5_K | 11008x4096 | 20.628548 | 1.502700 | 8.742984 | ok |
| cuda | quantize |  | Q6_K | 11008x4096 | 1.939200 | 93.004884 |  | ok |
| cuda | dequantize |  | Q6_K | 11008x4096 | 0.269449 | 137.268811 | 669.348869 | ok |
| libgguf_cpu | quantize | 1_thread | Q6_K | 11008x4096 | 1729.489933 | 0.104282 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q6_K | 11008x4096 | 63.914500 | 0.578693 | 2.821818 | ok |
| libgguf_cpu | quantize | default | Q6_K | 11008x4096 | 98.328096 | 1.834217 |  | ok |
| libgguf_cpu | dequantize | default | Q6_K | 11008x4096 | 20.163052 | 1.834389 | 8.944830 | ok |
| cuda | quantize |  | Q8_0 | 11008x4096 | 1.053901 | 171.130973 |  | ok |
| cuda | dequantize |  | Q8_0 | 11008x4096 | 0.277231 | 172.804730 | 650.558983 | ok |
| libgguf_cpu | quantize | 1_thread | Q8_0 | 11008x4096 | 18.658543 | 9.666086 |  | ok |
| libgguf_cpu | dequantize | 1_thread | Q8_0 | 11008x4096 | 22.687223 | 2.111621 | 7.949632 | ok |
| libgguf_cpu | quantize | default | Q8_0 | 11008x4096 | 15.709534 | 11.480612 |  | ok |
| libgguf_cpu | dequantize | default | Q8_0 | 11008x4096 | 18.310646 | 2.616337 | 9.849738 | ok |

Full CSV/JSON contain all qtypes and shapes.
