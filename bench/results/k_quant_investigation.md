# CUDA K-Quant Investigation

GPU target: NVIDIA GeForce RTX 3090. Roofline reference: 936.2 GB/s theoretical memory bandwidth.

Benchmark output:

- `bench/results/k_quant_investigation.csv`
- `bench/results/k_quant_investigation.json`

The benchmark computes:

- `traffic_gb_s = (decoded_bytes + encoded_bytes) / seconds`
- `roofline_pct = traffic_gb_s / 936.2 * 100`
- `blocks_per_s = gguf_256_value_blocks / seconds`

## Current Kernel Shape

`Q2_K`, `Q4_K`, `Q5_K`, and `Q6_K` still launch one CUDA thread per 256-value GGUF block. `Q3_K` is the exception after the previous warp-per-block rewrite. `cuobjdump` resource usage confirms the scalar K kernels still spill large per-thread stack frames:

| kernel | registers | stack |
|---|---:|---:|
| Q2_K scalar | 71 | 416 B |
| Q4_K scalar | 95 | 320 B |
| Q4_K templated scalar | 158 | 320 B |
| Q4_K templated no-laux | 190 | 320 B |
| Q5_K scalar | 95 | 336 B |
| Q6_K scalar | 85 | 336 B |
| Q3_K warp | 40 | 0 B |
| Q4_K cooperative prototype | 56 | 0 B |

## Production K-Quant Results

| qtype | 4096x4096 ms | 4096 roofline | 11008x4096 ms | 11008 roofline |
|---|---:|---:|---:|---:|
| Q2_K | 0.683927 | 11.34% | 1.529902 | 13.62% |
| Q3_K | 0.347535 | 22.84% | 0.941246 | 22.67% |
| Q4_K | 0.909710 | 8.99% | 2.153610 | 10.20% |
| Q5_K | 1.110648 | 7.56% | 2.482420 | 9.09% |
| Q6_K | 0.817860 | 10.56% | 1.972742 | 11.77% |

Q3_K is now the fastest K quant and reaches the highest traffic roofline percentage, which supports the warp-per-block direction.

## Q4_K Phase Variants

| variant | 4096x4096 ms | 4096 roofline | 11008x4096 ms | 11008 roofline | exact |
|---|---:|---:|---:|---:|---|
| baseline | 0.901907 | 9.07% | 2.098092 | 10.47% | ok |
| load_only | 0.260431 | 31.40% | 2.992993 | 7.34% | n/a |
| stats_only | 0.601069 | 13.60% | 1.521211 | 14.44% | n/a |
| quant_no_pack | 0.749670 | 10.91% | 1.829457 | 12.01% | n/a |
| cooperative | 21.873890 | 0.37% | 58.891415 | 0.37% | ok |
| templated_no_laux | 2.561948 | 3.19% | 6.273441 | 3.50% | ok |
| templated_laux | 2.500631 | 3.27% | 5.039197 | 4.36% | ok |
| warp_search | 4.075982 | 2.01% | 11.008883 | 2.00% | mismatch |
| warp_block_search | 4.036011 | 2.03% | 10.904375 | 2.02% | mismatch |
| lane_subgroups | 0.637838 | 12.82% | 1.712618 | 12.83% | ok |

The current Q4_K kernel is not memory-bound. If it were memory-bound, the full kernel would be much closer to the load-only path and to the 936.2 GB/s roofline. Instead, full Q4_K reaches about 9-10% of roofline, while stats-only and quant-no-pack are materially faster than the full kernel. The dominant cost is the scalar algorithmic work in the weighted `make_qkx2_quants` search plus final requantization/packing, not raw DRAM bandwidth.

The CTA-per-K-block cooperative prototype is byte-exact, but it is much slower. It removes local stack spilling, but launching one CTA per 256-value block with lane-0 subgroup searches leaves most lanes idle during the expensive search and creates far too many CTAs. This is a useful negative result: stack removal alone is not enough, and cooperative kernels must parallelize the subgroup search itself rather than only distribute subgroups across warps.

The templated scalar helpers are also byte-exact but slower. The `laux`-preserving template raises register use from 95 to 158, while the no-`laux` recompute variant raises it to 190; both still carry a 320 B stack frame. For Q4_K on this compiler/GPU, compile-time specialization and q recomputation increase pressure enough to lose badly. This suggests the safe scalar track is not the right next production path for Q4_K unless it is more selectively applied to smaller `N=16` Q2_K groups.

The true warp-search prototypes parallelize the inner 32-value subgroup search with warp reductions. They are not byte-exact on the benchmark data because the reduction order differs from the scalar helper, and they are slower on real shapes. The one-CTA-per-K-block version is dominated by CTA count; the one-warp-per-K-block version fixes CTA count but still spends about 4.1 ms on 4096x4096 and 11.0 ms on 11008x4096. This means a naive warp-reduction rewrite is not a production path either: it needs an exact reduction/order strategy and likely less per-candidate synchronization before it can compete.

The `lane_subgroups` variant is the first positive Q4_K structural result. It maps one warp to 32 independent subgroup searches, so each lane runs the original scalar `make_qkx2_quants` order for one 32-value subgroup. A warp therefore covers four Q4_K blocks, and an eight-warp CTA covers 32 Q4_K blocks. This preserves byte exactness while filling all lanes with useful scalar subgroup work. It improves Q4_K from 0.901907 ms to 0.637838 ms on 4096x4096 and from 2.098092 ms to 1.712618 ms on 11008x4096. This confirms the most promising exact direction is parallelizing independent scalar subgroup searches, not changing the reduction order inside one subgroup.

## Largest Opportunity

`Q5_K` is the largest practical optimization target by current runtime: it is the slowest K quant on both real-world shapes, and it shares the same `make_qkx2_quants` structure as Q4_K while adding high-bit packing. `Q4_K` remains the best representative prototype because it isolates the shared scale/min search and 4-bit packing without Q5 high-bit complexity.

`Q3_K` is the evidence that the direction can work: after converting it to a warp-per-block design, it is roughly 2.4x faster than Q4_K on 4096x4096 and has no local stack spill.

## Recommended Next Strategy

Do not pursue one-CTA-per-K-block with lane-0 serial subgroup work or a naive warp-reduction search. The next prototype should build on `lane_subgroups`: keep multiple GGUF blocks per CTA and parallelize independent scalar subgroup searches while preserving scalar accumulation/order inside each subgroup.

## Promoted Q4_K Baseline

The `lane_subgroups` design was promoted to the production Q4_K quantization kernel and the benchmark-only variants/API were removed. The production kernel uses 128 CUDA threads per CTA. A 128/256/512 thread sweep was effectively flat on the target shapes, so 128 was selected to keep shared-memory use lower while preserving the same throughput envelope.

| shape | old baseline ms | promoted ms | delta | traffic GB/s | exact |
|---|---:|---:|---:|---:|---|
| 64x4096 | 0.231510 | 0.030141 | -86.98% | 39.68 | ok |
| 4096x4096 | 0.901907 | 0.645208 | -28.46% | 118.64 | ok |
| 11008x4096 | 2.098092 | 1.688075 | -19.54% | 121.87 | ok |

`cuobjdump` reports the promoted Q4_K kernel at 127 registers and 0 B stack, compared with the scalar Q4_K baseline at 95 registers and 320 B stack. This is the tradeoff that won in practice: higher register pressure, but no local stack spill and much better lane utilization by assigning independent subgroup searches across warp lanes.

The generalized lesson for the remaining K quants is to keep byte-exact scalar subgroup search order within a lane, but schedule many independent subgroup searches per warp/CTA. Q5_K is the next best target because it is currently the slowest K quant and shares the Q4_K scale/min search structure, with extra high-bit packing that can be optimized after the lane-subgroup staging is in place.

1. Use the promoted Q4_K kernel as the reference implementation for the exact lane-subgroup pattern.
2. Keep the byte-exact `make_qkx2_quants` math and evaluation order for candidate scoring where required.
3. Apply the same lane-per-subgroup design to Q5_K next, using scalar subgroup search per lane and then cooperative high-bit packing.
4. Reuse the dynamic shared-memory staging layout for Q2_K/Q5_K/Q6_K where their subgroup structure fits.
5. Reduce packing/metadata serial sections only after the exact lane-subgroup schedule is established for each qtype.

The immediate success criterion should be beating Q4_K `quant_no_pack` plus packing overhead, not merely eliminating stack usage. The target is to move Q4_K from roughly 10% roofline toward Q3_K's current 23-24% traffic roofline while preserving byte-exact output.
