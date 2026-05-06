#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q8_0_warp(const float * __restrict__ x, block_q8_0 * __restrict__ y, int64_t n_blocks) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_block + warp;
    if (ib >= n_blocks) {
        return;
    }

    constexpr unsigned mask = 0xffffffffu;
    const float v = x[ib * QK8_0 + lane];
    const float amax = __shfl_sync(mask, gguf_cuda_warp_reduce_max(fabsf(v)), 0);
    const float d = __fdiv_rn(amax, 127.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    if (lane == 0) {
        y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
    }
    y[ib].qs[lane] = (int8_t)roundf(__fmul_rn(v, id));
}


void gguf_cuda_quantize_launch_q8_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int q_warp_threads = 128;
        constexpr int q_blocks_per_cta = q_warp_threads / 32;
        const int64_t n_blocks = k / QK8_0;
        const int blocks = (int)((n_blocks + q_blocks_per_cta - 1) / q_blocks_per_cta);
        quantize_block_q8_0_warp<<<blocks, q_warp_threads, 0, stream>>>(x, (block_q8_0 *)y, n_blocks);
}
