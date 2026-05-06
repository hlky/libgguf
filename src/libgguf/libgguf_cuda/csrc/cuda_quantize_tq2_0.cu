#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __device__ __forceinline__ int gguf_cuda_tq_round_away(float v) {
    const int abs_i = (int)(fabsf(v) + 0.5f);
    return v < 0.0f ? -abs_i : abs_i;
}

static __global__ void quantize_block_tq2_0_warp(const float * __restrict__ x, block_tq2_0 * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_cta = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_cta + warp;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_K;
    float amax = 0.0f;
    for (int j = lane; j < QK_K; j += 32) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }
    amax = __shfl_sync(mask, gguf_cuda_warp_reduce_max(amax), 0);

    const float d = amax;
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    if (lane == 0) {
        y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < (int)sizeof(block_tq2_0::qs); j += 32) {
        const int offset = 4 * j;
        uint8_t q = 0;
        for (int n = 0; n < 4; ++n) {
            const int xi = gguf_cuda_tq_round_away(__fmul_rn(xb[offset + lane + n * 32], id)) + 1;
            q += (xi & 3) << (2 * n);
        }
        y[ib].qs[j + lane] = q;
    }
}


void gguf_cuda_quantize_launch_tq2_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int tq2_threads = 128;
    constexpr int tq2_blocks_per_cta = tq2_threads / 32;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + tq2_blocks_per_cta - 1) / tq2_blocks_per_cta);
    quantize_block_tq2_0_warp<<<blocks, tq2_threads, 0, stream>>>(x, (block_tq2_0 *)y, n_blocks);
}
