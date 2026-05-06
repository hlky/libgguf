#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_mxfp4_warp(const float * __restrict__ x, block_mxfp4 * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_cta = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_cta + warp;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_MXFP4;
    float amax = gguf_cuda_warp_reduce_max(fabsf(xb[lane]));
    amax = __shfl_sync(mask, amax, 0);

    const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2.0f + 127.0f) : 0;
    const float d = gguf_cuda_e8m0_to_fp32_half(e);
    if (lane == 0) {
        y[ib].e = e;
    }

    if (lane < QK_MXFP4 / 2) {
        const uint8_t x0 = gguf_cuda_best_index_mxfp4(xb[lane], d);
        const uint8_t x1 = gguf_cuda_best_index_mxfp4(xb[QK_MXFP4 / 2 + lane], d);
        y[ib].qs[lane] = x0 | (x1 << 4);
    }
}


void gguf_cuda_quantize_launch_mxfp4(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int mxfp4_threads = 128;
    constexpr int mxfp4_blocks_per_cta = mxfp4_threads / 32;
    const int64_t n_blocks = k / QK_MXFP4;
    const int blocks = (int)((n_blocks + mxfp4_blocks_per_cta - 1) / mxfp4_blocks_per_cta);
    quantize_block_mxfp4_warp<<<blocks, mxfp4_threads, 0, stream>>>(x, (block_mxfp4 *)y, n_blocks);
}
