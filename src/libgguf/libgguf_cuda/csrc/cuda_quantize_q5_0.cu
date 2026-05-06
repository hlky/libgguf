#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q5_0_warp(const float * __restrict__ x, block_q5_0 * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_block + warp;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK5_0;
    const float max = gguf_cuda_warp_reduce_absmax_first(xb[lane], lane);
    const float d = __fdiv_rn(max, -16.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    uint8_t xi0 = 0;
    uint8_t xi1 = 0;
    if (lane < QK5_0 / 2) {
        const float x0 = __fmul_rn(xb[lane], id);
        const float x1 = __fmul_rn(xb[QK5_0 / 2 + lane], id);
        xi0 = gguf_cuda_min_u8(31, (int8_t)(x0 + 16.5f));
        xi1 = gguf_cuda_min_u8(31, (int8_t)(x1 + 16.5f));
        y[ib].qs[lane] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
    }
    const uint32_t qh0 = __ballot_sync(mask, lane < QK5_0 / 2 && (xi0 & 0x10u)) & 0xffffu;
    const uint32_t qh1 = (__ballot_sync(mask, lane < QK5_0 / 2 && (xi1 & 0x10u)) & 0xffffu) << 16;
    if (lane == 0) {
        y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
        gguf_cuda_store_u32_le(y[ib].qh, qh0 | qh1);
    }
}


void gguf_cuda_quantize_launch_q5_0(
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
        const int64_t n_blocks = k / QK5_0;
        const int blocks = (int)((n_blocks + q_blocks_per_cta - 1) / q_blocks_per_cta);
        quantize_block_q5_0_warp<<<blocks, q_warp_threads, 0, stream>>>(x, (block_q5_0 *)y, n_blocks);
}
