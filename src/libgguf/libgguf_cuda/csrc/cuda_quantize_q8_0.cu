#include "libgguf_gpu_compat.h"

#include "cuda_quantize_q8_0.cuh"
#include "cuda_quantize_kernels.h"

template <typename input_t>
static __global__ void quantize_block_q8_0_warp(const input_t * __restrict__ x, block_q8_0 * __restrict__ y, int64_t n_blocks) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_block + warp;
    if (ib >= n_blocks) {
        return;
    }

    int8_t quantized = 0;
    const float d = gguf_cuda_quantize_q8_0_lane(x[ib * QK8_0 + lane], &quantized);
    if (lane == 0) {
        y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
    }
    y[ib].qs[lane] = quantized;
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
        quantize_block_q8_0_warp<float><<<blocks, q_warp_threads, 0, stream>>>(x, (block_q8_0 *)y, n_blocks);
}

void gguf_cuda_quantize_launch_q8_0_bf16(
    const void * x,
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
    quantize_block_q8_0_warp<gguf_gpu_bfloat16><<<blocks, q_warp_threads, 0, stream>>>(
        static_cast<const gguf_gpu_bfloat16 *>(x), (block_q8_0 *)y, n_blocks);
}
