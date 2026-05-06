#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q1_0(const float * __restrict__ x, block_q1_0 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK1_0;
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; ++j) {
        sum_abs += fabsf(xb[j]);
    }
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(sum_abs / QK1_0);

    for (int j = 0; j < QK1_0 / 8; ++j) {
        uint8_t q = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if (xb[8 * j + bit] >= 0.0f) {
                q |= 1 << bit;
            }
        }
        y[ib].qs[j] = q;
    }
}


void gguf_cuda_quantize_launch_q1_0(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    const int threads = 256;
        const int64_t n_blocks = k / QK1_0;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_q1_0<<<blocks, threads, 0, stream>>>(x, (block_q1_0 *)y, n_blocks);
}
