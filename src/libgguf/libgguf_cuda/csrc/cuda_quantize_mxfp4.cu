#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_mxfp4(const float * __restrict__ x, block_mxfp4 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_MXFP4;
    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4; ++j) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }

    const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2.0f + 127.0f) : 0;
    const float d = gguf_cuda_e8m0_to_fp32_half(e);
    y[ib].e = e;

    for (int j = 0; j < QK_MXFP4 / 2; ++j) {
        const uint8_t x0 = gguf_cuda_best_index_mxfp4(xb[j], d);
        const uint8_t x1 = gguf_cuda_best_index_mxfp4(xb[QK_MXFP4 / 2 + j], d);
        y[ib].qs[j] = x0 | (x1 << 4);
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
    const int threads = 256;
        const int64_t n_blocks = k / QK_MXFP4;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_mxfp4<<<blocks, threads, 0, stream>>>(x, (block_mxfp4 *)y, n_blocks);
}
