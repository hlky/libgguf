#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_tq2_0(const float * __restrict__ x, block_tq2_0 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_K;
    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }

    const float d = amax;
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);

    int offset = 0;
    for (size_t j = 0; j < sizeof(block_tq2_0::qs); j += 32) {
        for (size_t m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (size_t n = 0; n < 4; ++n) {
                const int xi = lroundf(__fmul_rn(xb[offset + m + n * 32], id)) + 1;
                q += (xi & 3) << (2 * n);
            }
            y[ib].qs[j + m] = q;
        }
        offset += 4 * 32;
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
    const int threads = 256;
        const int64_t n_blocks = k / QK_K;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_tq2_0<<<blocks, threads, 0, stream>>>(x, (block_tq2_0 *)y, n_blocks);
}
