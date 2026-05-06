#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q4_1(const float * __restrict__ x, block_q4_1 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK4_1;
    float min = 3.4028234663852886e38f;
    float max = -3.4028234663852886e38f;
    for (int j = 0; j < QK4_1; ++j) {
        const float v = xb[j];
        if (v < min) {
            min = v;
        }
        if (v > max) {
            max = v;
        }
    }

    const float d = __fdiv_rn(max - min, 15.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
    y[ib].m = gguf_cuda_compute_fp32_to_fp16(min);

    for (int j = 0; j < QK4_1 / 2; ++j) {
        const float x0 = __fmul_rn(xb[j] - min, id);
        const float x1 = __fmul_rn(xb[QK4_1 / 2 + j] - min, id);

        const uint8_t xi0 = gguf_cuda_min_u8(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = gguf_cuda_min_u8(15, (int8_t)(x1 + 0.5f));

        y[ib].qs[j] = xi0 | (xi1 << 4);
    }
}


void gguf_cuda_quantize_launch_q4_1(
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
        const int64_t n_blocks = k / QK4_1;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_q4_1<<<blocks, threads, 0, stream>>>(x, (block_q4_1 *)y, n_blocks);
}
