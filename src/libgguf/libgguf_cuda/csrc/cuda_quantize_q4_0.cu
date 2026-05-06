#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q4_0(const float * __restrict__ x, block_q4_0 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK4_0;
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        const float v = xb[j];
        const float av = fabsf(v);
        if (amax < av) {
            amax = av;
            max = v;
        }
    }

    const float d = __fdiv_rn(max, -8.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);

    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = __fmul_rn(xb[j], id);
        const float x1 = __fmul_rn(xb[QK4_0 / 2 + j], id);

        const uint8_t xi0 = gguf_cuda_min_u8(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = gguf_cuda_min_u8(15, (int8_t)(x1 + 8.5f));

        y[ib].qs[j] = xi0 | (xi1 << 4);
    }
}


void gguf_cuda_quantize_launch_q4_0(
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
        const int64_t n_blocks = k / QK4_0;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_q4_0<<<blocks, threads, 0, stream>>>(x, (block_q4_0 *)y, n_blocks);
}
