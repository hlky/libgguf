#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_iq4_xs(const float * __restrict__ x, block_iq4_xs * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    float weight[32];
    float scales[QK_K / 32];
    uint8_t l[QK_K];
    gguf_cuda_quantize_iq4_nl_impl(
        QK_K, 32, x + iblock * QK_K, &y[iblock].d, y[iblock].qs, &y[iblock].scales_h, y[iblock].scales_l,
        scales, weight, l, 7);
}


void gguf_cuda_quantize_launch_iq4_xs(
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
        quantize_block_iq4_xs<<<blocks, threads, 0, stream>>>(x, (block_iq4_xs *)y, n_blocks);
}
