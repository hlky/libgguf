#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_nvfp4(const float * __restrict__ x, block_nvfp4 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb0 = x + ib * QK_NVFP4;
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
        const float * xb = xb0 + s * QK_NVFP4_SUB;
        float amax = 0.0f;
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = fmaxf(amax, fabsf(xb[j]));
        }
        const uint8_t ue = gguf_cuda_fp32_to_ue4m3(amax / 6.0f);
        y[ib].d[s] = ue;
        const float d = gguf_cuda_ue4m3_to_fp32(ue);
        for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            const uint8_t x0 = gguf_cuda_best_index_mxfp4(xb[j], d);
            const uint8_t x1 = gguf_cuda_best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);
            y[ib].qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


void gguf_cuda_quantize_launch_nvfp4(
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
        const int64_t n_blocks = k / QK_NVFP4;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_nvfp4<<<blocks, threads, 0, stream>>>(x, (block_nvfp4 *)y, n_blocks);
}
