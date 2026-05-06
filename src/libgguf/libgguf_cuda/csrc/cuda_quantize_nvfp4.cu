#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __device__ __forceinline__ float gguf_cuda_subwarp8_reduce_max(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value = fmaxf(value, __shfl_down_sync(mask, value, 4, 8));
    value = fmaxf(value, __shfl_down_sync(mask, value, 2, 8));
    value = fmaxf(value, __shfl_down_sync(mask, value, 1, 8));
    return value;
}

static __global__ void quantize_block_nvfp4_warp(const float * __restrict__ x, block_nvfp4 * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_cta = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_cta + warp;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb0 = x + ib * QK_NVFP4;
    const int s = lane >> 3;
    const int j = lane & 7;
    const float * xb = xb0 + s * QK_NVFP4_SUB;
    float amax = fmaxf(fabsf(xb[j]), fabsf(xb[QK_NVFP4_SUB / 2 + j]));
    amax = gguf_cuda_subwarp8_reduce_max(amax);
    amax = __shfl_sync(mask, amax, s * 8);

    const uint8_t ue = gguf_cuda_fp32_to_ue4m3(amax / 6.0f);
    if (j == 0) {
        y[ib].d[s] = ue;
    }
    const float d = gguf_cuda_ue4m3_to_fp32(ue);
    const uint8_t x0 = gguf_cuda_best_index_mxfp4(xb[j], d);
    const uint8_t x1 = gguf_cuda_best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);
    y[ib].qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
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
    constexpr int nvfp4_threads = 128;
    constexpr int nvfp4_blocks_per_cta = nvfp4_threads / 32;
    const int64_t n_blocks = k / QK_NVFP4;
    const int blocks = (int)((n_blocks + nvfp4_blocks_per_cta - 1) / nvfp4_blocks_per_cta);
    quantize_block_nvfp4_warp<<<blocks, nvfp4_threads, 0, stream>>>(x, (block_nvfp4 *)y, n_blocks);
}
