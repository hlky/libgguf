#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ __forceinline__ void quantize_nvfp4(const src_t * x, block_nvfp4 & y)
{
    const auto mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int s = lane >> 3;
    const int j = lane & 7;
    const src_t * xb = x + s * QK_NVFP4_SUB;
    float amax = fmaxf(
        fabsf(static_cast<float>(xb[j])),
        fabsf(static_cast<float>(xb[QK_NVFP4_SUB / 2 + j])));
    amax = fmaxf(amax, __shfl_down_sync(mask, amax, 4, 8));
    amax = fmaxf(amax, __shfl_down_sync(mask, amax, 2, 8));
    amax = fmaxf(amax, __shfl_down_sync(mask, amax, 1, 8));
    amax = __shfl_sync(mask, amax, s * 8, libgguf::gpu::subgroup32::width);

    const uint8_t ue = fp32_to_ue4m3(amax / 6.0f);
    if (j == 0)
    {
        y.d[s] = ue;
    }
    const float d = ue4m3_to_fp32(ue);
    const uint8_t x0 = best_index_mxfp4(static_cast<float>(xb[j]), d);
    const uint8_t x1 = best_index_mxfp4(static_cast<float>(xb[QK_NVFP4_SUB / 2 + j]), d);
    y.qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
}

template<typename src_t>
static __global__ void quantize_block_nvfp4(
    const src_t * __restrict__ x,
    block_nvfp4 * __restrict__ y,
    int64_t n_blocks)
{
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
    if (ib >= n_blocks)
    {
        return;
    }
    quantize_nvfp4(x + ib * QK_NVFP4, y[ib]);
}

}
}
