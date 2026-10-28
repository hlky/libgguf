#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ __forceinline__ void quantize_mxfp4(const src_t * x, block_mxfp4 & y)
{
    const auto mask = __activemask();
    const int lane = threadIdx.x & 31;
    float amax = warp_reduce_max(fabsf(static_cast<float>(x[lane])));
    amax = __shfl_sync(mask, amax, 0, libgguf::gpu::subgroup32::width);

    const uint8_t e = amax > 0.0f ? static_cast<uint8_t>(floorf(log2f(amax)) - 2.0f + 127.0f) : 0;
    const float d = e8m0_to_fp32_half(e);
    if (lane == 0)
    {
        y.e = e;
    }

    if (lane < QK_MXFP4 / 2)
    {
        const uint8_t x0 = best_index_mxfp4(static_cast<float>(x[lane]), d);
        const uint8_t x1 = best_index_mxfp4(static_cast<float>(x[QK_MXFP4 / 2 + lane]), d);
        y.qs[lane] = x0 | (x1 << 4);
    }
}

template<typename src_t>
static __global__ void quantize_block_mxfp4(
    const src_t * __restrict__ x,
    block_mxfp4 * __restrict__ y,
    int64_t n_blocks)
{
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
    if (ib >= n_blocks)
    {
        return;
    }
    quantize_mxfp4(x + ib * QK_MXFP4, y[ib]);
}

}
}
