#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ __forceinline__ float quantize_q8_0(src_t value, int8_t & q)
{
    const float v = static_cast<float>(value);
    const auto mask = __activemask();
    const float amax = __shfl_sync(mask, warp_reduce_max(fabsf(v)), 0);
    const float d = __fdiv_rn(amax, 127.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    q = static_cast<int8_t>(roundf(__fmul_rn(v, id)));
    return d;
}

template<typename src_t>
static __global__ void quantize_block_q8_0(
    const src_t * __restrict__ x,
    block_q8_0 * __restrict__ y,
    int64_t n_blocks)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
    if (ib >= n_blocks)
    {
        return;
    }

    int8_t q;
    const float d = quantize_q8_0(x[ib * QK8_0 + lane], q);
    if (lane == 0)
    {
        y[ib].d = compute_fp32_to_fp16(d);
    }
    y[ib].qs[lane] = q;
}

}
}
