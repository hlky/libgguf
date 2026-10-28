#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ __forceinline__ void quantize_q5_0(const src_t * x, block_q5_0 & y)
{
    const int lane = threadIdx.x & 31;
    const float max_value = warp_reduce_absmax_first(static_cast<float>(x[lane]), lane);
    const float d = __fdiv_rn(max_value, -16.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    uint8_t xi0 = 0;
    uint8_t xi1 = 0;
    if (lane < QK5_0 / 2)
    {
        const float x0 = __fmul_rn(static_cast<float>(x[lane]), id);
        const float x1 = __fmul_rn(static_cast<float>(x[QK5_0 / 2 + lane]), id);
        xi0 = min_u8(31, static_cast<int8_t>(x0 + 16.5f));
        xi1 = min_u8(31, static_cast<int8_t>(x1 + 16.5f));
        y.qs[lane] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
    }
    const uint32_t qh0 = subgroup32::ballot(lane < QK5_0 / 2 && (xi0 & 0x10u)) & 0xffffu;
    const uint32_t qh1 = (subgroup32::ballot(lane < QK5_0 / 2 && (xi1 & 0x10u)) & 0xffffu) << 16;
    if (lane == 0)
    {
        y.d = compute_fp32_to_fp16(d);
        store_u32_le(y.qh, qh0 | qh1);
    }
}

template<typename src_t>
static __global__ void quantize_block_q5_0(
    const src_t * __restrict__ x,
    block_q5_0 * __restrict__ y,
    int64_t n_blocks)
{
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
    if (ib >= n_blocks)
    {
        return;
    }
    quantize_q5_0(x + ib * QK5_0, y[ib]);
}

}
}
