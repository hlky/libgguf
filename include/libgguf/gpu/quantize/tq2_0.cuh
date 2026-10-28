#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ void quantize_tq2_0(
    const src_t * x,
    block_tq2_0 * y,
    int64_t n_blocks)
{
    const auto mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_cta = blockDim.x >> 5;
    const int64_t ib = static_cast<int64_t>(blockIdx.x) * warps_per_cta + warp;
    if (ib >= n_blocks)
    {
        return;
    }

    const src_t * xb = x + ib * QK_K;
    float amax = 0.0f;
    for (int j = lane; j < QK_K; j += 32)
    {
        amax = fmaxf(amax, fabsf(static_cast<float>(xb[j])));
    }
    amax = __shfl_sync(mask, warp_reduce_max(amax), 0, libgguf::gpu::subgroup32::width);

    const float d = amax;
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    if (lane == 0)
    {
        y[ib].d = compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < static_cast<int>(sizeof(block_tq2_0::qs)); j += 32)
    {
        const int offset = 4 * j;
        uint8_t q = 0;
        for (int n = 0; n < 4; ++n)
        {
            const int xi = tq_round_away(
                __fmul_rn(static_cast<float>(xb[offset + lane + n * 32]), id)) + 1;
            q += (xi & 3) << (2 * n);
        }
        y[ib].qs[j + lane] = q;
    }
}

template<typename src_t>
static __global__ void quantize_block_tq2_0(
    const src_t * __restrict__ x,
    block_tq2_0 * __restrict__ y,
    int64_t n_blocks)
{
    quantize_tq2_0(x, y, n_blocks);
}

}
}
