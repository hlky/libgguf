#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ __forceinline__ void quantize_q1_0(const src_t * x, block_q1_0 & y)
{
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; ++j)
    {
        sum_abs += fabsf(static_cast<float>(x[j]));
    }
    y.d = compute_fp32_to_fp16(sum_abs / QK1_0);

    for (int j = 0; j < QK1_0 / 8; ++j)
    {
        uint8_t q = 0;
        for (int bit = 0; bit < 8; ++bit)
        {
            if (static_cast<float>(x[8 * j + bit]) >= 0.0f)
            {
                q |= 1 << bit;
            }
        }
        y.qs[j] = q;
    }
}

template<typename src_t>
static __global__ void quantize_block_q1_0(
    const src_t * __restrict__ x,
    block_q1_0 * __restrict__ y,
    int64_t n_blocks)
{
    const int64_t ib = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks)
    {
        return;
    }
    quantize_q1_0(x + ib * QK1_0, y[ib]);
}

}
}
