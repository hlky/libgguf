#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ __forceinline__ void quantize_q4_0(const src_t * x, block_q4_0 & y)
{
    float amax = 0.0f;
    float max_value = 0.0f;
    for (int j = 0; j < QK4_0; ++j)
    {
        const float value = static_cast<float>(x[j]);
        const float abs_value = fabsf(value);
        if (amax < abs_value)
        {
            amax = abs_value;
            max_value = value;
        }
    }

    const float d = __fdiv_rn(max_value, -8.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y.d = compute_fp32_to_fp16(d);

    for (int j = 0; j < QK4_0 / 2; ++j)
    {
        const float x0 = __fmul_rn(static_cast<float>(x[j]), id);
        const float x1 = __fmul_rn(static_cast<float>(x[QK4_0 / 2 + j]), id);
        const uint8_t xi0 = min_u8(15, static_cast<int8_t>(x0 + 8.5f));
        const uint8_t xi1 = min_u8(15, static_cast<int8_t>(x1 + 8.5f));
        y.qs[j] = xi0 | (xi1 << 4);
    }
}

template<typename src_t>
static __global__ void quantize_block_q4_0(
    const src_t * __restrict__ x,
    block_q4_0 * __restrict__ y,
    int64_t n_blocks)
{
    const int64_t ib = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks)
    {
        return;
    }
    quantize_q4_0(x + ib * QK4_0, y[ib]);
}

}
}
