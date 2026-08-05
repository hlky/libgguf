#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ void quantize_iq4_nl(
    const src_t * x,
    block_iq4_nl * y,
    int64_t n_blocks)
{
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks)
    {
        return;
    }

    float scale;
    float weight[QK4_NL];
    uint8_t l[QK4_NL];
    uint16_t scales_h_unused;
    quantize_iq4_nl_impl(
        QK4_NL, 32, x + iblock * QK4_NL, &y[iblock].d, y[iblock].qs,
        &scales_h_unused, nullptr, &scale, weight, l, 7);
}

template<typename src_t>
static __global__ void quantize_block_iq4_nl(
    const src_t * __restrict__ x,
    block_iq4_nl * __restrict__ y,
    int64_t n_blocks)
{
    quantize_iq4_nl(x, y, n_blocks);
}

}
}
