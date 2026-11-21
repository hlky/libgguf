#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ void quantize_iq4_xs(
    const src_t * x,
    block_iq4_xs * y,
    int64_t n_blocks, const float * importance, int64_t blocks_per_row)
{
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks)
    {
        return;
    }

    float weight[32];
    float scales[QK_K / 32];
    uint8_t l[QK_K];
    quantize_iq4_nl_impl(
        QK_K, 32, x + iblock * QK_K, &y[iblock].d, y[iblock].qs,
        &y[iblock].scales_h, y[iblock].scales_l, scales, weight, l, 7, importance ? importance + (iblock % blocks_per_row) * QK_K : nullptr);
}

template<typename src_t>
static __global__ void quantize_block_iq4_xs(
    const src_t * __restrict__ x,
    block_iq4_xs * __restrict__ y,
    int64_t n_blocks, const float * importance, int64_t blocks_per_row)
{
    quantize_iq4_xs(x, y, n_blocks, importance, blocks_per_row);
}

}
}
