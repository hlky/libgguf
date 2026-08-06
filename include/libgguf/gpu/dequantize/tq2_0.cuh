#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename dst_t>
static __device__ __forceinline__ void dequantize_tq2_0(
    const void * vx, dst_t * yb, const int ib, const int byte) {
    const block_tq2_0 * x = (const block_tq2_0 *) vx;
    const int group = byte / 32;
    const int local = byte - group * 32;
    const uint8_t packed = x[ib].qs[byte];
    const float d = __half2float(load_half(x[ib].d));
    yb += group * 128 + local;
#pragma unroll
    for (int plane = 0; plane < 4; ++plane) {
        const int q = ((packed >> (2 * plane)) & 3) - 1;
        yb[plane * 32] = convert_from_float<dst_t>(d * q);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_tq2_0(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int packed_index = blockDim.x*blockIdx.x + threadIdx.x;
    const int n_blocks = k / QK_K;
    if (packed_index >= n_blocks * (int)sizeof(block_tq2_0::qs)) {
        return;
    }
    const int threads_per_block = (int)sizeof(block_tq2_0::qs);
    const int ib = packed_index / threads_per_block;
    const int byte = packed_index - ib * threads_per_block;
    dequantize_tq2_0(vx, y + ib*QK_K, ib, byte);
}

}
}
