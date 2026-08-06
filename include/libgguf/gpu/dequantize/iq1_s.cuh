#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s(
    const void * vx, dst_t * yy, const int64_t i, const int64_t tid) {
    const block_iq1_s * x = (const block_iq1_s  *) vx;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
    const float d = __half2float(load_half(x[i].d)) * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    const uint64_t grid = iq1s_grid[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    const int8_t * q = (const int8_t *)&grid;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_iq1_s(vx, yy + blockIdx.x*QK_K, blockIdx.x, threadIdx.x);
}

}
}
