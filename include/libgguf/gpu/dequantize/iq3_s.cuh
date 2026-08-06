#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(
    const void * vx, dst_t * yy, const int i, const int tid) {
    const block_iq3_s * x = (const block_iq3_s *) vx;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xs_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3xs_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = __half2float(load_half(x[i].d)) * (1.0f + 2.0f * ((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        const float v1 = grid1[j];
        const float v2 = grid2[j];
        y[j+0] = d * v1 * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * v2 * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_iq3_s(vx, yy + blockIdx.x*QK_K, blockIdx.x, threadIdx.x);
}

}
}
