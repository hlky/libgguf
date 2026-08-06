#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(
    const void * vx, dst_t * yy, const int i, const int tid) {
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = __half2float(load_half(x[ib].d));
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_iq4_nl(vx, yy + blockIdx.x*QK_K, blockIdx.x, threadIdx.x);
}

}
}
