#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_k(
    const void * vx, dst_t * yy, const int i, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + 64*il + n*ir;

    const float dall = __half2float(low_half(x[i].dm));
    const float dmin = __half2float(high_half(x[i].dm));

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = convert_from_float<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = convert_from_float<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_q4_k(vx, yy + blockIdx.x*QK_K, blockIdx.x, threadIdx.x);
}

}
}
