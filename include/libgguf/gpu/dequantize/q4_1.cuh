#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __half2float(low_half(x[ib].dm));
    const dfloat m = __half2float(high_half(x[ib].dm));

    const int vui = x[ib].qs[iqs];

    v.x = (vui & 0xF) * d + m;
    v.y = (vui >> 4) * d + m;
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    if (i >= k) {
        return;
    }

    const int ib = i/QK4_1;
    const int iqs = (i%QK4_1)/QR4_1;
    const int iybs = i - i%QK4_1;
    dfloat2 v;
    dequantize_q4_1(vx, ib, iqs, v);
    y[iybs + iqs] = convert_from_float<dst_t>(v.x);
    y[iybs + iqs + QK4_1/2] = convert_from_float<dst_t>(v.y);
}

}
}
