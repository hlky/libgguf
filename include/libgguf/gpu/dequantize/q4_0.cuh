#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = __half2float(load_half(x[ib].d));

    const int vui = x[ib].qs[iqs];

    v.x = ((vui & 0xF) - 8) * d;
    v.y = ((vui >> 4) - 8) * d;
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    if (i >= k) {
        return;
    }

    const int ib = i/QK4_0;
    const int iqs = (i%QK4_0)/QR4_0;
    const int iybs = i - i%QK4_0;
    dfloat2 v;
    dequantize_q4_0(vx, ib, iqs, v);
    y[iybs + iqs] = convert_from_float<dst_t>(v.x);
    y[iybs + iqs + QK4_0/2] = convert_from_float<dst_t>(v.y);
}

}
}
