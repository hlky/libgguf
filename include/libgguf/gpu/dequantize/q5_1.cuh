#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __half2float(low_half(x[ib].dm));
    const dfloat m = __half2float(high_half(x[ib].dm));

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0) * d + m;
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1) * d + m;
}

template<typename dst_t>
static __global__ void dequantize_block_q5_1(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    if (i >= k) {
        return;
    }

    const int ib = i/QK5_1;
    const int iqs = (i%QK5_1)/QR5_1;
    const int iybs = i - i%QK5_1;
    dfloat2 v;
    dequantize_q5_1(vx, ib, iqs, v);
    y[iybs + iqs] = convert_from_float<dst_t>(v.x);
    y[iybs + iqs + QK5_1/2] = convert_from_float<dst_t>(v.y);
}

}
}
