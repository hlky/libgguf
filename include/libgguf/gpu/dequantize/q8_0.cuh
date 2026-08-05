#pragma once

#include <libgguf/gpu/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ void dequantize_q8_0(
    const void *vx,
    const int ib,
    const int iqs,
    dfloat2 &v)
{
    const block_q8_0 *x = (const block_q8_0 *)vx;
    const dfloat d = __half2float(load_half(x[ib].d));

    v.x = x[ib].qs[iqs + 0] * d;
    v.y = x[ib].qs[iqs + 1] * d;
}

template<typename dst_t>
static __global__ void dequantize_block_q8_0(const void * __restrict__ vx, dst_t * __restrict__ y, const int k)
{
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    if (i >= k) {
        return;
    }

    const int ib = i/QK8_0;
    const int iqs = (i%QK8_0)/QR8_0;
    const int iybs = i - i%QK8_0;
    dfloat2 v;
    dequantize_q8_0(vx, ib, iqs, v);
    y[iybs + iqs] = convert_from_float<dst_t>(v.x);
    y[iybs + iqs + 1] = convert_from_float<dst_t>(v.y);
}

}
}
