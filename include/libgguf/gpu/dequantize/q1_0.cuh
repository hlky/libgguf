#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ void dequantize_q1_0(
    const void * vx, const int i, dfloat2 & v) {
    const block_q1_0 * x = (const block_q1_0 *) vx;
    const int out = 2 * i;
    const int ib = out / QK1_0;
    const int j = out - ib * QK1_0;
    const int byte = j >> 3;
    const int bit = j & 7;
    const float d = __half2float(load_half(x[ib].d));
    const uint8_t packed = x[ib].qs[byte];
    const int q0 = (packed >> bit) & 1;
    const int q1 = (packed >> (bit + 1)) & 1;
    v.x = q0 ? d : -d;
    v.y = q1 ? d : -d;
}

template<typename dst_t>
static __global__ void dequantize_block_q1_0_pair(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k / 2) {
        return;
    }
    dfloat2 v;
    dequantize_q1_0(vx, i, v);
    y[2*i] = convert_from_float<dst_t>(v.x);
    y[2*i + 1] = convert_from_float<dst_t>(v.y);
}

}
}
