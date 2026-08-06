#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_k(
    const void * vx, dst_t * yy, const int i, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    dst_t * y = yy + 128*ip + il;

    const float d = __half2float(load_half(x[i].d));

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = convert_from_float<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = convert_from_float<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = convert_from_float<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = convert_from_float<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

template<typename dst_t>
static __global__ void dequantize_block_q6_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_q6_k(vx, yy + blockIdx.x*QK_K, blockIdx.x, threadIdx.x);
}

}
}
