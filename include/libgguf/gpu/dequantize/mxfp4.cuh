#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ void dequantize_mxfp4(
    const void * vx, const int packed_index, dfloat2 & v) {
    const block_mxfp4 * x = (const block_mxfp4 *) vx;
    const int ib = packed_index / (QK_MXFP4 / 2);
    const int byte = packed_index - ib * (QK_MXFP4 / 2);
    const uint8_t packed = x[ib].qs[byte];
    const float d = e8m0_to_float(x[ib].e);
    v.x = d * kvalues_e2m1[packed & 15];
    v.y = d * kvalues_e2m1[packed >> 4];
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int packed_index = blockDim.x*blockIdx.x + threadIdx.x;
    const int n_blocks = k / QK_MXFP4;
    if (packed_index >= n_blocks * (QK_MXFP4 / 2)) {
        return;
    }
    dfloat2 v;
    dequantize_mxfp4(vx, packed_index, v);
    const int ib = packed_index / (QK_MXFP4 / 2);
    const int byte = packed_index - ib * (QK_MXFP4 / 2);
    dst_t * yb = y + ib * QK_MXFP4;
    yb[byte] = convert_from_float<dst_t>(v.x);
    yb[byte + 16] = convert_from_float<dst_t>(v.y);
}

}
}
