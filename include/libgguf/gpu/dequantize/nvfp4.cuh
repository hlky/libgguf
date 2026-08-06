#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ void dequantize_nvfp4(
    const void * vx, const int packed_index, dfloat2 & v) {
    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const int ib = packed_index / (QK_NVFP4 / 2);
    const int byte = packed_index - ib * (QK_NVFP4 / 2);
    const int group = byte / 8;
    const int local = byte - group * 8;
    const uint8_t packed = x[ib].qs[byte];
    const float d = ue4m3_to_float(x[ib].d[group]);
    v.x = d * kvalues_e2m1[packed & 15];
    v.y = d * kvalues_e2m1[packed >> 4];
}

template<typename dst_t>
static __global__ void dequantize_block_nvfp4(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int packed_index = blockDim.x*blockIdx.x + threadIdx.x;
    const int n_blocks = k / QK_NVFP4;
    if (packed_index >= n_blocks * (QK_NVFP4 / 2)) {
        return;
    }
    dfloat2 v;
    dequantize_nvfp4(vx, packed_index, v);
    const int ib = packed_index / (QK_NVFP4 / 2);
    const int byte = packed_index - ib * (QK_NVFP4 / 2);
    const int group = byte / 8;
    const int local = byte - group * 8;
    dst_t * yb = y + ib * QK_NVFP4 + group * 16 + local;
    yb[0] = convert_from_float<dst_t>(v.x);
    yb[8] = convert_from_float<dst_t>(v.y);
}

}
}
