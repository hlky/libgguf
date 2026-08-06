#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

static __device__ __forceinline__ float dequantize_bf16(const void * vx, const int i) {
    const uint16_t value = ((const uint16_t *) vx)[i];
    return bitcast_u32_to_float((uint32_t)value << 16);
}

template<typename dst_t>
static __global__ void dequantize_block_bf16(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    y[i] = convert_from_float<dst_t>(dequantize_bf16(vx, i));
}

}
}
