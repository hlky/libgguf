#pragma once

#include <libgguf/gpu/dequantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename dst_t>
static __global__ void dequantize_block_bf16(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const uint16_t value = ((const uint16_t *) vx)[i];
    y[i] = convert_from_float<dst_t>(bitcast_u32_to_float((uint32_t)value << 16));
}

}
}
