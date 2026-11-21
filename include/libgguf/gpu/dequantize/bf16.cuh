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

template<typename dst_t, int Items = 1>
static __global__ void dequantize_block_bf16(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int64_t first = int64_t(blockDim.x) * blockIdx.x * Items + threadIdx.x;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int64_t i = first + item * blockDim.x;
        if (i < k)
            y[i] = convert_from_float<dst_t>(dequantize_bf16(vx, int(i)));
    }
}

}
}
