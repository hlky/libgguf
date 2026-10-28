#pragma once

#include <libgguf/gpu/quantize/common.cuh>
#include <libgguf/gpu/quantize/fragment.cuh>

namespace libgguf
{
namespace gpu
{

namespace q8_0_detail
{

struct NativeOperations
{
    static __device__ __forceinline__ float reduce_absmax(float value)
    {
        const auto mask = __activemask();
        return __shfl_sync(mask, warp_reduce_max(fabsf(value)), 0, libgguf::gpu::subgroup32::width);
    }

    static __device__ __forceinline__ float divide(float numerator, float denominator)
    {
        return __fdiv_rn(numerator, denominator);
    }

    static __device__ __forceinline__ float multiply(float lhs, float rhs)
    {
        return __fmul_rn(lhs, rhs);
    }

    static __device__ __forceinline__ int8_t round_to_int8(float value)
    {
        return static_cast<int8_t>(roundf(value));
    }
};

}

template<typename src_t>
static __device__ __forceinline__ float quantize_q8_0(src_t value, int8_t & q)
{
    return BlockFragmentQuantizer<GGML_TYPE_Q8_0>::quantize_lane(
        value, q, q8_0_detail::NativeOperations{});
}

template<typename src_t>
static __global__ void quantize_block_q8_0(
    const src_t * __restrict__ x,
    block_q8_0 * __restrict__ y,
    int64_t n_blocks)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
    if (ib >= n_blocks)
    {
        return;
    }

    int8_t q;
    const float d = quantize_q8_0(x[ib * QK8_0 + lane], q);
    if (lane == 0)
    {
        y[ib].d = compute_fp32_to_fp16(d);
    }
    y[ib].qs[lane] = q;
}

}
}
