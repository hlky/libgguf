#pragma once

#include "cuda_quantize_common.cuh"

template <typename input_t>
static __device__ __forceinline__ float q8_0_input_to_float(input_t value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float q8_0_input_to_float(gguf_gpu_bfloat16 value) {
    return gguf_gpu_bfloat16_to_float(value);
}

// Shared lane-level implementation used by libgguf's ordinary Q8_0 launcher
// and kernels that consume the same packed representation directly.
template <typename input_t>
static __device__ __forceinline__ float gguf_cuda_quantize_q8_0_lane(
    input_t value, int8_t * quantized
) {
    constexpr unsigned mask = 0xffffffffu;
    const float v = q8_0_input_to_float(value);
    const float amax = GGUF_GPU_SHFL_SYNC(mask, gguf_cuda_warp_reduce_max(fabsf(v)), 0);
    const float d = __fdiv_rn(amax, 127.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    *quantized = (int8_t)roundf(__fmul_rn(v, id));
    return d;
}
