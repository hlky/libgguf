#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <c10/util/BFloat16.h>

#include "libgguf_internal.h"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

static __device__ __forceinline__ half gguf_cuda_load_half(ggml_half value) {
    return __ushort_as_half(value);
}

static __device__ __forceinline__ uint32_t gguf_cuda_fp32_to_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static __device__ __forceinline__ float gguf_cuda_bits_to_fp32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static __device__ __forceinline__ int gguf_cuda_nearest_int(float value) {
    const float rounded = value + 12582912.0f;
    int bits;
    memcpy(&bits, &rounded, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

static __device__ __forceinline__ ggml_half gguf_cuda_compute_fp32_to_fp16(float value) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(value) * scale_to_inf) * scale_to_zero;

    const uint32_t w = gguf_cuda_fp32_to_bits(value);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    base = gguf_cuda_bits_to_fp32((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = gguf_cuda_fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (ggml_half)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

static __device__ __forceinline__ half gguf_cuda_low_half(ggml_half2 value) {
    return __ushort_as_half(value & 0xffffu);
}

static __device__ __forceinline__ half gguf_cuda_high_half(ggml_half2 value) {
    return __ushort_as_half(value >> 16);
}

#include "libgguf_cuda_tables.cuh"

typedef float dfloat;
typedef float2 dfloat2;
typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);

template<typename dst_t>
using to_cuda_ggml_t = void (*)(const void * __restrict__ x, dst_t * __restrict__ y, int k, cudaStream_t stream);

template<typename dst_t>
static __device__ __forceinline__ dst_t convert_from_half(half val) {
    return val;
}

template<>
__device__ __forceinline__ c10::BFloat16 convert_from_half<c10::BFloat16>(half val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(__half2float(val));
#else
    return __half2float(val);
#endif
}

template<>
__device__ __forceinline__ float convert_from_half<float>(half val) {
    return __half2float(val);
}

template<typename dst_t>
static __device__ __forceinline__ dst_t convert_from_float(float val) {
    return val;
}

template<>
__device__ __forceinline__ half convert_from_float<half>(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ c10::BFloat16 convert_from_float<c10::BFloat16>(float val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(val);
#else
    return val;
#endif
}
