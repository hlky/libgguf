#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <c10/util/BFloat16.h>

#include "libgguf_internal.h"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

static __device__ __forceinline__ half gguf_cuda_load_half(ggml_half value) {
    return __ushort_as_half(value);
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
