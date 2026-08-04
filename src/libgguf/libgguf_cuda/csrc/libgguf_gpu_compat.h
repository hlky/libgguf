#pragma once

// Keep the existing libgguf CUDA kernels source-compatible with HIP. This
// header only maps runtime types, includes, and warp-shuffle spellings; kernel
// algorithms and packed formats remain owned by their existing source files.
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using cudaStream_t = hipStream_t;
using gguf_gpu_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float
gguf_gpu_bfloat16_to_float(gguf_gpu_bfloat16 value) {
  return static_cast<float>(value);
}

#define GGUF_GPU_SHFL_SYNC(mask, value, source_lane)                            \
  __shfl((value), (source_lane), warpSize)
#define GGUF_GPU_SHFL_DOWN_SYNC(mask, value, lane_delta)                       \
  __shfl_down((value), (lane_delta), warpSize)
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using gguf_gpu_bfloat16 = __nv_bfloat16;

static __device__ __forceinline__ float
gguf_gpu_bfloat16_to_float(gguf_gpu_bfloat16 value) {
  return __bfloat162float(value);
}

#define GGUF_GPU_SHFL_SYNC(mask, value, source_lane)                           \
  __shfl_sync((mask), (value), (source_lane))
#define GGUF_GPU_SHFL_DOWN_SYNC(mask, value, lane_delta)                       \
  __shfl_down_sync((mask), (value), (lane_delta))
#endif
