#pragma once

#include <cstdint>

#include <libgguf/gpu/compat.h>

enum class libgguf_gpu_dtype : uint8_t
{
  f32,
  f16,
  bf16,
};

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
using StreamType = hipStream_t;
using libgguf_gpu_bfloat16 = __hip_bfloat16;
#else
using StreamType = cudaStream_t;
using libgguf_gpu_bfloat16 = __nv_bfloat16;
#endif
