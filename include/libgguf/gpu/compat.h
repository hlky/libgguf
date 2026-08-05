#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#endif
