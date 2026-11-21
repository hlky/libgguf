#pragma once

#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/types.h>

extern "C" hipError_t libgguf_gfx942_bf16_cast(const void *x, void *y, int count,
    libgguf_gpu_dtype dtype, int threads, int items, hipStream_t stream);
