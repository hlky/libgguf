#pragma once

#include <cstdint>

#include <libgguf/libgguf.h>
#include <libgguf/gpu/types.h>

bool dequantize_type_supported(enum ggml_type type);

void dequantize_row(
    const void *x,
    void *y,
    enum ggml_type type,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream);
