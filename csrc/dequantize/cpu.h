#pragma once

#include <cstddef>
#include <cstdint>

#include <libgguf/types.h>

size_t libgguf_dequantize_chunk_serial(
    enum ggml_type type,
    const void *src,
    float *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row);
