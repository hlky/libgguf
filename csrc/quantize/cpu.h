#pragma once

#include <cstddef>
#include <cstdint>

#include <libgguf/types.h>

size_t libgguf_quantize_chunk_serial(
    enum ggml_type type,
    const float *src,
    void *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row,
    const float *imatrix);
