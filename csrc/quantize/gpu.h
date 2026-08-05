#pragma once

#include <cstdint>

#include <libgguf/libgguf.h>
#include <libgguf/gpu/types.h>

bool quantize_type_supported(enum ggml_type type);

void quantize_row(
    const float *x,
    const float *quant_weights,
    void *y,
    enum ggml_type type,
    int64_t k,
    int64_t n_per_row,
    StreamType stream);
