#pragma once

#include "common/libgguf_common.h"

#define LIBGGUF_DEQUANT_DEFINE_BACKEND(name, block_type, backend)                                    \
  extern "C" void dequantize_row_##name##_ref(const block_type *RESTRICT x, float *RESTRICT y, int64_t k); \
  extern "C" void dequantize_row_##name##_##backend(const block_type *RESTRICT x, float *RESTRICT y, int64_t k) \
  {                                                                                                 \
    dequantize_row_##name##_ref(x, y, k);                                                           \
  }

