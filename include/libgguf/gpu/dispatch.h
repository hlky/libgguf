#pragma once

#include <libgguf/gpu/types.h>

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  do {                                                \
    switch (TYPE) {                                   \
      case libgguf_gpu_dtype::f32: {                  \
        using scalar_t = float;                       \
        __VA_ARGS__();                                \
        break;                                        \
      }                                               \
      case libgguf_gpu_dtype::f16: {                  \
        using scalar_t = half;                        \
        __VA_ARGS__();                                \
        break;                                        \
      }                                               \
      case libgguf_gpu_dtype::bf16: {                 \
        using scalar_t = libgguf_gpu_bfloat16;        \
        __VA_ARGS__();                                \
        break;                                        \
      }                                               \
    }                                                 \
  } while (false)
