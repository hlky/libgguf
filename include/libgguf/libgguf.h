#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#include <libgguf/types.h>

#if (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define LIBGGUF_API
#elif defined(_WIN32) && !defined(__MINGW32__)
#define LIBGGUF_API __declspec(dllexport)
#else
#define LIBGGUF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif
  LIBGGUF_API size_t libgguf_row_size(enum ggml_type type, int64_t n_per_row);
  LIBGGUF_API int64_t libgguf_block_size(enum ggml_type type);
  LIBGGUF_API size_t libgguf_type_size(enum ggml_type type);
  LIBGGUF_API const char *libgguf_type_name(enum ggml_type type);
  LIBGGUF_API bool libgguf_quantize_requires_imatrix(enum ggml_type type);
  LIBGGUF_API void libgguf_quantize_free(void);
  LIBGGUF_API size_t libgguf_quantize_chunk(
      enum ggml_type type,
      const float *src,
      void *dst,
      int64_t start,
      int64_t nrows,
      int64_t n_per_row,
      const float *imatrix);
  LIBGGUF_API size_t libgguf_dequantize_chunk(
      enum ggml_type type,
      const void *src,
      float *dst,
      int64_t start,
      int64_t nrows,
      int64_t n_per_row);
#ifdef __cplusplus
}
#endif
