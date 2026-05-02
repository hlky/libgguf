#pragma once
#include <cstddef>
#include <cstdint>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#if defined(_WIN32) && !defined(__MINGW32__)
#define LIBGGUF_API __declspec(dllexport)
#else
#define LIBGGUF_API __attribute__((visibility("default")))
#endif

// NOTE: always add types at the end of the enum to keep backward compatibility
enum ggml_type
{
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 = 5, support has been removed
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S = 19,
  GGML_TYPE_IQ4_NL = 20,
  GGML_TYPE_IQ3_S = 21,
  GGML_TYPE_IQ2_S = 22,
  GGML_TYPE_IQ4_XS = 23,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_IQ1_M = 29,
  GGML_TYPE_BF16 = 30,
  // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
  // GGML_TYPE_Q4_0_4_8 = 32,
  // GGML_TYPE_Q4_0_8_8 = 33,
  GGML_TYPE_TQ1_0 = 34,
  GGML_TYPE_TQ2_0 = 35,
  // GGML_TYPE_IQ4_NL_4_4 = 36,
  // GGML_TYPE_IQ4_NL_4_8 = 37,
  // GGML_TYPE_IQ4_NL_8_8 = 38,
  GGML_TYPE_MXFP4 = 39, // MXFP4 (1 block)
  GGML_TYPE_NVFP4 = 40, // NVFP4 (4 blocks, E4M3 scale)
  GGML_TYPE_Q1_0 = 41,
  GGML_TYPE_COUNT = 42,
};

// precision
enum ggml_prec
{
  GGML_PREC_DEFAULT = 0, // stored as ggml_tensor.op_params, 0 by default
  GGML_PREC_F32 = 10,
};

// model file types
enum ggml_ftype
{
  GGML_FTYPE_UNKNOWN = -1,
  GGML_FTYPE_ALL_F32 = 0,
  GGML_FTYPE_MOSTLY_F16 = 1,           // except 1d tensors
  GGML_FTYPE_MOSTLY_Q4_0 = 2,          // except 1d tensors
  GGML_FTYPE_MOSTLY_Q4_1 = 3,          // except 1d tensors
  GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
  GGML_FTYPE_MOSTLY_Q8_0 = 7,          // except 1d tensors
  GGML_FTYPE_MOSTLY_Q5_0 = 8,          // except 1d tensors
  GGML_FTYPE_MOSTLY_Q5_1 = 9,          // except 1d tensors
  GGML_FTYPE_MOSTLY_Q2_K = 10,         // except 1d tensors
  GGML_FTYPE_MOSTLY_Q3_K = 11,         // except 1d tensors
  GGML_FTYPE_MOSTLY_Q4_K = 12,         // except 1d tensors
  GGML_FTYPE_MOSTLY_Q5_K = 13,         // except 1d tensors
  GGML_FTYPE_MOSTLY_Q6_K = 14,         // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ2_XXS = 15,      // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ2_XS = 16,       // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ3_XXS = 17,      // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ1_S = 18,        // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ4_NL = 19,       // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ3_S = 20,        // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ2_S = 21,        // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ4_XS = 22,       // except 1d tensors
  GGML_FTYPE_MOSTLY_IQ1_M = 23,        // except 1d tensors
  GGML_FTYPE_MOSTLY_BF16 = 24,         // except 1d tensors
  GGML_FTYPE_MOSTLY_MXFP4 = 25,        // except 1d tensors
  GGML_FTYPE_MOSTLY_NVFP4 = 26,        // except 1d tensors
  GGML_FTYPE_MOSTLY_Q1_0 = 27,         // except 1d tensors
};

extern "C"
{
  LIBGGUF_API size_t libgguf_row_size(enum ggml_type type, int64_t n_per_row);
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
}
