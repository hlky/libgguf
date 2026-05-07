#include "common/libgguf_common.h"
#include "common/libgguf_backend.h"

#include <cstring>

typedef void (*libgguf_dequant_kernel_fn)(const void *RESTRICT, float *RESTRICT, int64_t);

#define LIBGGUF_DEQUANT_TYPES(X)          \
  X(Q1_0, q1_0, block_q1_0)               \
  X(Q4_0, q4_0, block_q4_0)               \
  X(Q8_0, q8_0, block_q8_0)               \
  X(Q4_1, q4_1, block_q4_1)               \
  X(Q5_0, q5_0, block_q5_0)               \
  X(Q5_1, q5_1, block_q5_1)               \
  X(Q2_K, q2_K, block_q2_K)               \
  X(Q3_K, q3_K, block_q3_K)               \
  X(Q4_K, q4_K, block_q4_K)               \
  X(Q5_K, q5_K, block_q5_K)               \
  X(Q6_K, q6_K, block_q6_K)               \
  X(IQ2_XXS, iq2_xxs, block_iq2_xxs)      \
  X(IQ2_XS, iq2_xs, block_iq2_xs)         \
  X(IQ2_S, iq2_s, block_iq2_s)            \
  X(IQ3_XXS, iq3_xxs, block_iq3_xxs)      \
  X(IQ3_S, iq3_s, block_iq3_s)            \
  X(IQ1_S, iq1_s, block_iq1_s)            \
  X(IQ1_M, iq1_m, block_iq1_m)            \
  X(IQ4_NL, iq4_nl, block_iq4_nl)         \
  X(IQ4_XS, iq4_xs, block_iq4_xs)         \
  X(TQ1_0, tq1_0, block_tq1_0)            \
  X(TQ2_0, tq2_0, block_tq2_0)            \
  X(MXFP4, mxfp4, block_mxfp4)            \
  X(NVFP4, nvfp4, block_nvfp4)

#define DECLARE_REF_KERNEL(upper, name, block_type) \
  extern "C" void dequantize_row_##name(const block_type *RESTRICT x, float *RESTRICT y, int64_t k);

LIBGGUF_DEQUANT_TYPES(DECLARE_REF_KERNEL)

#undef DECLARE_REF_KERNEL

#if !LIBGGUF_CPU_BACKEND_REF
#define DECLARE_BACKEND_KERNEL(upper, name, block_type) \
  extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(dequantize_row_##name)(const block_type *RESTRICT x, \
                                                                    float *RESTRICT y, int64_t k);

LIBGGUF_DEQUANT_TYPES(DECLARE_BACKEND_KERNEL)

#undef DECLARE_BACKEND_KERNEL
#endif

#define WRAP_REF_KERNEL(name, block_type)                                                             \
  static void libgguf_dequant_##name##_ref_wrap(const void *RESTRICT x, float *RESTRICT y, int64_t k)  \
  {                                                                                                   \
    dequantize_row_##name((const block_type *)x, y, k);                                               \
  }

#define DEFINE_REF_WRAPPER(upper, name, block_type) \
  WRAP_REF_KERNEL(name, block_type)

LIBGGUF_DEQUANT_TYPES(DEFINE_REF_WRAPPER)

#undef WRAP_REF_KERNEL
#undef DEFINE_REF_WRAPPER

#if !LIBGGUF_CPU_BACKEND_REF
#define WRAP_BACKEND_KERNEL(name, block_type)                                                              \
  static void libgguf_dequant_##name##_compiled_wrap(const void *RESTRICT x, float *RESTRICT y, int64_t k)  \
  {                                                                                                        \
    LIBGGUF_CPU_BACKEND_SYMBOL(dequantize_row_##name)((const block_type *)x, y, k);                        \
  }

#define DEFINE_BACKEND_WRAPPER(upper, name, block_type) \
  WRAP_BACKEND_KERNEL(name, block_type)

LIBGGUF_DEQUANT_TYPES(DEFINE_BACKEND_WRAPPER)

#undef WRAP_BACKEND_KERNEL
#undef DEFINE_BACKEND_WRAPPER
#endif

static const char *libgguf_dequant_selected_backend(enum ggml_type type)
{
  switch (type)
  {
#define BACKEND_CASE(upper, name, block_type) \
  case GGML_TYPE_##upper:                     \
    return LIBGGUF_CPU_BACKEND_NAME;
    LIBGGUF_DEQUANT_TYPES(BACKEND_CASE)
#undef BACKEND_CASE
  default:
    return "unknown";
  }
}

static libgguf_dequant_kernel_fn libgguf_dequant_kernel_for_backend(enum ggml_type type, const char *backend)
{
  switch (type)
  {
#if LIBGGUF_CPU_BACKEND_REF
#define KERNEL_CASE(upper, name, block_type) \
  case GGML_TYPE_##upper:                                    \
    if (libgguf_cpu_backend_is_ref_request(backend))          \
    {                                                        \
      return libgguf_dequant_##name##_ref_wrap;              \
    }                                                        \
    return nullptr;
#else
#define KERNEL_CASE(upper, name, block_type) \
  case GGML_TYPE_##upper:                                    \
    if (libgguf_cpu_backend_is_ref_request(backend))          \
    {                                                        \
      return libgguf_dequant_##name##_ref_wrap;              \
    }                                                        \
    if (libgguf_cpu_backend_is_compiled_request(backend))     \
    {                                                        \
      return libgguf_dequant_##name##_compiled_wrap;         \
    }                                                        \
    return nullptr;
#endif
    LIBGGUF_DEQUANT_TYPES(KERNEL_CASE)
#undef KERNEL_CASE
  default:
    return nullptr;
  }
}

extern "C" const char *libgguf_dequant_backend(int type)
{
  return libgguf_dequant_selected_backend((enum ggml_type)type);
}

extern "C" int libgguf_dequant_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

extern "C" size_t libgguf_dequantize_for_backend(
    int type,
    const char *backend,
    const void *src,
    float *dst,
    int64_t nrows,
    int64_t n_per_row)
{
  if (nrows <= 0 || n_per_row <= 0)
  {
    return 0;
  }
  libgguf_dequant_kernel_fn kernel = libgguf_dequant_kernel_for_backend((enum ggml_type)type, backend);
  if (kernel == nullptr)
  {
    return 0;
  }
  kernel(src, dst, nrows * n_per_row);
  return (size_t)nrows * (size_t)n_per_row * sizeof(float);
}
