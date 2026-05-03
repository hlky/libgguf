#include "common/libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstring>

typedef void (*libgguf_dequant_kernel_fn)(const void *RESTRICT, float *RESTRICT, int64_t);

struct libgguf_dequant_selection
{
  const char *backend;
  libgguf_dequant_kernel_fn kernel;
};

#define LIBGGUF_DEQUANT_TYPES(X) \
  X(Q1_0, q1_0, block_q1_0, sse4_1)      \
  X(Q4_1, q4_1, block_q4_1, sse2)        \
  X(Q5_0, q5_0, block_q5_0, sse4_1)      \
  X(Q5_1, q5_1, block_q5_1, ref)         \
  X(Q2_K, q2_K, block_q2_K, sse2)        \
  X(Q3_K, q3_K, block_q3_K, sse2)        \
  X(Q4_K, q4_K, block_q4_K, ref)         \
  X(Q5_K, q5_K, block_q5_K, ref)         \
  X(Q6_K, q6_K, block_q6_K, sse2)        \
  X(IQ2_XXS, iq2_xxs, block_iq2_xxs, avx2) \
  X(IQ2_XS, iq2_xs, block_iq2_xs, sse4_1)  \
  X(IQ2_S, iq2_s, block_iq2_s, sse2)       \
  X(IQ3_XXS, iq3_xxs, block_iq3_xxs, avx2) \
  X(IQ3_S, iq3_s, block_iq3_s, avx2)       \
  X(IQ1_S, iq1_s, block_iq1_s, ref)        \
  X(IQ1_M, iq1_m, block_iq1_m, ref)        \
  X(IQ4_NL, iq4_nl, block_iq4_nl, ref)     \
  X(IQ4_XS, iq4_xs, block_iq4_xs, ref)     \
  X(TQ1_0, tq1_0, block_tq1_0, ref)        \
  X(TQ2_0, tq2_0, block_tq2_0, ref)        \
  X(MXFP4, mxfp4, block_mxfp4, ref)        \
  X(NVFP4, nvfp4, block_nvfp4, ref)

extern "C" void dequantize_row_q4_0_ref(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_sse2(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_sse4_1(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_avx2(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" const char *libgguf_dequant_q4_0_backend(void);

extern "C" void dequantize_row_q8_0_ref(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_sse2(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_sse4_1(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_avx2(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
extern "C" const char *libgguf_dequant_q8_0_backend(void);

#define DECLARE_KERNELS(upper, name, block_type, default_backend)                                    \
  extern "C" void dequantize_row_##name##_ref(const block_type *RESTRICT x, float *RESTRICT y, int64_t k);    \
  extern "C" void dequantize_row_##name##_sse2(const block_type *RESTRICT x, float *RESTRICT y, int64_t k);   \
  extern "C" void dequantize_row_##name##_sse4_1(const block_type *RESTRICT x, float *RESTRICT y, int64_t k); \
  extern "C" void dequantize_row_##name##_avx2(const block_type *RESTRICT x, float *RESTRICT y, int64_t k);

LIBGGUF_DEQUANT_TYPES(DECLARE_KERNELS)

#undef DECLARE_KERNELS

static bool libgguf_dequant_backend_supported(const char *backend)
{
  if (backend == nullptr)
  {
    return false;
  }
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (std::strcmp(backend, "ref") == 0)
  {
    return true;
  }
  if (std::strcmp(backend, "sse2") == 0)
  {
    return features.sse2;
  }
  if (std::strcmp(backend, "sse4_1") == 0)
  {
    return features.sse4_1;
  }
  if (std::strcmp(backend, "avx2") == 0)
  {
    return features.avx2;
  }
  return false;
}

#define WRAP_KERNEL(name, backend, block_type)                                                        \
  static void libgguf_dequant_##name##_##backend##_wrap(const void *RESTRICT x, float *RESTRICT y, int64_t k) \
  {                                                                                                   \
    dequantize_row_##name##_##backend((const block_type *)x, y, k);                                   \
  }

#define DEFINE_WRAPPERS(upper, name, block_type, default_backend) \
  WRAP_KERNEL(name, ref, block_type)             \
  WRAP_KERNEL(name, sse2, block_type)            \
  WRAP_KERNEL(name, sse4_1, block_type)          \
  WRAP_KERNEL(name, avx2, block_type)

LIBGGUF_DEQUANT_TYPES(DEFINE_WRAPPERS)

#undef DEFINE_WRAPPERS

WRAP_KERNEL(q4_0, ref, block_q4_0)
WRAP_KERNEL(q4_0, sse2, block_q4_0)
WRAP_KERNEL(q4_0, sse4_1, block_q4_0)
WRAP_KERNEL(q4_0, avx2, block_q4_0)
WRAP_KERNEL(q8_0, ref, block_q8_0)
WRAP_KERNEL(q8_0, sse2, block_q8_0)
WRAP_KERNEL(q8_0, sse4_1, block_q8_0)
WRAP_KERNEL(q8_0, avx2, block_q8_0)

#undef WRAP_KERNEL

#define SELECT_DEFAULT(name, backend_name)                         \
  if (std::strcmp(preferred, #backend_name) == 0 &&                \
      libgguf_dequant_backend_supported(#backend_name))            \
  {                                                               \
    return {#backend_name, libgguf_dequant_##name##_##backend_name##_wrap}; \
  }

#define DEFINE_SELECTION(upper, name, block_type, default_backend)                      \
  static libgguf_dequant_selection libgguf_dequant_##name##_select_kernel()             \
  {                                                                                     \
    const libgguf_cpu_features &features = libgguf_get_cpu_features();                  \
    const char *preferred = #default_backend;                                           \
    SELECT_DEFAULT(name, ref)                                                           \
    SELECT_DEFAULT(name, sse2)                                                          \
    SELECT_DEFAULT(name, sse4_1)                                                        \
    SELECT_DEFAULT(name, avx2)                                                          \
    if (features.avx2)                                                                  \
    {                                                                                   \
      return {"avx2", libgguf_dequant_##name##_avx2_wrap};                             \
    }                                                                                   \
    if (features.sse4_1)                                                                \
    {                                                                                   \
      return {"sse4_1", libgguf_dequant_##name##_sse4_1_wrap};                         \
    }                                                                                   \
    if (features.sse2)                                                                  \
    {                                                                                   \
      return {"sse2", libgguf_dequant_##name##_sse2_wrap};                             \
    }                                                                                   \
    return {"ref", libgguf_dequant_##name##_ref_wrap};                                 \
  }                                                                                     \
                                                                                        \
  static const libgguf_dequant_selection &libgguf_dequant_##name##_selected()           \
  {                                                                                     \
    static const libgguf_dequant_selection selected = libgguf_dequant_##name##_select_kernel(); \
    return selected;                                                                    \
  }                                                                                     \
                                                                                        \
  extern "C" void dequantize_row_##name(const block_type *RESTRICT x, float *RESTRICT y, int64_t k) \
  {                                                                                     \
    libgguf_dequant_##name##_selected().kernel(x, y, k);                                \
  }

LIBGGUF_DEQUANT_TYPES(DEFINE_SELECTION)

#undef DEFINE_SELECTION
#undef SELECT_DEFAULT

static const char *libgguf_dequant_selected_backend(enum ggml_type type)
{
  switch (type)
  {
  case GGML_TYPE_Q4_0:
    return libgguf_dequant_q4_0_backend();
  case GGML_TYPE_Q8_0:
    return libgguf_dequant_q8_0_backend();
#define BACKEND_CASE(upper, name, block_type, default_backend) \
  case GGML_TYPE_##upper:                     \
    return libgguf_dequant_##name##_selected().backend;
    LIBGGUF_DEQUANT_TYPES(BACKEND_CASE)
#undef BACKEND_CASE
  default:
    return "unknown";
  }
}

static libgguf_dequant_kernel_fn libgguf_dequant_kernel_for_backend(enum ggml_type type, const char *backend)
{
  if (!libgguf_dequant_backend_supported(backend))
  {
    return nullptr;
  }

#define BACKEND_TO_KERNEL(name, backend_name)     \
  if (std::strcmp(backend, #backend_name) == 0)   \
  {                                               \
    return libgguf_dequant_##name##_##backend_name##_wrap; \
  }

#define KERNEL_CASE(upper, name, block_type, default_backend) \
  case GGML_TYPE_##upper:                    \
    BACKEND_TO_KERNEL(name, ref)             \
    BACKEND_TO_KERNEL(name, sse2)            \
    BACKEND_TO_KERNEL(name, sse4_1)          \
    BACKEND_TO_KERNEL(name, avx2)            \
    return nullptr;

  switch (type)
  {
  case GGML_TYPE_Q4_0:
    BACKEND_TO_KERNEL(q4_0, ref)
    BACKEND_TO_KERNEL(q4_0, sse2)
    BACKEND_TO_KERNEL(q4_0, sse4_1)
    BACKEND_TO_KERNEL(q4_0, avx2)
    return nullptr;
  case GGML_TYPE_Q8_0:
    BACKEND_TO_KERNEL(q8_0, ref)
    BACKEND_TO_KERNEL(q8_0, sse2)
    BACKEND_TO_KERNEL(q8_0, sse4_1)
    BACKEND_TO_KERNEL(q8_0, avx2)
    return nullptr;
    LIBGGUF_DEQUANT_TYPES(KERNEL_CASE)
  default:
    return nullptr;
  }

#undef KERNEL_CASE
#undef BACKEND_TO_KERNEL
}

extern "C" const char *libgguf_dequant_backend(int type)
{
  return libgguf_dequant_selected_backend((enum ggml_type)type);
}

extern "C" int libgguf_dequant_cpu_supports_backend(const char *backend)
{
  return libgguf_dequant_backend_supported(backend) ? 1 : 0;
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
