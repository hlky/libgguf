#include "libgguf_common.h"
#include "common/libgguf_backend.h"

typedef void (*libgguf_q8_0_kernel_fn)(const float *RESTRICT, block_q8_0 *RESTRICT, int64_t);

struct libgguf_q8_0_selection
{
  const char *backend;
  libgguf_q8_0_kernel_fn kernel;
};

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q8_0)(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k);
#endif

void quantize_row_q8_0(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++)
    {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j)
    {
      const float x0 = x[i * QK8_0 + j] * id;
      y[i].qs[j] = roundf(x0);
    }
  }
}

static libgguf_q8_0_selection libgguf_q8_0_select_kernel()
{
#if LIBGGUF_CPU_BACKEND_REF
  return {LIBGGUF_CPU_BACKEND_NAME, quantize_row_q8_0};
#else
  return {LIBGGUF_CPU_BACKEND_NAME, LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q8_0)};
#endif
}

static const libgguf_q8_0_selection &libgguf_q8_0_selected()
{
  static const libgguf_q8_0_selection selected = libgguf_q8_0_select_kernel();
  return selected;
}

static libgguf_q8_0_kernel_fn libgguf_q8_0_kernel()
{
  return libgguf_q8_0_selected().kernel;
}

extern "C" const char *libgguf_q8_0_backend(void)
{
  return libgguf_q8_0_selected().backend;
}

extern "C" int libgguf_q8_0_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

static libgguf_q8_0_kernel_fn libgguf_q8_0_kernel_for_backend(const char *backend)
{
  if (libgguf_cpu_backend_is_ref_request(backend))
  {
    return quantize_row_q8_0;
  }
#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
  {
    return LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q8_0);
  }
#endif
  return nullptr;
}

extern "C" size_t libgguf_quantize_q8_0_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q8_0_kernel_fn kernel = libgguf_q8_0_kernel_for_backend(backend);
  if (!kernel)
    return 0;

  kernel(src, (block_q8_0 *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
}

size_t quantize_q8_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  libgguf_q8_0_kernel()(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
}
