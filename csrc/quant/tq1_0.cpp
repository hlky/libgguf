#include "libgguf_common.h"
#include "common/libgguf_backend.h"

typedef void (*libgguf_tq1_0_kernel_fn)(const float *RESTRICT, block_tq1_0 *RESTRICT, int64_t);

struct libgguf_tq1_0_selection
{
  const char *backend;
  libgguf_tq1_0_kernel_fn kernel;
};

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_tq1_0)(const float *RESTRICT x, block_tq1_0 *RESTRICT y, int64_t k);
#endif

void quantize_row_tq1_0(const float *RESTRICT x, block_tq1_0 *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int64_t i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK_K; j++)
    {
      const float v = x[j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    // 5 elements per byte, along 32 bytes
    for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32)
    {
      for (size_t m = 0; m < 32; ++m)
      {
        uint8_t q = 0;
        for (size_t n = 0; n < 5; ++n)
        {
          int xi = lroundf(x[m + n * 32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
          q *= 3;
          q += xi;
        }
        // ceiling division (243 == pow(3, 5))
        q = ((uint16_t)q * 256 + (243 - 1)) / 243;
        y[i].qs[j + m] = q;
      }
      x += 5 * 32;
    }
    // along 16 bytes
    for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16)
    {
      for (size_t m = 0; m < 16; ++m)
      {
        uint8_t q = 0;
        for (size_t n = 0; n < 5; ++n)
        {
          int xi = lroundf(x[m + n * 16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
          q *= 3;
          q += xi;
        }
        // ceiling division (243 == pow(3, 5))
        q = ((uint16_t)q * 256 + (243 - 1)) / 243;
        y[i].qs[j + m] = q;
      }
      x += 5 * 16;
    }
    // 4 elements per byte
    for (size_t j = 0; j < sizeof(y->qh); ++j)
    {
      uint8_t q = 0;
      for (size_t m = 0; m < 4; ++m)
      {
        // -1, 0, 1 -> 0, 1, 2
        int xi = lroundf(x[j + m * sizeof(y->qh)] * id) + 1;
        q *= 3;
        q += xi;
      }
      // shift the first value to the most significant trit
      q *= 3;
      // ceiling division (243 == pow(3, 5))
      q = ((uint16_t)q * 256 + (243 - 1)) / 243;
      y[i].qh[j] = q;
    }
    x += 4 * sizeof(y->qh);
  }
}

static libgguf_tq1_0_selection libgguf_tq1_0_select_kernel()
{
#if LIBGGUF_CPU_BACKEND_REF
  return {LIBGGUF_CPU_BACKEND_NAME, quantize_row_tq1_0};
#else
  return {LIBGGUF_CPU_BACKEND_NAME, LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_tq1_0)};
#endif
}

static const libgguf_tq1_0_selection &libgguf_tq1_0_selected()
{
  static const libgguf_tq1_0_selection selected = libgguf_tq1_0_select_kernel();
  return selected;
}

static libgguf_tq1_0_kernel_fn libgguf_tq1_0_kernel()
{
  return libgguf_tq1_0_selected().kernel;
}

extern "C" const char *libgguf_tq1_0_backend(void)
{
  return libgguf_tq1_0_selected().backend;
}

extern "C" int libgguf_tq1_0_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

static libgguf_tq1_0_kernel_fn libgguf_tq1_0_kernel_for_backend(const char *backend)
{
  if (libgguf_cpu_backend_is_ref_request(backend))
  {
    return quantize_row_tq1_0;
  }
#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
  {
    return LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_tq1_0);
  }
#endif
  return nullptr;
}

extern "C" size_t libgguf_quantize_tq1_0_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_tq1_0_kernel_fn kernel = libgguf_tq1_0_kernel_for_backend(backend);
  if (!kernel)
  {
    return 0;
  }
  kernel(src, (block_tq1_0 *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_TQ1_0, n_per_row);
}

size_t quantize_tq1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  const size_t row_size = libgguf_row_size(GGML_TYPE_TQ1_0, n_per_row);
  libgguf_tq1_0_kernel()(src, (block_tq1_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}
