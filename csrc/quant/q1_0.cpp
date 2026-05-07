#include "libgguf_common.h"
#include "common/libgguf_backend.h"

typedef void (*libgguf_q1_0_kernel_fn)(const float *RESTRICT, block_q1_0 *RESTRICT, int64_t);

struct libgguf_q1_0_selection
{
  const char *backend;
  libgguf_q1_0_kernel_fn kernel;
};

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q1_0)(const float *RESTRICT x, block_q1_0 *RESTRICT y, int64_t k);
#endif

void quantize_row_q1_0(const float *RESTRICT x, block_q1_0 *RESTRICT y, int64_t k)
{
  static const int qk = QK1_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float sum_abs = 0.0f;
    for (int j = 0; j < qk; j++)
    {
      sum_abs += fabsf(x[i * qk + j]);
    }
    const float d = sum_abs / qk;

    y[i].d = GGML_FP32_TO_FP16(d);

    // Clear all bits first
    for (int j = 0; j < qk / 8; ++j)
    {
      y[i].qs[j] = 0;
    }

    // Just store sign of each weight directly (no normalization)
    for (int j = 0; j < qk; ++j)
    {
      const int bit_index = j;
      const int byte_index = bit_index / 8;
      const int bit_offset = bit_index % 8;

      if (x[i * qk + j] >= 0.0f)
      {
        y[i].qs[byte_index] |= (1 << bit_offset);
      }
    }
  }
}

// reference implementation for deterministic creation of model files

static libgguf_q1_0_selection libgguf_q1_0_select_kernel()
{
#if LIBGGUF_CPU_BACKEND_REF
  return {LIBGGUF_CPU_BACKEND_NAME, quantize_row_q1_0};
#else
  return {LIBGGUF_CPU_BACKEND_NAME, LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q1_0)};
#endif
}

static const libgguf_q1_0_selection &libgguf_q1_0_selected()
{
  static const libgguf_q1_0_selection selected = libgguf_q1_0_select_kernel();
  return selected;
}

extern "C" const char *libgguf_q1_0_backend(void)
{
  return libgguf_q1_0_selected().backend;
}

extern "C" int libgguf_q1_0_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

static libgguf_q1_0_kernel_fn libgguf_q1_0_kernel_for_backend(const char *backend)
{
  if (libgguf_cpu_backend_is_ref_request(backend))
    return quantize_row_q1_0;
#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
    return LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q1_0);
#endif
  return nullptr;
}

extern "C" size_t libgguf_quantize_q1_0_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q1_0_kernel_fn kernel = libgguf_q1_0_kernel_for_backend(backend);
  if (!kernel)
    return 0;
  kernel(src, (block_q1_0 *)dst, nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q1_0, n_per_row);
}

size_t quantize_q1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  GGML_UNUSED(quant_weights);
  libgguf_q1_0_selected().kernel(src, (block_q1_0 *)dst, nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_Q1_0, n_per_row);
}
