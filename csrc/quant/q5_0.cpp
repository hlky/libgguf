#include "libgguf_common.h"
#include "common/libgguf_backend.h"

#include <cstring>

typedef void (*libgguf_q5_0_kernel_fn)(const float *RESTRICT, block_q5_0 *RESTRICT, int64_t);

struct libgguf_q5_0_selection
{
  const char *backend;
  libgguf_q5_0_kernel_fn kernel;
};

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q5_0)(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t k);
#endif

void quantize_row_q5_0(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t k)
{
  static const int qk = QK5_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++)
    {
      const float v = x[i * qk + j];
      if (amax < fabsf(v))
      {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    uint32_t qh = 0;

    for (int j = 0; j < qk / 2; ++j)
    {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
      const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

      y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10u) >> 4) << (j + qk / 2);
    }

    memcpy(&y[i].qh, &qh, sizeof(qh));
  }
}

static void quantize_row_q5_0_impl(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  if (!quant_weights)
  {
    quantize_row_q5_0(x, y, n_per_row);
    return;
  }

  float weight[QK5_0];
  int8_t L[QK5_0];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK5_0;
  for (int ib = 0; ib < nb; ++ib)
  {
    const float *xb = x + QK5_0 * ib;
    const float *qw = quant_weights + QK5_0 * ib;
    for (int j = 0; j < QK5_0; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float d = make_qx_quants(QK5_0, 16, xb, L, 1, weight);
    y[ib].d = GGML_FP32_TO_FP16(d);

    uint32_t qh = 0;

    for (int j = 0; j < 16; ++j)
    {
      const uint8_t xi0 = L[j];
      const uint8_t xi1 = L[j + 16];
      y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }

    memcpy(&y[ib].qh, &qh, sizeof(qh));
  }
}

static libgguf_q5_0_selection libgguf_q5_0_select_kernel()
{
#if LIBGGUF_CPU_BACKEND_REF
  return {LIBGGUF_CPU_BACKEND_NAME, quantize_row_q5_0};
#else
  return {LIBGGUF_CPU_BACKEND_NAME, LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q5_0)};
#endif
}

static const libgguf_q5_0_selection &libgguf_q5_0_selected()
{
  static const libgguf_q5_0_selection selected = libgguf_q5_0_select_kernel();
  return selected;
}

extern "C" const char *libgguf_q5_0_backend(void)
{
  return libgguf_q5_0_selected().backend;
}

extern "C" int libgguf_q5_0_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

static libgguf_q5_0_kernel_fn libgguf_q5_0_kernel_for_backend(const char *backend)
{
  if (libgguf_cpu_backend_is_ref_request(backend))
    return quantize_row_q5_0;
#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
    return LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q5_0);
#endif
  return nullptr;
}

extern "C" size_t libgguf_quantize_q5_0_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q5_0_kernel_fn kernel = libgguf_q5_0_kernel_for_backend(backend);
  if (!kernel)
    return 0;
  kernel(src, (block_q5_0 *)dst, nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q5_0, n_per_row);
}

size_t quantize_q5_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    libgguf_q5_0_selected().kernel(src, (block_q5_0 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q5_0, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q5_0, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_q5_0_impl(src, (block_q5_0 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}
