#include "libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstring>

typedef void (*libgguf_q4_0_kernel_fn)(const float *RESTRICT, block_q4_0 *RESTRICT, int64_t);

struct libgguf_q4_0_selection
{
  const char *backend;
  libgguf_q4_0_kernel_fn kernel;
};

extern "C" void quantize_row_q4_0_sse2(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_sse4_1(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_avx2(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k);

void quantize_row_q4_0_ref(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k)
{
  static const int qk = QK4_0;

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

    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < qk / 2; ++j)
    {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static libgguf_q4_0_selection libgguf_q4_0_select_kernel()
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();

  if (features.sse4_1)
  {
    return {"sse4_1", quantize_row_q4_0_sse4_1};
  }
  if (features.avx2)
  {
    return {"avx2", quantize_row_q4_0_avx2};
  }
  if (features.sse2)
  {
    return {"sse2", quantize_row_q4_0_sse2};
  }
  return {"ref", quantize_row_q4_0_ref};
}

static const libgguf_q4_0_selection &libgguf_q4_0_selected()
{
  static const libgguf_q4_0_selection selected = libgguf_q4_0_select_kernel();
  return selected;
}

static libgguf_q4_0_kernel_fn libgguf_q4_0_kernel()
{
  return libgguf_q4_0_selected().kernel;
}

extern "C" const char *libgguf_q4_0_backend(void)
{
  return libgguf_q4_0_selected().backend;
}

extern "C" int libgguf_q4_0_cpu_supports_backend(const char *backend)
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (backend == nullptr)
  {
    return 0;
  }
  if (std::strcmp(backend, "ref") == 0)
  {
    return 1;
  }
  if (std::strcmp(backend, "sse2") == 0)
  {
    return features.sse2 ? 1 : 0;
  }
  if (std::strcmp(backend, "sse4_1") == 0)
  {
    return features.sse4_1 ? 1 : 0;
  }
  if (std::strcmp(backend, "avx2") == 0)
  {
    return features.avx2 ? 1 : 0;
  }
  return 0;
}

extern "C" size_t libgguf_quantize_q4_0_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q4_0_kernel_fn kernel = nullptr;
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
  {
    kernel = quantize_row_q4_0_ref;
  }
  else if (std::strcmp(backend, "sse2") == 0)
  {
    if (!libgguf_get_cpu_features().sse2)
    {
      return 0;
    }
    kernel = quantize_row_q4_0_sse2;
  }
  else if (std::strcmp(backend, "sse4_1") == 0)
  {
    if (!libgguf_get_cpu_features().sse4_1)
    {
      return 0;
    }
    kernel = quantize_row_q4_0_sse4_1;
  }
  else if (std::strcmp(backend, "avx2") == 0)
  {
    if (!libgguf_get_cpu_features().avx2)
    {
      return 0;
    }
    kernel = quantize_row_q4_0_avx2;
  }
  else
  {
    return 0;
  }

  kernel(src, (block_q4_0 *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q4_0, n_per_row);
}

static void quantize_row_q4_0_impl(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  if (!quant_weights)
  {
    quantize_row_q4_0_ref(x, y, n_per_row);
    return;
  }

  float weight[QK4_0];
  int8_t L[QK4_0];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK4_0;
  for (int ib = 0; ib < nb; ++ib)
  {
    const float *xb = x + QK4_0 * ib;
    const float *qw = quant_weights + QK4_0 * ib;
    for (int j = 0; j < QK4_0; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float d = make_qx_quants(QK4_0, 8, xb, L, 1, weight);
    y[ib].d = GGML_FP32_TO_FP16(d);
    for (int j = 0; j < 16; ++j)
    {
      y[ib].qs[j] = L[j] | (L[j + 16] << 4);
    }
  }
}

size_t quantize_q4_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    libgguf_q4_0_kernel()(src, (block_q4_0 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q4_0, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q4_0, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_q4_0_impl(src, (block_q4_0 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

