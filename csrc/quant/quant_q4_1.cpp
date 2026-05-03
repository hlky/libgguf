#include "libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstring>

typedef void (*libgguf_q4_1_kernel_fn)(const float *RESTRICT, block_q4_1 *RESTRICT, int64_t);

struct libgguf_q4_1_selection
{
  const char *backend;
  libgguf_q4_1_kernel_fn kernel;
};

extern "C" void quantize_row_q4_1_sse2(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_sse4_1(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_avx2(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k);

void quantize_row_q4_1_ref(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k)
{
  const int qk = QK4_1;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < qk; j++)
    {
      const float v = x[i * qk + j];

      if (v < min)
        min = v;
      if (v > max)
        max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);
    y[i].m = GGML_FP32_TO_FP16(min);

    for (int j = 0; j < qk / 2; ++j)
    {
      const float x0 = (x[i * qk + 0 + j] - min) * id;
      const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static void quantize_row_q4_1_impl(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  if (!quant_weights)
  {
    quantize_row_q4_1_ref(x, y, n_per_row);
    return;
  }

  float weight[QK4_1];
  uint8_t L[QK4_1], Laux[QK4_1];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK4_1;
  for (int ib = 0; ib < nb; ++ib)
  {
    const float *xb = x + QK4_1 * ib;
    const float *qw = quant_weights + QK4_1 * ib;
    for (int j = 0; j < QK4_1; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float min;
    float d = make_qkx3_quants(QK4_1, 15, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
    y[ib].d = GGML_FP32_TO_FP16(d);
    y[ib].m = GGML_FP32_TO_FP16(-min);
    for (int j = 0; j < 16; ++j)
    {
      y[ib].qs[j] = L[j] | (L[j + 16] << 4);
    }
  }
}

static libgguf_q4_1_selection libgguf_q4_1_select_kernel()
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();

  if (features.sse2)
    return {"sse2", quantize_row_q4_1_sse2};
  if (features.sse4_1)
    return {"sse4_1", quantize_row_q4_1_sse4_1};
  if (features.avx2)
    return {"avx2", quantize_row_q4_1_avx2};
  return {"ref", quantize_row_q4_1_ref};
}

static const libgguf_q4_1_selection &libgguf_q4_1_selected()
{
  static const libgguf_q4_1_selection selected = libgguf_q4_1_select_kernel();
  return selected;
}

extern "C" const char *libgguf_q4_1_backend(void)
{
  return libgguf_q4_1_selected().backend;
}

extern "C" int libgguf_q4_1_cpu_supports_backend(const char *backend)
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
    return 1;
  if (std::strcmp(backend, "sse2") == 0)
    return features.sse2 ? 1 : 0;
  if (std::strcmp(backend, "sse4_1") == 0)
    return features.sse4_1 ? 1 : 0;
  if (std::strcmp(backend, "avx2") == 0)
    return features.avx2 ? 1 : 0;
  return 0;
}

static libgguf_q4_1_kernel_fn libgguf_q4_1_kernel_for_backend(const char *backend)
{
  if (!libgguf_q4_1_cpu_supports_backend(backend))
    return nullptr;
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
    return quantize_row_q4_1_ref;
  if (std::strcmp(backend, "sse2") == 0)
    return quantize_row_q4_1_sse2;
  if (std::strcmp(backend, "sse4_1") == 0)
    return quantize_row_q4_1_sse4_1;
  if (std::strcmp(backend, "avx2") == 0)
    return quantize_row_q4_1_avx2;
  return nullptr;
}

extern "C" size_t libgguf_quantize_q4_1_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q4_1_kernel_fn kernel = libgguf_q4_1_kernel_for_backend(backend);
  if (!kernel)
    return 0;
  kernel(src, (block_q4_1 *)dst, nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q4_1, n_per_row);
}

size_t quantize_q4_1(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    libgguf_q4_1_selected().kernel(src, (block_q4_1 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q4_1, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q4_1, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_q4_1_impl(src, (block_q4_1 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

static void quantize_row_q5_0_impl(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  if (!quant_weights)
  {
    quantize_row_q5_0_ref(x, y, n_per_row);
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

