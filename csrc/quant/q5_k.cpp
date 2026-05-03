#include "libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstring>

typedef void (*libgguf_q5_k_kernel_fn)(const float *RESTRICT, block_q5_K *RESTRICT, int64_t);

struct libgguf_q5_k_selection
{
  const char *backend;
  libgguf_q5_k_kernel_fn kernel;
};

extern "C" void quantize_row_q5_K_sse2(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_sse4_1(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_avx2(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t k);

void quantize_row_q5_K(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  uint8_t L[QK_K];
  float mins[QK_K / 32];
  float scales[QK_K / 32];
  float weights[32];
  uint8_t Laux[32];

  for (int i = 0; i < nb; i++)
  {
    float max_scale = 0; // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      // scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l)
        sum_x2 += x[32 * j + l] * x[32 * j + l];
      float av_x = sqrtf(sum_x2 / 32);
      for (int l = 0; l < 32; ++l)
        weights[l] = av_x + fabsf(x[32 * j + l]);
      scales[j] = make_qkx2_quants(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
      float scale = scales[j];
      if (scale > max_scale)
      {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min)
      {
        max_min = min;
      }
    }

    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4)
      {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      }
      else
      {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = GGML_FP32_TO_FP16(max_scale / 63.f);
    y[i].dmin = GGML_FP32_TO_FP16(max_min / 63.f);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
        continue;
      const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii)
      {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(31, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *RESTRICT qh = y[i].qh;
    uint8_t *RESTRICT ql = y[i].qs;
    memset(qh, 0, QK_K / 8);

    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64)
    {
      for (int j = 0; j < 32; ++j)
      {
        int l1 = L[n + j];
        if (l1 > 15)
        {
          l1 -= 16;
          qh[j] |= m1;
        }
        int l2 = L[n + j + 32];
        if (l2 > 15)
        {
          l2 -= 16;
          qh[j] |= m2;
        }
        ql[j] = l1 | (l2 << 4);
      }
      m1 <<= 2;
      m2 <<= 2;
      ql += 32;
    }

    x += QK_K;
  }
}

static libgguf_q5_k_kernel_fn libgguf_q5_k_kernel_for_backend(const char *backend)
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
  {
    return quantize_row_q5_K;
  }
  if (std::strcmp(backend, "sse2") == 0 && features.sse2)
  {
    return quantize_row_q5_K_sse2;
  }
  if (std::strcmp(backend, "sse4_1") == 0 && features.sse4_1)
  {
    return quantize_row_q5_K_sse4_1;
  }
  if (std::strcmp(backend, "avx2") == 0 && features.avx2)
  {
    return quantize_row_q5_K_avx2;
  }
  return nullptr;
}

static libgguf_q5_k_selection libgguf_q5_k_select_kernel()
{
  if (libgguf_q5_k_kernel_fn kernel = libgguf_q5_k_kernel_for_backend("sse4_1"))
  {
    return {"sse4_1", kernel};
  }
  if (libgguf_q5_k_kernel_fn kernel = libgguf_q5_k_kernel_for_backend("sse2"))
  {
    return {"sse2", kernel};
  }
  if (libgguf_q5_k_kernel_fn kernel = libgguf_q5_k_kernel_for_backend("avx2"))
  {
    return {"avx2", kernel};
  }
  return {"ref", quantize_row_q5_K};
}

static const libgguf_q5_k_selection &libgguf_q5_k_selected()
{
  static const libgguf_q5_k_selection selected = libgguf_q5_k_select_kernel();
  return selected;
}

static libgguf_q5_k_kernel_fn libgguf_q5_k_kernel()
{
  return libgguf_q5_k_selected().kernel;
}

extern "C" const char *libgguf_q5_k_backend(void)
{
  return libgguf_q5_k_selected().backend;
}

extern "C" int libgguf_q5_k_cpu_supports_backend(const char *backend)
{
  return libgguf_q5_k_kernel_for_backend(backend) ? 1 : 0;
}

extern "C" size_t libgguf_quantize_q5_k_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q5_k_kernel_fn kernel = libgguf_q5_k_kernel_for_backend(backend);
  if (!kernel)
  {
    return 0;
  }
  kernel(src, (block_q5_K *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q5_K, n_per_row);
}

static void quantize_row_q5_K_impl(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  uint8_t Ls[QK_K / 32];
  uint8_t Lm[QK_K / 32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];
  float sw[QK_K / 32];
  float weights[32];

  for (int i = 0; i < nb; i++)
  {

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l)
      sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; ++j)
    {
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * i + 32 * j;
        for (int l = 0; l < 32; ++l)
          weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
      }
      else
      {
        for (int l = 0; l < 32; ++l)
          weights[l] = av_x + fabsf(x[32 * j + l]);
      }
      float sumw = 0;
      for (int l = 0; l < 32; ++l)
        sumw += weights[l];
      sw[j] = sumw;

      scales[j] = make_qkx3_quants(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = make_qp_quants(QK_K / 32, 63, mins, Lm, sw);

    for (int j = 0; j < QK_K / 32; ++j)
    {
      uint8_t ls = Ls[j];
      uint8_t lm = Lm[j];
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4)
      {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      }
      else
      {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = GGML_FP32_TO_FP16(d_block);
    y[i].dmin = GGML_FP32_TO_FP16(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
        continue;
      const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii)
      {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(31, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *RESTRICT qh = y[i].qh;
    uint8_t *RESTRICT ql = y[i].qs;
    memset(qh, 0, QK_K / 8);

    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64)
    {
      for (int j = 0; j < 32; ++j)
      {
        int l1 = L[n + j];
        if (l1 > 15)
        {
          l1 -= 16;
          qh[j] |= m1;
        }
        int l2 = L[n + j + 32];
        if (l2 > 15)
        {
          l2 -= 16;
          qh[j] |= m2;
        }
        ql[j] = l1 | (l2 << 4);
      }
      m1 <<= 2;
      m2 <<= 2;
      ql += 32;
    }

    x += QK_K;
  }
}

size_t quantize_q5_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q5_K, n_per_row);
  if (!quant_weights)
  {
    libgguf_q5_k_kernel()(src, (block_q5_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_row_q5_K_impl(src, (block_q5_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

// ====================== 6-bit (de)-quantization
