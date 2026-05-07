#include "libgguf_common.h"
#include "common/libgguf_backend.h"

#include <cstring>

typedef void (*libgguf_q2_k_kernel_fn)(const float *RESTRICT, block_q2_K *RESTRICT, int64_t);

struct libgguf_q2_k_selection
{
  const char *backend;
  libgguf_q2_k_kernel_fn kernel;
};

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q2_K)(const float *RESTRICT x, block_q2_K *RESTRICT y, int64_t k);
#endif

void quantize_row_q2_K(const float *RESTRICT x, block_q2_K *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[16];
  float weights[16];
  float mins[QK_K / 16];
  float scales[QK_K / 16];

  const float q4scale = 15.f;

  for (int i = 0; i < nb; i++)
  {
    float max_scale = 0; // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      for (int l = 0; l < 16; ++l)
        weights[l] = fabsf(x[16 * j + l]);
      scales[j] = make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
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

    if (max_scale > 0)
    {
      float iscale = q4scale / max_scale;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        int l = nearest_int(iscale * scales[j]);
        y[i].scales[j] = l;
      }
      y[i].d = GGML_FP32_TO_FP16(max_scale / q4scale);
    }
    else
    {
      for (int j = 0; j < QK_K / 16; ++j)
        y[i].scales[j] = 0;
      y[i].d = GGML_FP32_TO_FP16(0.f);
    }
    if (max_min > 0)
    {
      float iscale = q4scale / max_min;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        int l = nearest_int(iscale * mins[j]);
        y[i].scales[j] |= (l << 4);
      }
      y[i].dmin = GGML_FP32_TO_FP16(max_min / q4scale);
    }
    else
    {
      y[i].dmin = GGML_FP32_TO_FP16(0.f);
    }
    for (int j = 0; j < QK_K / 16; ++j)
    {
      const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
      if (!d)
        continue;
      const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int((x[16 * j + ii] + dm) / d);
        l = MAX(0, MIN(3, l));
        L[16 * j + ii] = l;
      }
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      for (int l = 0; l < 32; ++l)
      {
        y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }

    x += QK_K;
  }
}

static libgguf_q2_k_kernel_fn libgguf_q2_k_kernel_for_backend(const char *backend)
{
  if (libgguf_cpu_backend_is_ref_request(backend))
  {
    return quantize_row_q2_K;
  }
#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
  {
    return LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q2_K);
  }
#endif
  return nullptr;
}

static libgguf_q2_k_selection libgguf_q2_k_select_kernel()
{
#if LIBGGUF_CPU_BACKEND_REF
  return {LIBGGUF_CPU_BACKEND_NAME, quantize_row_q2_K};
#else
  return {LIBGGUF_CPU_BACKEND_NAME, LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q2_K)};
#endif
}

static const libgguf_q2_k_selection &libgguf_q2_k_selected()
{
  static const libgguf_q2_k_selection selected = libgguf_q2_k_select_kernel();
  return selected;
}

static libgguf_q2_k_kernel_fn libgguf_q2_k_kernel()
{
  return libgguf_q2_k_selected().kernel;
}

extern "C" const char *libgguf_q2_k_backend(void)
{
  return libgguf_q2_k_selected().backend;
}

extern "C" int libgguf_q2_k_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

extern "C" size_t libgguf_quantize_q2_k_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q2_k_kernel_fn kernel = libgguf_q2_k_kernel_for_backend(backend);
  if (!kernel)
  {
    return 0;
  }
  kernel(src, (block_q2_K *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q2_K, n_per_row);
}

static void quantize_row_q2_K_impl(const float *RESTRICT x, block_q2_K *RESTRICT y, int k, const float *RESTRICT quant_weights)
{
  assert(quant_weights);
  assert(k % QK_K == 0);
  const int nb = k / QK_K;
  const bool requantize = true;

  uint8_t L[QK_K];
  uint8_t Laux[16];
  float mins[QK_K / 16];
  float scales[QK_K / 16];
  float sw[QK_K / 16];
  float weight[16];
  uint8_t Ls[QK_K / 16], Lm[QK_K / 16];

  for (int i = 0; i < nb; i++)
  {
    memset(sw, 0, QK_K / 16 * sizeof(float));
    float sumx2 = 0;
    for (int j = 0; j < QK_K; ++j)
      sumx2 += x[j] * x[j];
    float sigma2 = sumx2 / QK_K;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      const float *RESTRICT qw = quant_weights + QK_K * i + 16 * j;
      for (int l = 0; l < 16; ++l)
        weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
      for (int l = 0; l < QK_K / 16; ++l)
        sw[j] += weight[l];
      scales[j] = make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float dm, mm;
    dm = make_qp_quants(QK_K / 16, 15, scales, Ls, sw);
    mm = make_qp_quants(QK_K / 16, 15, mins, Lm, sw);

    y[i].d = GGML_FP32_TO_FP16(dm);
    y[i].dmin = GGML_FP32_TO_FP16(mm);
    dm = GGML_FP16_TO_FP32(y[i].d);
    mm = GGML_FP16_TO_FP32(y[i].dmin);

    for (int j = 0; j < QK_K / 16; ++j)
    {
      y[i].scales[j] = Ls[j] | (Lm[j] << 4);
    }

    if (requantize)
    {
      for (int j = 0; j < QK_K / 16; ++j)
      {
        const float d = dm * (y[i].scales[j] & 0xF);
        if (!d)
          continue;
        const float m = mm * (y[i].scales[j] >> 4);
        for (int ii = 0; ii < 16; ++ii)
        {
          int l = nearest_int((x[16 * j + ii] + m) / d);
          l = MAX(0, MIN(3, l));
          L[16 * j + ii] = l;
        }
      }
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      for (int l = 0; l < 32; ++l)
      {
        y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }

    x += QK_K;
  }
}

size_t quantize_q2_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q2_K, n_per_row);
  if (!quant_weights)
  {
    libgguf_q2_k_kernel()(src, (block_q2_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_row_q2_K_impl(src, (block_q2_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

//========================= 3-bit (de)-quantization
