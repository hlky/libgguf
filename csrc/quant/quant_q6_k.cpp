#include "libgguf_common.h"
#include "common/libgguf_cpu.h"

#include <cstring>

typedef void (*libgguf_q6_k_kernel_fn)(const float *RESTRICT, block_q6_K *RESTRICT, int64_t);

struct libgguf_q6_k_selection
{
  const char *backend;
  libgguf_q6_k_kernel_fn kernel;
};

extern "C" void quantize_row_q6_K_sse2(const float *RESTRICT x, block_q6_K *RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_sse4_1(const float *RESTRICT x, block_q6_K *RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_avx2(const float *RESTRICT x, block_q6_K *RESTRICT y, int64_t k);

void quantize_row_q6_K_ref(const float *RESTRICT x, block_q6_K *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++)
  {

    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib)
    {

      const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale)
      {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS)
    {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = GGML_FP32_TO_FP16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = GGML_FP32_TO_FP16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
      y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j)
    {
      float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
      if (!d)
      {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *RESTRICT ql = y[i].ql;
    uint8_t *RESTRICT qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128)
    {
      for (int l = 0; l < 32; ++l)
      {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}

static libgguf_q6_k_kernel_fn libgguf_q6_k_kernel_for_backend(const char *backend)
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
  {
    return quantize_row_q6_K_ref;
  }
  if (std::strcmp(backend, "sse2") == 0 && features.sse2)
  {
    return quantize_row_q6_K_sse2;
  }
  if (std::strcmp(backend, "sse4_1") == 0 && features.sse4_1)
  {
    return quantize_row_q6_K_sse4_1;
  }
  if (std::strcmp(backend, "avx2") == 0 && features.avx2)
  {
    return quantize_row_q6_K_avx2;
  }
  return nullptr;
}

static libgguf_q6_k_selection libgguf_q6_k_select_kernel()
{
  if (libgguf_q6_k_kernel_fn kernel = libgguf_q6_k_kernel_for_backend("avx2"))
  {
    return {"avx2", kernel};
  }
  if (libgguf_q6_k_kernel_fn kernel = libgguf_q6_k_kernel_for_backend("sse4_1"))
  {
    return {"sse4_1", kernel};
  }
  if (libgguf_q6_k_kernel_fn kernel = libgguf_q6_k_kernel_for_backend("sse2"))
  {
    return {"sse2", kernel};
  }
  return {"ref", quantize_row_q6_K_ref};
}

static const libgguf_q6_k_selection &libgguf_q6_k_selected()
{
  static const libgguf_q6_k_selection selected = libgguf_q6_k_select_kernel();
  return selected;
}

static libgguf_q6_k_kernel_fn libgguf_q6_k_kernel()
{
  return libgguf_q6_k_selected().kernel;
}

extern "C" const char *libgguf_q6_k_backend(void)
{
  return libgguf_q6_k_selected().backend;
}

extern "C" int libgguf_q6_k_cpu_supports_backend(const char *backend)
{
  return libgguf_q6_k_kernel_for_backend(backend) ? 1 : 0;
}

extern "C" size_t libgguf_quantize_q6_k_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q6_k_kernel_fn kernel = libgguf_q6_k_kernel_for_backend(backend);
  if (!kernel)
  {
    return 0;
  }
  kernel(src, (block_q6_K *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q6_K, n_per_row);
}

static void quantize_row_q6_K_impl(const float *RESTRICT x, block_q6_K *RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];
  // float   weights[16];

  for (int i = 0; i < nb; i++)
  {

    // float sum_x2 = 0;
    // for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
    // float sigma2 = sum_x2/QK_K;

    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib)
    {

      float scale;
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * i + 16 * ib;
        // for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
        // scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, qw);
      }
      else
      {
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
      }
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale)
      {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS)
    {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = GGML_FP32_TO_FP16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = GGML_FP32_TO_FP16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
      y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j)
    {
      float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
      if (!d)
      {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *RESTRICT ql = y[i].ql;
    uint8_t *RESTRICT qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128)
    {
      for (int l = 0; l < 32; ++l)
      {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}

size_t quantize_q6_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q6_K, n_per_row);
  if (!quant_weights)
  {
    libgguf_q6_k_kernel()(src, (block_q6_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_row_q6_K_impl(src, (block_q6_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

