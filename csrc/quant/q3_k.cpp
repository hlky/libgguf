#include "libgguf_common.h"
#include "common/libgguf_backend.h"

#include <cstring>

typedef void (*libgguf_q3_k_kernel_fn)(const float *RESTRICT, block_q3_K *RESTRICT, int64_t);

struct libgguf_q3_k_selection
{
  const char *backend;
  libgguf_q3_k_kernel_fn kernel;
};

#if !LIBGGUF_CPU_BACKEND_REF
extern "C" void LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q3_K)(const float *RESTRICT x, block_q3_K *RESTRICT y, int64_t k);
#endif

void quantize_row_q3_K(const float *RESTRICT x, block_q3_K *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++)
  {

    float max_scale = 0;
    float amax = 0;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j, true);
      float scale = fabsf(scales[j]);
      if (scale > amax)
      {
        amax = scale;
        max_scale = scales[j];
      }
    }

    memset(y[i].scales, 0, 12);
    if (max_scale)
    {
      float iscale = -32.f / max_scale;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        int8_t l = nearest_int(iscale * scales[j]);
        l = MAX(-32, MIN(31, l)) + 32;
        if (j < 8)
        {
          y[i].scales[j] = l & 0xF;
        }
        else
        {
          y[i].scales[j - 8] |= ((l & 0xF) << 4);
        }
        l >>= 4;
        y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
      }
      y[i].d = GGML_FP32_TO_FP16(1 / iscale);
    }
    else
    {
      y[i].d = GGML_FP32_TO_FP16(0.f);
    }

    int8_t sc;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
      float d = GGML_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
      {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-4, MIN(3, l));
        L[16 * j + ii] = l + 4;
      }
    }

    memset(y[i].hmask, 0, QK_K / 8);
    // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j)
    {
      if (L[j] > 3)
      {
        y[i].hmask[m] |= hm;
        L[j] -= 4;
      }
      if (++m == QK_K / 8)
      {
        m = 0;
        hm <<= 1;
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

static libgguf_q3_k_kernel_fn libgguf_q3_k_kernel_for_backend(const char *backend)
{
  if (libgguf_cpu_backend_is_ref_request(backend))
  {
    return quantize_row_q3_K;
  }
#if !LIBGGUF_CPU_BACKEND_REF
  if (libgguf_cpu_backend_is_compiled_request(backend))
  {
    return LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q3_K);
  }
#endif
  return nullptr;
}

static libgguf_q3_k_selection libgguf_q3_k_select_kernel()
{
#if LIBGGUF_CPU_BACKEND_REF
  return {LIBGGUF_CPU_BACKEND_NAME, quantize_row_q3_K};
#else
  return {LIBGGUF_CPU_BACKEND_NAME, LIBGGUF_CPU_BACKEND_SYMBOL(quantize_row_q3_K)};
#endif
}

static const libgguf_q3_k_selection &libgguf_q3_k_selected()
{
  static const libgguf_q3_k_selection selected = libgguf_q3_k_select_kernel();
  return selected;
}

static libgguf_q3_k_kernel_fn libgguf_q3_k_kernel()
{
  return libgguf_q3_k_selected().kernel;
}

extern "C" const char *libgguf_q3_k_backend(void)
{
  return libgguf_q3_k_selected().backend;
}

extern "C" int libgguf_q3_k_cpu_supports_backend(const char *backend)
{
  return libgguf_cpu_backend_supports_request(backend) ? 1 : 0;
}

extern "C" size_t libgguf_quantize_q3_k_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_q3_k_kernel_fn kernel = libgguf_q3_k_kernel_for_backend(backend);
  if (!kernel)
  {
    return 0;
  }
  kernel(src, (block_q3_K *)dst, (int64_t)nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_Q3_K, n_per_row);
}

static void quantize_row_q3_K_impl(const float *RESTRICT x, block_q3_K *RESTRICT y, int64_t n_per_row, const float *RESTRICT quant_weights)
{
  assert(n_per_row % QK_K == 0);
  const int nb = n_per_row / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];
  float weight[16];
  float sw[QK_K / 16];
  int8_t Ls[QK_K / 16];

  for (int i = 0; i < nb; i++)
  {

    float sumx2 = 0;
    for (int j = 0; j < QK_K; ++j)
      sumx2 += x[j] * x[j];
    float sigma2 = 2 * sumx2 / QK_K;

    for (int j = 0; j < QK_K / 16; ++j)
    {
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * i + 16 * j;
        for (int l = 0; l < 16; ++l)
          weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
      }
      else
      {
        for (int l = 0; l < 16; ++l)
          weight[l] = x[16 * j + l] * x[16 * j + l];
      }
      float sumw = 0;
      for (int l = 0; l < 16; ++l)
        sumw += weight[l];
      sw[j] = sumw;

      scales[j] = make_qx_quants(16, 4, x + 16 * j, L + 16 * j, 1, weight);
    }

    memset(y[i].scales, 0, 12);

    float d_block = make_qx_quants(QK_K / 16, 32, scales, Ls, 1, sw);
    for (int j = 0; j < QK_K / 16; ++j)
    {
      int l = Ls[j];
      if (j < 8)
      {
        y[i].scales[j] = l & 0xF;
      }
      else
      {
        y[i].scales[j - 8] |= ((l & 0xF) << 4);
      }
      l >>= 4;
      y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
    }
    y[i].d = GGML_FP32_TO_FP16(d_block);

    int8_t sc;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
      float d = GGML_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
      {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-4, MIN(3, l));
        L[16 * j + ii] = l + 4;
      }
    }

    memset(y[i].hmask, 0, QK_K / 8);
    // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j)
    {
      if (L[j] > 3)
      {
        y[i].hmask[m] |= hm;
        L[j] -= 4;
      }
      if (++m == QK_K / 8)
      {
        m = 0;
        hm <<= 1;
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

size_t quantize_q3_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q3_K, n_per_row);
  if (!quant_weights)
  {
    libgguf_q3_k_kernel()(src, (block_q3_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_row_q3_K_impl(src, (block_q3_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

// ====================== 4-bit (de)-quantization
