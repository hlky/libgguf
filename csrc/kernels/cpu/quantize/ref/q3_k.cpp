#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


void quantize_block_q3_k(const float *LIBGGUF_RESTRICT x, block_q3_K *LIBGGUF_RESTRICT y, int64_t k)
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
        l = libgguf_max(-32, libgguf_min(31, l)) + 32;
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
      y[i].d = LIBGGUF_FP32_TO_FP16(1 / iscale);
    }
    else
    {
      y[i].d = LIBGGUF_FP32_TO_FP16(0.f);
    }

    int8_t sc;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
      float d = LIBGGUF_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
      {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int(x[16 * j + ii] / d);
        l = libgguf_max(-4, libgguf_min(3, l));
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


static void quantize_weighted_q3_K(const float *LIBGGUF_RESTRICT x, block_q3_K *LIBGGUF_RESTRICT y, int64_t n_per_row, const float *LIBGGUF_RESTRICT quant_weights)
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
    y[i].d = LIBGGUF_FP32_TO_FP16(d_block);

    int8_t sc;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
      float d = LIBGGUF_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
      {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int(x[16 * j + ii] / d);
        l = libgguf_max(-4, libgguf_min(3, l));
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

size_t quantize_q3_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q3_K, n_per_row);
  if (!quant_weights)
  {
    quantize_block_q3_k(src, (block_q3_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_weighted_q3_K(src, (block_q3_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

