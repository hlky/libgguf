#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <emmintrin.h>
#include <xmmintrin.h>

static inline __m128i q3_k_nearest_i32(__m128 v)
{
  const __m128 magic = _mm_set1_ps(12582912.0f);
  return _mm_sub_epi32(
      _mm_and_si128(_mm_castps_si128(_mm_add_ps(v, magic)), _mm_set1_epi32(0x007fffff)),
      _mm_set1_epi32(0x00400000));
}

static inline void q3_k_quantize_16(const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, float d)
{
  const __m128 dv = _mm_set1_ps(d);
  const __m128i min_q = _mm_set1_epi16(-4);
  const __m128i max_q = _mm_set1_epi16(3);
  const __m128i offset = _mm_set1_epi16(4);
  const __m128i zero = _mm_setzero_si128();

  for (int ii = 0; ii < 16; ii += 8)
  {
    const __m128 v0 = _mm_div_ps(_mm_loadu_ps(x + ii), dv);
    const __m128 v1 = _mm_div_ps(_mm_loadu_ps(x + ii + 4), dv);
    __m128i q = _mm_packs_epi32(q3_k_nearest_i32(v0), q3_k_nearest_i32(v1));
    q = _mm_add_epi16(_mm_min_epi16(_mm_max_epi16(q, min_q), max_q), offset);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q3_k_update_hmask_32(int8_t *LIBGGUF_RESTRICT L, uint8_t *LIBGGUF_RESTRICT hmask, uint8_t hm)
{
  const __m128i threshold = _mm_set1_epi8(3);
  const __m128i high_bit = _mm_set1_epi8((char)hm);
  const __m128i four = _mm_set1_epi8(4);
  for (int l = 0; l < 32; l += 16)
  {
    const __m128i q = _mm_loadu_si128((const __m128i *)(L + l));
    const __m128i high = _mm_cmpgt_epi8(q, threshold);
    const __m128i old_h = _mm_loadu_si128((const __m128i *)(hmask + l));
    _mm_storeu_si128((__m128i *)(hmask + l), _mm_or_si128(old_h, _mm_and_si128(high, high_bit)));
    _mm_storeu_si128((__m128i *)(L + l), _mm_sub_epi8(q, _mm_and_si128(high, four)));
  }
}

static inline void q3_k_pack_128(const int8_t *LIBGGUF_RESTRICT L, uint8_t *LIBGGUF_RESTRICT q)
{
  const __m128i mask = _mm_set1_epi8(0x03);
  for (int l = 0; l < 32; l += 16)
  {
    const __m128i q0 = _mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 0)), mask);
    const __m128i q1 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 32)), mask), 2);
    const __m128i q2 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 64)), mask), 4);
    const __m128i q3 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 96)), mask), 6);
    _mm_storeu_si128((__m128i *)(q + l), _mm_or_si128(_mm_or_si128(q0, q1), _mm_or_si128(q2, q3)));
  }
}

extern "C" void quantize_block_q3_k(const float *LIBGGUF_RESTRICT x, block_q3_K *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int64_t i = 0; i < nb; i++)
  {
    float max_scale = 0;
    float amax = 0;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j, true);
      const float scale = fabsf(scales[j]);
      if (scale > amax)
      {
        amax = scale;
        max_scale = scales[j];
      }
    }

    memset(y[i].scales, 0, 12);
    if (max_scale)
    {
      const float iscale = -32.f / max_scale;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        int8_t l = (int8_t)nearest_int(iscale * scales[j]);
        l = (int8_t)(libgguf_max(-32, libgguf_min(31, l)) + 32);
        if (j < 8)
        {
          y[i].scales[j] = l & 0xF;
        }
        else
        {
          y[i].scales[j - 8] |= (uint8_t)((l & 0xF) << 4);
        }
        l >>= 4;
        y[i].scales[j % 4 + 8] |= (uint8_t)(l << (2 * (j / 4)));
      }
      y[i].d = LIBGGUF_FP32_TO_FP16(1 / iscale);
    }
    else
    {
      y[i].d = LIBGGUF_FP32_TO_FP16(0.f);
    }

    const float d_base = LIBGGUF_FP16_TO_FP32(y[i].d);
    for (int j = 0; j < QK_K / 16; ++j)
    {
      int8_t sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (int8_t)((sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32);
      const float d = d_base * sc;
      if (!d)
      {
        continue;
      }
      q3_k_quantize_16(x + 16 * j, L + 16 * j, d);
    }

    memset(y[i].hmask, 0, QK_K / 8);
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; j += 32)
    {
      q3_k_update_hmask_32(L + j, y[i].hmask, hm);
      hm <<= 1;
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      q3_k_pack_128(L + j, y[i].qs + j / 4);
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

