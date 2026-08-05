#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <emmintrin.h>
#include <xmmintrin.h>

static inline __m128i q5_k_nearest_i32(__m128 v)
{
  const __m128 magic = _mm_set1_ps(12582912.0f);
  return _mm_sub_epi32(
      _mm_and_si128(_mm_castps_si128(_mm_add_ps(v, magic)), _mm_set1_epi32(0x007fffff)),
      _mm_set1_epi32(0x00400000));
}

static inline void q5_k_fill_abs_plus_32(const float *LIBGGUF_RESTRICT x, float av_x, float *LIBGGUF_RESTRICT weights)
{
  const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
  const __m128 av = _mm_set1_ps(av_x);
  for (int l = 0; l < 32; l += 4)
  {
    _mm_storeu_ps(weights + l, _mm_add_ps(av, _mm_and_ps(_mm_loadu_ps(x + l), abs_mask)));
  }
}

static inline void q5_k_quantize_32(const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT L, float d, float dm)
{
  const __m128 dv = _mm_set1_ps(d);
  const __m128 dmv = _mm_set1_ps(dm);
  const __m128i zero = _mm_setzero_si128();
  const __m128i max_q = _mm_set1_epi16(31);

  for (int ii = 0; ii < 32; ii += 8)
  {
    const __m128 v0 = _mm_div_ps(_mm_add_ps(_mm_loadu_ps(x + ii), dmv), dv);
    const __m128 v1 = _mm_div_ps(_mm_add_ps(_mm_loadu_ps(x + ii + 4), dmv), dv);
    __m128i q = _mm_packs_epi32(q5_k_nearest_i32(v0), q5_k_nearest_i32(v1));
    q = _mm_min_epi16(_mm_max_epi16(q, zero), max_q);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q5_k_pack_64(const uint8_t *LIBGGUF_RESTRICT L, uint8_t *LIBGGUF_RESTRICT ql, uint8_t *LIBGGUF_RESTRICT qh,
                                     uint8_t m1, uint8_t m2)
{
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m128i threshold = _mm_set1_epi8(15);
  const __m128i m1v = _mm_set1_epi8((char)m1);
  const __m128i m2v = _mm_set1_epi8((char)m2);

  for (int j = 0; j < 32; j += 16)
  {
    const __m128i l1 = _mm_loadu_si128((const __m128i *)(L + j));
    const __m128i l2 = _mm_loadu_si128((const __m128i *)(L + j + 32));
    const __m128i lo = _mm_and_si128(l1, low_mask);
    const __m128i hi = _mm_slli_epi16(_mm_and_si128(l2, low_mask), 4);
    const __m128i h1 = _mm_and_si128(_mm_cmpgt_epi8(l1, threshold), m1v);
    const __m128i h2 = _mm_and_si128(_mm_cmpgt_epi8(l2, threshold), m2v);
    const __m128i old_h = _mm_loadu_si128((const __m128i *)(qh + j));
    _mm_storeu_si128((__m128i *)(ql + j), _mm_or_si128(lo, hi));
    _mm_storeu_si128((__m128i *)(qh + j), _mm_or_si128(old_h, _mm_or_si128(h1, h2)));
  }
}

extern "C" void quantize_block_q5_k(const float *LIBGGUF_RESTRICT x, block_q5_K *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];
  float weights[32];

  for (int64_t i = 0; i < nb; i++)
  {
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l)
      {
        sum_x2 += x[32 * j + l] * x[32 * j + l];
      }
      const float av_x = sqrtf(sum_x2 / 32);
      q5_k_fill_abs_plus_32(x + 32 * j, av_x, weights);
      scales[j] = make_qkx2_quants(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux,
                                                -0.5f, 0.1f, 15, false);
      if (scales[j] > max_scale)
      {
        max_scale = scales[j];
      }
      if (mins[j] > max_min)
      {
        max_min = mins[j];
      }
    }

    const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      uint8_t ls = (uint8_t)nearest_int(inv_scale * scales[j]);
      uint8_t lm = (uint8_t)nearest_int(inv_min * mins[j]);
      ls = libgguf_min(63, ls);
      lm = libgguf_min(63, lm);
      if (j < 4)
      {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      }
      else
      {
        y[i].scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
        y[i].scales[j - 4] |= (uint8_t)((ls >> 4) << 6);
        y[i].scales[j - 0] |= (uint8_t)((lm >> 4) << 6);
      }
    }
    y[i].d = LIBGGUF_FP32_TO_FP16(max_scale / 63.f);
    y[i].dmin = LIBGGUF_FP32_TO_FP16(max_min / 63.f);

    const float d_base = LIBGGUF_FP16_TO_FP32(y[i].d);
    const float dm_base = LIBGGUF_FP16_TO_FP32(y[i].dmin);
    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = d_base * sc;
      if (!d)
      {
        continue;
      }
      const float dm = dm_base * m;
      q5_k_quantize_32(x + 32 * j, L + 32 * j, d, dm);
    }

    uint8_t *LIBGGUF_RESTRICT qh = y[i].qh;
    uint8_t *LIBGGUF_RESTRICT ql = y[i].qs;
    memset(qh, 0, QK_K / 8);

    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64)
    {
      q5_k_pack_64(L + n, ql, qh, m1, m2);
      m1 <<= 2;
      m2 <<= 2;
      ql += 32;
    }

    x += QK_K;
  }
}


static void quantize_weighted_q5_K(const float *LIBGGUF_RESTRICT x, block_q5_K *LIBGGUF_RESTRICT y, int64_t n_per_row, const float *quant_weights)
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
      ls = libgguf_min(63, ls);
      lm = libgguf_min(63, lm);
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
    y[i].d = LIBGGUF_FP32_TO_FP16(d_block);
    y[i].dmin = LIBGGUF_FP32_TO_FP16(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = LIBGGUF_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
        continue;
      const float dm = LIBGGUF_FP16_TO_FP32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii)
      {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = libgguf_max(0, libgguf_min(31, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *LIBGGUF_RESTRICT qh = y[i].qh;
    uint8_t *LIBGGUF_RESTRICT ql = y[i].qs;
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

size_t quantize_q5_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q5_K, n_per_row);
  if (!quant_weights)
  {
    quantize_block_q5_k(src, (block_q5_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_weighted_q5_K(src, (block_q5_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

