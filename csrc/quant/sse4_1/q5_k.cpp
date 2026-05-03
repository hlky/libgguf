#include "libgguf_common.h"

extern "C" float libgguf_make_qkx2_quants_sse4_1(int n, int nmax, const float *RESTRICT x,
                                                  const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                  float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                  float rmin, float rdelta, int nstep, bool use_mad);

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_Q5_K_SSE4_1 1
#endif

#if defined(LIBGGUF_BUILD_Q5_K_SSE4_1)
static inline __m128i q5_k_nearest_i32_sse4_1(__m128 v)
{
  const __m128 magic = _mm_set1_ps(12582912.0f);
  return _mm_sub_epi32(
      _mm_and_si128(_mm_castps_si128(_mm_add_ps(v, magic)), _mm_set1_epi32(0x007fffff)),
      _mm_set1_epi32(0x00400000));
}

static inline void q5_k_fill_abs_plus_32_sse4_1(const float *RESTRICT x, float av_x, float *RESTRICT weights)
{
  const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
  const __m128 av = _mm_set1_ps(av_x);
  for (int l = 0; l < 32; l += 4)
  {
    _mm_storeu_ps(weights + l, _mm_add_ps(av, _mm_and_ps(_mm_loadu_ps(x + l), abs_mask)));
  }
}

static inline void q5_k_quantize_32_sse4_1(const float *RESTRICT x, uint8_t *RESTRICT L, float d, float dm)
{
  const __m128 dv = _mm_set1_ps(d);
  const __m128 dmv = _mm_set1_ps(dm);
  const __m128i zero = _mm_setzero_si128();
  const __m128i max_q = _mm_set1_epi16(31);

  for (int ii = 0; ii < 32; ii += 8)
  {
    const __m128 v0 = _mm_div_ps(_mm_add_ps(_mm_loadu_ps(x + ii), dmv), dv);
    const __m128 v1 = _mm_div_ps(_mm_add_ps(_mm_loadu_ps(x + ii + 4), dmv), dv);
    __m128i q = _mm_packs_epi32(q5_k_nearest_i32_sse4_1(v0), q5_k_nearest_i32_sse4_1(v1));
    q = _mm_min_epi16(_mm_max_epi16(q, zero), max_q);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q5_k_pack_64_sse4_1(const uint8_t *RESTRICT L, uint8_t *RESTRICT ql, uint8_t *RESTRICT qh,
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
#endif

extern "C" void quantize_row_q5_K_sse4_1(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_Q5_K_SSE4_1)
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
      q5_k_fill_abs_plus_32_sse4_1(x + 32 * j, av_x, weights);
      scales[j] = libgguf_make_qkx2_quants_sse4_1(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux,
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
      ls = MIN(63, ls);
      lm = MIN(63, lm);
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
    y[i].d = GGML_FP32_TO_FP16(max_scale / 63.f);
    y[i].dmin = GGML_FP32_TO_FP16(max_min / 63.f);

    const float d_base = GGML_FP16_TO_FP32(y[i].d);
    const float dm_base = GGML_FP16_TO_FP32(y[i].dmin);
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
      q5_k_quantize_32_sse4_1(x + 32 * j, L + 32 * j, d, dm);
    }

    uint8_t *RESTRICT qh = y[i].qh;
    uint8_t *RESTRICT ql = y[i].qs;
    memset(qh, 0, QK_K / 8);

    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64)
    {
      q5_k_pack_64_sse4_1(L + n, ql, qh, m1, m2);
      m1 <<= 2;
      m2 <<= 2;
      ql += 32;
    }

    x += QK_K;
  }
#else
  quantize_row_q5_K(x, y, k);
#endif
}
