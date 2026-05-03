#include "libgguf_common.h"

extern "C" float libgguf_make_qkx2_quants_avx2(int n, int nmax, const float *RESTRICT x,
                                                const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                float rmin, float rdelta, int nstep, bool use_mad);

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_Q4_K_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_Q4_K_AVX2)
static inline __m256i q4_k_nearest_i32_avx2(__m256 v)
{
  const __m256 magic = _mm256_set1_ps(12582912.0f);
  return _mm256_sub_epi32(
      _mm256_and_si256(_mm256_castps_si256(_mm256_add_ps(v, magic)), _mm256_set1_epi32(0x007fffff)),
      _mm256_set1_epi32(0x00400000));
}

static inline void q4_k_fill_abs_plus_32_avx2(const float *RESTRICT x, float av_x, float *RESTRICT weights)
{
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256 av = _mm256_set1_ps(av_x);
  for (int l = 0; l < 32; l += 8)
  {
    _mm256_storeu_ps(weights + l, _mm256_add_ps(av, _mm256_and_ps(_mm256_loadu_ps(x + l), abs_mask)));
  }
}

static inline void q4_k_quantize_32_avx2(const float *RESTRICT x, uint8_t *RESTRICT L, float d, float dm)
{
  const __m256 dv = _mm256_set1_ps(d);
  const __m256 dmv = _mm256_set1_ps(dm);
  const __m128i zero = _mm_setzero_si128();
  const __m128i max_q = _mm_set1_epi16(15);

  for (int ii = 0; ii < 32; ii += 8)
  {
    const __m256 v = _mm256_div_ps(_mm256_add_ps(_mm256_loadu_ps(x + ii), dmv), dv);
    const __m256i q32 = q4_k_nearest_i32_avx2(v);
    __m128i q = _mm_packs_epi32(_mm256_castsi256_si128(q32), _mm256_extracti128_si256(q32, 1));
    q = _mm_min_epi16(_mm_max_epi16(q, zero), max_q);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q4_k_pack_64_avx2(const uint8_t *RESTRICT L, uint8_t *RESTRICT q)
{
  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i lo = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)L), low_mask);
  const __m256i hi = _mm256_slli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 32)), low_mask), 4);
  _mm256_storeu_si256((__m256i *)q, _mm256_or_si256(lo, hi));
}
#endif

extern "C" void quantize_row_q4_K_avx2(const float *RESTRICT x, block_q4_K *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_Q4_K_AVX2)
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  float weights[32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++)
  {
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l)
        sum_x2 += x[32 * j + l] * x[32 * j + l];
      float av_x = sqrtf(sum_x2 / 32);
      q4_k_fill_abs_plus_32_avx2(x + 32 * j, av_x, weights);
      scales[j] = libgguf_make_qkx2_quants_avx2(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
      if (scales[j] > max_scale)
      {
        max_scale = scales[j];
      }
      if (mins[j] > max_min)
      {
        max_min = mins[j];
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

    const float d_base = GGML_FP16_TO_FP32(y[i].d);
    const float dm_base = GGML_FP16_TO_FP32(y[i].dmin);
    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j)
    {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = d_base * sc;
      if (!d)
        continue;
      const float dm = dm_base * m;
      q4_k_quantize_32_avx2(x + 32 * j, L + 32 * j, d, dm);
    }

    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64)
    {
      q4_k_pack_64_avx2(L + j, q);
      q += 32;
    }

    x += QK_K;
  }
#else
  quantize_row_q4_K(x, y, k);
#endif
}
