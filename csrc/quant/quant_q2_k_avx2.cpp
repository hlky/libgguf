#include "libgguf_common.h"

extern "C" float libgguf_make_qkx2_quants_avx2(int n, int nmax, const float *RESTRICT x,
                                                const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                float rmin, float rdelta, int nstep, bool use_mad);

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_Q2_K_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_Q2_K_AVX2)
static inline __m256i q2_k_nearest_i32_avx2(__m256 v)
{
  const __m256 magic = _mm256_set1_ps(12582912.0f);
  return _mm256_sub_epi32(
      _mm256_and_si256(_mm256_castps_si256(_mm256_add_ps(v, magic)), _mm256_set1_epi32(0x007fffff)),
      _mm256_set1_epi32(0x00400000));
}

static inline void q2_k_fill_abs_16_avx2(const float *RESTRICT x, float *RESTRICT weights)
{
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  _mm256_storeu_ps(weights + 0, _mm256_and_ps(_mm256_loadu_ps(x + 0), abs_mask));
  _mm256_storeu_ps(weights + 8, _mm256_and_ps(_mm256_loadu_ps(x + 8), abs_mask));
}

static inline void q2_k_quantize_16_avx2(const float *RESTRICT x, uint8_t *RESTRICT L, float d, float dm)
{
  const __m256 dv = _mm256_set1_ps(d);
  const __m256 dmv = _mm256_set1_ps(dm);
  const __m128i zero = _mm_setzero_si128();
  const __m128i max_q = _mm_set1_epi16(3);

  for (int ii = 0; ii < 16; ii += 8)
  {
    const __m256 v = _mm256_div_ps(_mm256_add_ps(_mm256_loadu_ps(x + ii), dmv), dv);
    const __m256i q32 = q2_k_nearest_i32_avx2(v);
    __m128i q = _mm_packs_epi32(_mm256_castsi256_si128(q32), _mm256_extracti128_si256(q32, 1));
    q = _mm_min_epi16(_mm_max_epi16(q, zero), max_q);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q2_k_pack_128_avx2(const uint8_t *RESTRICT L, uint8_t *RESTRICT q)
{
  const __m256i mask = _mm256_set1_epi8(0x03);
  const __m256i q0 = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 0)), mask);
  const __m256i q1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 32)), mask), 2);
  const __m256i q2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 64)), mask), 4);
  const __m256i q3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 96)), mask), 6);
  _mm256_storeu_si256((__m256i *)q, _mm256_or_si256(_mm256_or_si256(q0, q1), _mm256_or_si256(q2, q3)));
}
#endif

extern "C" void quantize_row_q2_K_avx2(const float *RESTRICT x, block_q2_K *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_Q2_K_AVX2)
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[16];
  float weights[16];
  float mins[QK_K / 16];
  float scales[QK_K / 16];
  const float q4scale = 15.f;

  for (int64_t i = 0; i < nb; i++)
  {
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      q2_k_fill_abs_16_avx2(x + 16 * j, weights);
      scales[j] = libgguf_make_qkx2_quants_avx2(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux,
                                                -0.5f, 0.1f, 15, true);
      if (scales[j] > max_scale)
      {
        max_scale = scales[j];
      }
      if (mins[j] > max_min)
      {
        max_min = mins[j];
      }
    }

    if (max_scale > 0)
    {
      const float iscale = q4scale / max_scale;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        y[i].scales[j] = (uint8_t)nearest_int(iscale * scales[j]);
      }
      y[i].d = GGML_FP32_TO_FP16(max_scale / q4scale);
    }
    else
    {
      memset(y[i].scales, 0, QK_K / 16);
      y[i].d = GGML_FP32_TO_FP16(0.f);
    }

    if (max_min > 0)
    {
      const float iscale = q4scale / max_min;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        y[i].scales[j] |= (uint8_t)(nearest_int(iscale * mins[j]) << 4);
      }
      y[i].dmin = GGML_FP32_TO_FP16(max_min / q4scale);
    }
    else
    {
      y[i].dmin = GGML_FP32_TO_FP16(0.f);
    }

    const float d_base = GGML_FP16_TO_FP32(y[i].d);
    const float dm_base = GGML_FP16_TO_FP32(y[i].dmin);
    for (int j = 0; j < QK_K / 16; ++j)
    {
      const float d = d_base * (y[i].scales[j] & 0xF);
      if (!d)
      {
        continue;
      }
      const float dm = dm_base * (y[i].scales[j] >> 4);
      q2_k_quantize_16_avx2(x + 16 * j, L + 16 * j, d, dm);
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      q2_k_pack_128_avx2(L + j, y[i].qs + j / 4);
    }

    x += QK_K;
  }
#else
  quantize_row_q2_K_ref(x, y, k);
#endif
}
