#include "libgguf_common.h"

extern "C" float libgguf_make_q3_quants_avx2(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                              bool do_rmse);

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_Q3_K_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_Q3_K_AVX2)
static inline __m256i q3_k_nearest_i32_avx2(__m256 v)
{
  const __m256 magic = _mm256_set1_ps(12582912.0f);
  return _mm256_sub_epi32(
      _mm256_and_si256(_mm256_castps_si256(_mm256_add_ps(v, magic)), _mm256_set1_epi32(0x007fffff)),
      _mm256_set1_epi32(0x00400000));
}

static inline void q3_k_quantize_16_avx2(const float *RESTRICT x, int8_t *RESTRICT L, float d)
{
  const __m256 dv = _mm256_set1_ps(d);
  const __m128i min_q = _mm_set1_epi16(-4);
  const __m128i max_q = _mm_set1_epi16(3);
  const __m128i offset = _mm_set1_epi16(4);
  const __m128i zero = _mm_setzero_si128();

  for (int ii = 0; ii < 16; ii += 8)
  {
    const __m256 v = _mm256_div_ps(_mm256_loadu_ps(x + ii), dv);
    const __m256i q32 = q3_k_nearest_i32_avx2(v);
    __m128i q = _mm_packs_epi32(_mm256_castsi256_si128(q32), _mm256_extracti128_si256(q32, 1));
    q = _mm_add_epi16(_mm_min_epi16(_mm_max_epi16(q, min_q), max_q), offset);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q3_k_update_hmask_32_avx2(int8_t *RESTRICT L, uint8_t *RESTRICT hmask, uint8_t hm)
{
  const __m256i threshold = _mm256_set1_epi8(3);
  const __m256i high_bit = _mm256_set1_epi8((char)hm);
  const __m256i four = _mm256_set1_epi8(4);
  const __m256i q = _mm256_loadu_si256((const __m256i *)L);
  const __m256i high = _mm256_cmpgt_epi8(q, threshold);
  const __m256i old_h = _mm256_loadu_si256((const __m256i *)hmask);
  _mm256_storeu_si256((__m256i *)hmask, _mm256_or_si256(old_h, _mm256_and_si256(high, high_bit)));
  _mm256_storeu_si256((__m256i *)L, _mm256_sub_epi8(q, _mm256_and_si256(high, four)));
}

static inline void q3_k_pack_128_avx2(const int8_t *RESTRICT L, uint8_t *RESTRICT q)
{
  const __m256i mask = _mm256_set1_epi8(0x03);
  const __m256i q0 = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 0)), mask);
  const __m256i q1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 32)), mask), 2);
  const __m256i q2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 64)), mask), 4);
  const __m256i q3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(L + 96)), mask), 6);
  _mm256_storeu_si256((__m256i *)q, _mm256_or_si256(_mm256_or_si256(q0, q1), _mm256_or_si256(q2, q3)));
}
#endif

extern "C" void quantize_row_q3_K_avx2(const float *RESTRICT x, block_q3_K *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_Q3_K_AVX2)
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
      scales[j] = libgguf_make_q3_quants_avx2(16, 4, x + 16 * j, L + 16 * j, true);
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
        l = (int8_t)(MAX(-32, MIN(31, l)) + 32);
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
      y[i].d = GGML_FP32_TO_FP16(1 / iscale);
    }
    else
    {
      y[i].d = GGML_FP32_TO_FP16(0.f);
    }

    const float d_base = GGML_FP16_TO_FP32(y[i].d);
    for (int j = 0; j < QK_K / 16; ++j)
    {
      int8_t sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (int8_t)((sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32);
      const float d = d_base * sc;
      if (!d)
      {
        continue;
      }
      q3_k_quantize_16_avx2(x + 16 * j, L + 16 * j, d);
    }

    memset(y[i].hmask, 0, QK_K / 8);
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; j += 32)
    {
      q3_k_update_hmask_32_avx2(L + j, y[i].hmask, hm);
      hm <<= 1;
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      q3_k_pack_128_avx2(L + j, y[i].qs + j / 4);
    }

    x += QK_K;
  }
#else
  quantize_row_q3_K(x, y, k);
#endif
}
