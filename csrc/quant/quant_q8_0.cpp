#include "libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_USE_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_USE_SSE2 1
#endif

void quantize_row_q8_0_ref(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++)
    {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j)
    {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = roundf(x0);
    }
  }
}

#if defined(LIBGGUF_USE_AVX2)
static inline __m256i q8_0_round_away_from_zero_avx2(__m256 v, __m256 id)
{
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 scaled = _mm256_mul_ps(v, id);
  const __m256 abs_scaled = _mm256_andnot_ps(sign_mask, scaled);
  const __m256 rounded_abs = _mm256_add_ps(abs_scaled, half);
  const __m256i abs_i = _mm256_cvttps_epi32(rounded_abs);
  const __m256 negative = _mm256_cmp_ps(scaled, _mm256_setzero_ps(), _CMP_LT_OQ);
  const __m256i sign_i = _mm256_castps_si256(negative);
  return _mm256_sub_epi32(_mm256_xor_si256(abs_i, sign_i), sign_i);
}

static void quantize_row_q8_0_avx2(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  static_assert(QK8_0 == 32, "QK8_0 must be 32");

  const int nb = k / QK8_0;
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK8_0;
    __m256 maxv = _mm256_setzero_ps();
    for (int j = 0; j < QK8_0; j += 8)
    {
      const __m256 v = _mm256_loadu_ps(xb + j);
      maxv = _mm256_max_ps(maxv, _mm256_andnot_ps(sign_mask, v));
    }

    alignas(32) float max_parts[8];
    _mm256_store_ps(max_parts, maxv);
    const float amax = MAX(
        MAX(MAX(max_parts[0], max_parts[1]), MAX(max_parts[2], max_parts[3])),
        MAX(MAX(max_parts[4], max_parts[5]), MAX(max_parts[6], max_parts[7])));
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    const __m256 idv = _mm256_set1_ps(id);

    y[i].d = GGML_FP32_TO_FP16(d);

    const __m256i q0 = q8_0_round_away_from_zero_avx2(_mm256_loadu_ps(xb + 0), idv);
    const __m256i q1 = q8_0_round_away_from_zero_avx2(_mm256_loadu_ps(xb + 8), idv);
    const __m256i q2 = q8_0_round_away_from_zero_avx2(_mm256_loadu_ps(xb + 16), idv);
    const __m256i q3 = q8_0_round_away_from_zero_avx2(_mm256_loadu_ps(xb + 24), idv);

    const __m256i q01_i16 = _mm256_packs_epi32(q0, q1);
    const __m256i q23_i16 = _mm256_packs_epi32(q2, q3);
    const __m256i packed_lanes = _mm256_packs_epi16(q01_i16, q23_i16);
    const __m256i packed = _mm256_permutevar8x32_epi32(
        packed_lanes,
        _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256((__m256i *)y[i].qs, packed);
  }
}
#elif defined(LIBGGUF_USE_SSE2)
static inline __m128i q8_0_round_away_from_zero_sse2(__m128 v, __m128 id)
{
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128 scaled = _mm_mul_ps(v, id);
  const __m128 abs_scaled = _mm_andnot_ps(sign_mask, scaled);
  const __m128 rounded_abs = _mm_add_ps(abs_scaled, half);
  const __m128i abs_i = _mm_cvttps_epi32(rounded_abs);
  const __m128 negative = _mm_cmplt_ps(scaled, _mm_setzero_ps());
  const __m128i sign_i = _mm_castps_si128(negative);
  return _mm_sub_epi32(_mm_xor_si128(abs_i, sign_i), sign_i);
}

static void quantize_row_q8_0_sse2(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  static_assert(QK8_0 == 32, "QK8_0 must be 32");

  const int nb = k / QK8_0;
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK8_0;
    __m128 maxv = _mm_setzero_ps();
    for (int j = 0; j < QK8_0; j += 4)
    {
      const __m128 v = _mm_loadu_ps(xb + j);
      maxv = _mm_max_ps(maxv, _mm_andnot_ps(sign_mask, v));
    }

    alignas(16) float max_parts[4];
    _mm_store_ps(max_parts, maxv);
    const float amax = MAX(MAX(max_parts[0], max_parts[1]), MAX(max_parts[2], max_parts[3]));
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    const __m128 idv = _mm_set1_ps(id);

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; j += 8)
    {
      const __m128i q0 = q8_0_round_away_from_zero_sse2(_mm_loadu_ps(xb + j), idv);
      const __m128i q1 = q8_0_round_away_from_zero_sse2(_mm_loadu_ps(xb + j + 4), idv);
      const __m128i packed_i16 = _mm_packs_epi32(q0, q1);
      const __m128i packed_i8 = _mm_packs_epi16(packed_i16, _mm_setzero_si128());
      _mm_storel_epi64((__m128i *)(y[i].qs + j), packed_i8);
    }
  }
}
#endif

size_t quantize_q8_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  const size_t row_size = libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
#if defined(LIBGGUF_USE_AVX2)
  quantize_row_q8_0_avx2(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
#elif defined(LIBGGUF_USE_SSE2)
  quantize_row_q8_0_sse2(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
#else
  quantize_row_q8_0_ref(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
#endif
  return nrow * row_size;
}

