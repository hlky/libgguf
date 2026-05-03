#include "libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_AVX2)
static inline __m256i tq1_0_round_away_from_zero_8_avx2(
    __m256 v,
    __m256 id,
    __m256 sign_mask,
    __m256 half,
    __m256 zero,
    __m256i one)
{
  const __m256 scaled = _mm256_mul_ps(v, id);
  const __m256 abs_scaled = _mm256_andnot_ps(sign_mask, scaled);
  const __m256 rounded_abs = _mm256_add_ps(abs_scaled, half);
  const __m256i abs_i = _mm256_cvttps_epi32(rounded_abs);
  const __m256 negative = _mm256_cmp_ps(scaled, zero, _CMP_LT_OQ);
  const __m256i sign_i = _mm256_castps_si256(negative);
  return _mm256_add_epi32(_mm256_sub_epi32(_mm256_xor_si256(abs_i, sign_i), sign_i), one);
}

static inline __m128i tq1_0_round_away_from_zero_4_avx2(
    __m128 v,
    __m128 id,
    __m128 sign_mask,
    __m128 half,
    __m128 zero,
    __m128i one)
{
  const __m128 scaled = _mm_mul_ps(v, id);
  const __m128 abs_scaled = _mm_andnot_ps(sign_mask, scaled);
  const __m128 rounded_abs = _mm_add_ps(abs_scaled, half);
  const __m128i abs_i = _mm_cvttps_epi32(rounded_abs);
  const __m128 negative = _mm_cmplt_ps(scaled, zero);
  const __m128i sign_i = _mm_castps_si128(negative);
  return _mm_add_epi32(_mm_sub_epi32(_mm_xor_si128(abs_i, sign_i), sign_i), one);
}

static inline uint8_t tq1_0_pack5(int a, int b, int c, int d, int e)
{
  uint16_t q = (uint16_t)(((((a * 3 + b) * 3 + c) * 3 + d) * 3 + e));
  return (uint8_t)((q * 256 + 242) / 243);
}

static inline uint8_t tq1_0_pack4_high(int a, int b, int c, int d)
{
  uint16_t q = (uint16_t)(((((a * 3 + b) * 3 + c) * 3 + d) * 3));
  return (uint8_t)((q * 256 + 242) / 243);
}

static inline void tq1_0_pack_8_avx2(
    uint8_t *RESTRICT dst,
    const float *RESTRICT src,
    size_t stride,
    __m256 id,
    __m256 sign_mask,
    __m256 half,
    __m256 zero,
    __m256i one)
{
  int32_t q0[8], q1[8], q2[8], q3[8], q4[8];
  _mm256_storeu_si256((__m256i *)q0, tq1_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + 0 * stride), id, sign_mask, half, zero, one));
  _mm256_storeu_si256((__m256i *)q1, tq1_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + 1 * stride), id, sign_mask, half, zero, one));
  _mm256_storeu_si256((__m256i *)q2, tq1_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + 2 * stride), id, sign_mask, half, zero, one));
  _mm256_storeu_si256((__m256i *)q3, tq1_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + 3 * stride), id, sign_mask, half, zero, one));
  _mm256_storeu_si256((__m256i *)q4, tq1_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + 4 * stride), id, sign_mask, half, zero, one));
  for (int lane = 0; lane < 8; ++lane)
  {
    dst[lane] = tq1_0_pack5(q0[lane], q1[lane], q2[lane], q3[lane], q4[lane]);
  }
}

static inline void tq1_0_pack_high4_avx2(
    uint8_t *RESTRICT dst,
    const float *RESTRICT src,
    __m128 id,
    __m128 sign_mask,
    __m128 half,
    __m128 zero,
    __m128i one)
{
  int32_t q0[4], q1[4], q2[4], q3[4];
  _mm_storeu_si128((__m128i *)q0, tq1_0_round_away_from_zero_4_avx2(_mm_loadu_ps(src + 0 * sizeof(block_tq1_0::qh)), id, sign_mask, half, zero, one));
  _mm_storeu_si128((__m128i *)q1, tq1_0_round_away_from_zero_4_avx2(_mm_loadu_ps(src + 1 * sizeof(block_tq1_0::qh)), id, sign_mask, half, zero, one));
  _mm_storeu_si128((__m128i *)q2, tq1_0_round_away_from_zero_4_avx2(_mm_loadu_ps(src + 2 * sizeof(block_tq1_0::qh)), id, sign_mask, half, zero, one));
  _mm_storeu_si128((__m128i *)q3, tq1_0_round_away_from_zero_4_avx2(_mm_loadu_ps(src + 3 * sizeof(block_tq1_0::qh)), id, sign_mask, half, zero, one));
  for (int lane = 0; lane < 4; ++lane)
  {
    dst[lane] = tq1_0_pack4_high(q0[lane], q1[lane], q2[lane], q3[lane]);
  }
}
#endif

extern "C" void quantize_row_tq1_0_avx2(const float *RESTRICT x, block_tq1_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");
  static_assert(sizeof(block_tq1_0::qs) == 48, "unexpected TQ1_0 qs size");
  static_assert(sizeof(block_tq1_0::qh) == 4, "unexpected TQ1_0 qh size");

  const int64_t nb = k / QK_K;
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256i one = _mm256_set1_epi32(1);
  const __m128 sign_mask128 = _mm_set1_ps(-0.0f);
  const __m128 half128 = _mm_set1_ps(0.5f);
  const __m128 zero128 = _mm_setzero_ps();
  const __m128i one128 = _mm_set1_epi32(1);

  for (int64_t i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK_K;
    float amax = 0.0f;
    bool finite = true;
    for (int j = 0; j < QK_K; ++j)
    {
      const float v = xb[j];
      finite &= std::isfinite(v);
      amax = MAX(amax, fabsf(v));
    }
    if (!finite)
    {
      quantize_row_tq1_0_ref(xb, y + i, QK_K);
      continue;
    }

    const float d = amax;
    const float id = d ? 1.0f / d : 0.0f;
    const __m256 idv = _mm256_set1_ps(id);
    const __m128 idv128 = _mm_set1_ps(id);
    y[i].d = GGML_FP32_TO_FP16(d);

    for (size_t m = 0; m < 32; m += 8)
    {
      tq1_0_pack_8_avx2(y[i].qs + m, xb + m, 32, idv, sign_mask, half, zero, one);
    }
    for (size_t m = 0; m < 16; m += 8)
    {
      tq1_0_pack_8_avx2(y[i].qs + 32 + m, xb + 160 + m, 16, idv, sign_mask, half, zero, one);
    }
    tq1_0_pack_high4_avx2(y[i].qh, xb + 240, idv128, sign_mask128, half128, zero128, one128);
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
