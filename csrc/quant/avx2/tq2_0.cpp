#include "libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#include <cstring>
#define LIBGGUF_BUILD_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_AVX2)
static inline __m256i tq2_0_round_away_from_zero_8_avx2(
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

static inline uint64_t tq2_0_pack_8_u8_avx2(__m256i v)
{
  const __m256i packed_i16 = _mm256_packs_epi32(v, _mm256_setzero_si256());
  const __m256i packed_u8 = _mm256_packus_epi16(packed_i16, _mm256_setzero_si256());
  const uint32_t lo = (uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(packed_u8));
  const uint32_t hi = (uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(packed_u8, 1));
  return (uint64_t)lo | ((uint64_t)hi << 32);
}
#endif

extern "C" void quantize_row_tq2_0_avx2(const float *RESTRICT x, block_tq2_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");
  static_assert(sizeof(block_tq2_0::qs) == QK_K / 4, "unexpected TQ2_0 qs size");

  const int64_t nb = k / QK_K;
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256i one = _mm256_set1_epi32(1);

  for (int64_t i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK_K;
    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j)
    {
      const float v = xb[j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax;
    const float id = d ? 1.0f / d : 0.0f;
    const __m256 idv = _mm256_set1_ps(id);
    y[i].d = GGML_FP32_TO_FP16(d);

    for (size_t j = 0; j < sizeof(y->qs); j += 32)
    {
      const float *src = xb + j * 4;
      for (size_t m = 0; m < 32; m += 8)
      {
        const __m256i q0 = tq2_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + m), idv, sign_mask, half, zero, one);
        const __m256i q1 = tq2_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + m + 32), idv, sign_mask, half, zero, one);
        const __m256i q2 = tq2_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + m + 64), idv, sign_mask, half, zero, one);
        const __m256i q3 = tq2_0_round_away_from_zero_8_avx2(_mm256_loadu_ps(src + m + 96), idv, sign_mask, half, zero, one);
        const __m256i packed_i32 = _mm256_or_si256(
            _mm256_or_si256(q0, _mm256_slli_epi32(q1, 2)),
            _mm256_or_si256(_mm256_slli_epi32(q2, 4), _mm256_slli_epi32(q3, 6)));
        const uint64_t packed = tq2_0_pack_8_u8_avx2(packed_i32);
        std::memcpy(y[i].qs + j + m, &packed, sizeof(packed));
      }
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
