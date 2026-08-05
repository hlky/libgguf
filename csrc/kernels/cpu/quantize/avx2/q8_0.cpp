#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <immintrin.h>

static inline float q8_0_hmax_ps(__m256 v)
{
  __m128 maxv = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  __m128 shuf = _mm_shuffle_ps(maxv, maxv, _MM_SHUFFLE(2, 3, 0, 1));
  maxv = _mm_max_ps(maxv, shuf);
  shuf = _mm_shuffle_ps(maxv, maxv, _MM_SHUFFLE(1, 0, 3, 2));
  maxv = _mm_max_ps(maxv, shuf);
  return _mm_cvtss_f32(maxv);
}

static inline __m256i q8_0_round_away_from_zero(
    __m256 v,
    __m256 id,
    __m256 sign_mask,
    __m256 half,
    __m256 zero)
{
  const __m256 scaled = _mm256_mul_ps(v, id);
  const __m256 abs_scaled = _mm256_andnot_ps(sign_mask, scaled);
  const __m256 rounded_abs = _mm256_add_ps(abs_scaled, half);
  const __m256i abs_i = _mm256_cvttps_epi32(rounded_abs);
  const __m256 negative = _mm256_cmp_ps(scaled, zero, _CMP_LT_OQ);
  const __m256i sign_i = _mm256_castps_si256(negative);
  return _mm256_sub_epi32(_mm256_xor_si256(abs_i, sign_i), sign_i);
}

extern "C" void quantize_block_q8_0(const float *LIBGGUF_RESTRICT x, block_q8_0 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  static_assert(QK8_0 == 32, "QK8_0 must be 32");

  const int nb = k / QK8_0;
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256i pack_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK8_0;
    __m256 maxv = _mm256_setzero_ps();
    for (int j = 0; j < QK8_0; j += 8)
    {
      const __m256 v = _mm256_loadu_ps(xb + j);
      maxv = _mm256_max_ps(maxv, _mm256_andnot_ps(sign_mask, v));
    }

    const float amax = q8_0_hmax_ps(maxv);
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    const __m256 idv = _mm256_set1_ps(id);

    y[i].d = LIBGGUF_FP32_TO_FP16(d);

    const __m256i q0 = q8_0_round_away_from_zero(_mm256_loadu_ps(xb + 0), idv, sign_mask, half, zero);
    const __m256i q1 = q8_0_round_away_from_zero(_mm256_loadu_ps(xb + 8), idv, sign_mask, half, zero);
    const __m256i q2 = q8_0_round_away_from_zero(_mm256_loadu_ps(xb + 16), idv, sign_mask, half, zero);
    const __m256i q3 = q8_0_round_away_from_zero(_mm256_loadu_ps(xb + 24), idv, sign_mask, half, zero);

    const __m256i q01_i16 = _mm256_packs_epi32(q0, q1);
    const __m256i q23_i16 = _mm256_packs_epi32(q2, q3);
    const __m256i packed_lanes = _mm256_packs_epi16(q01_i16, q23_i16);
    const __m256i packed = _mm256_permutevar8x32_epi32(packed_lanes, pack_order);
    _mm256_storeu_si256((__m256i *)y[i].qs, packed);
  }
}


size_t quantize_q8_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  quantize_block_q8_0(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
}

