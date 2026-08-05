#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <smmintrin.h>

static inline __m128i tq2_0_round_away_from_zero_4(
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

static inline __m128i tq2_0_quantize_8_i16(
    const float *LIBGGUF_RESTRICT x,
    __m128 id,
    __m128 sign_mask,
    __m128 half,
    __m128 zero,
    __m128i one)
{
  const __m128i q0 = tq2_0_round_away_from_zero_4(_mm_loadu_ps(x), id, sign_mask, half, zero, one);
  const __m128i q1 = tq2_0_round_away_from_zero_4(_mm_loadu_ps(x + 4), id, sign_mask, half, zero, one);
  return _mm_packs_epi32(q0, q1);
}

extern "C" void quantize_block_tq2_0(const float *LIBGGUF_RESTRICT x, block_tq2_0 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");
  static_assert(sizeof(block_tq2_0::qs) == QK_K / 4, "unexpected TQ2_0 qs size");

  const int64_t nb = k / QK_K;
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128 zero = _mm_setzero_ps();
  const __m128i one = _mm_set1_epi32(1);
  const __m128i zero_i = _mm_setzero_si128();

  for (int64_t i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK_K;
    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j)
    {
      const float v = xb[j];
      amax = libgguf_max(amax, fabsf(v));
    }

    const float d = amax;
    const float id = d ? 1.0f / d : 0.0f;
    const __m128 idv = _mm_set1_ps(id);
    y[i].d = LIBGGUF_FP32_TO_FP16(d);

    for (size_t j = 0; j < sizeof(y->qs); j += 32)
    {
      const float *src = xb + j * 4;
      for (size_t m = 0; m < 32; m += 8)
      {
        const __m128i q0 = tq2_0_quantize_8_i16(src + m, idv, sign_mask, half, zero, one);
        const __m128i q1 = tq2_0_quantize_8_i16(src + m + 32, idv, sign_mask, half, zero, one);
        const __m128i q2 = tq2_0_quantize_8_i16(src + m + 64, idv, sign_mask, half, zero, one);
        const __m128i q3 = tq2_0_quantize_8_i16(src + m + 96, idv, sign_mask, half, zero, one);
        const __m128i packed_i16 = _mm_or_si128(
            _mm_or_si128(q0, _mm_slli_epi16(q1, 2)),
            _mm_or_si128(_mm_slli_epi16(q2, 4), _mm_slli_epi16(q3, 6)));
        const __m128i packed_u8 = _mm_packus_epi16(packed_i16, zero_i);
        _mm_storel_epi64((__m128i *)(y[i].qs + j + m), packed_u8);
      }
    }
  }
}


size_t quantize_tq2_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  const size_t row_size = libgguf_row_size(GGML_TYPE_TQ2_0, n_per_row);
  quantize_block_tq2_0(src, (block_tq2_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

