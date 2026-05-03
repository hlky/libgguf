#include "libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

#if defined(LIBGGUF_BUILD_SSE2)
static inline float q8_0_hmax_ps_sse2(__m128 v)
{
  __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
  v = _mm_max_ps(v, shuf);
  shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
  v = _mm_max_ps(v, shuf);
  return _mm_cvtss_f32(v);
}

static inline __m128i q8_0_round_away_from_zero_sse2(
    __m128 v,
    __m128 id,
    __m128 sign_mask,
    __m128 half,
    __m128 zero)
{
  const __m128 scaled = _mm_mul_ps(v, id);
  const __m128 abs_scaled = _mm_andnot_ps(sign_mask, scaled);
  const __m128 rounded_abs = _mm_add_ps(abs_scaled, half);
  const __m128i abs_i = _mm_cvttps_epi32(rounded_abs);
  const __m128 negative = _mm_cmplt_ps(scaled, zero);
  const __m128i sign_i = _mm_castps_si128(negative);
  return _mm_sub_epi32(_mm_xor_si128(abs_i, sign_i), sign_i);
}
#endif

extern "C" void quantize_row_q8_0_sse2(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK8_0 == 0);
  static_assert(QK8_0 == 32, "QK8_0 must be 32");

  const int nb = k / QK8_0;
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128 zero = _mm_setzero_ps();
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK8_0;
    __m128 maxv = _mm_setzero_ps();
    for (int j = 0; j < QK8_0; j += 4)
    {
      const __m128 v = _mm_loadu_ps(xb + j);
      maxv = _mm_max_ps(maxv, _mm_andnot_ps(sign_mask, v));
    }

    const float amax = q8_0_hmax_ps_sse2(maxv);
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    const __m128 idv = _mm_set1_ps(id);

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; j += 8)
    {
      const __m128i q0 = q8_0_round_away_from_zero_sse2(_mm_loadu_ps(xb + j), idv, sign_mask, half, zero);
      const __m128i q1 = q8_0_round_away_from_zero_sse2(_mm_loadu_ps(xb + j + 4), idv, sign_mask, half, zero);
      const __m128i packed_i16 = _mm_packs_epi32(q0, q1);
      const __m128i packed_i8 = _mm_packs_epi16(packed_i16, _mm_setzero_si128());
      _mm_storel_epi64((__m128i *)(y[i].qs + j), packed_i8);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
