#include "libgguf_common.h"

#if defined(__SSE4_1__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

#if defined(LIBGGUF_BUILD_SSE4_1)
static inline __m128i q4_1_quantize_4_i32_sse4_1(const float *RESTRICT x, __m128 minv, __m128 id, __m128 half, __m128i max_q)
{
  const __m128 scaled = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(x), minv), id), half);
  return _mm_min_epi32(_mm_cvttps_epi32(scaled), max_q);
}
#endif

extern "C" void quantize_row_q4_1_sse4_1(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK4_1 == 0);
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  const int nb = (int)(k / QK4_1);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128i max_q = _mm_set1_epi32(15);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK4_1;
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int j = 0; j < QK4_1; ++j)
    {
      const float v = xb[j];
      if (v < min)
        min = v;
      if (v > max)
        max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = GGML_FP32_TO_FP16(d);
    y[i].m = GGML_FP32_TO_FP16(min);

    const __m128 minv = _mm_set1_ps(min);
    const __m128 idv = _mm_set1_ps(id);
    for (int j = 0; j < QK4_1 / 2; j += 8)
    {
      const __m128i lo0 = q4_1_quantize_4_i32_sse4_1(xb + j, minv, idv, half, max_q);
      const __m128i lo1 = q4_1_quantize_4_i32_sse4_1(xb + j + 4, minv, idv, half, max_q);
      const __m128i hi0 = q4_1_quantize_4_i32_sse4_1(xb + QK4_1 / 2 + j, minv, idv, half, max_q);
      const __m128i hi1 = q4_1_quantize_4_i32_sse4_1(xb + QK4_1 / 2 + j + 4, minv, idv, half, max_q);
      const __m128i lo = _mm_packs_epi32(lo0, lo1);
      const __m128i hi = _mm_packs_epi32(hi0, hi1);
      const __m128i packed = _mm_or_si128(lo, _mm_slli_epi16(hi, 4));
      const __m128i bytes = _mm_packus_epi16(packed, _mm_setzero_si128());
      _mm_storel_epi64((__m128i *)(y[i].qs + j), bytes);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
