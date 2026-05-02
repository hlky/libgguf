#include "common/libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

extern "C" void dequantize_row_q8_0_sse2(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK8_0 == 0);
  static_assert(QK8_0 == 32, "QK8_0 must be 32");

  const int nb = k / QK8_0;
  const __m128i zero = _mm_setzero_si128();
  for (int i = 0; i < nb; ++i)
  {
    const __m128 scale = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    const __m128i q0 = _mm_loadu_si128((const __m128i *)(x[i].qs + 0));
    const __m128i q1 = _mm_loadu_si128((const __m128i *)(x[i].qs + 16));
    const __m128i q0sign = _mm_cmpgt_epi8(zero, q0);
    const __m128i q1sign = _mm_cmpgt_epi8(zero, q1);

    const __m128i q16a = _mm_unpacklo_epi8(q0, q0sign);
    const __m128i q16b = _mm_unpackhi_epi8(q0, q0sign);
    const __m128i q16c = _mm_unpacklo_epi8(q1, q1sign);
    const __m128i q16d = _mm_unpackhi_epi8(q1, q1sign);

    const __m128i q32a = _mm_unpacklo_epi16(q16a, _mm_cmpgt_epi16(zero, q16a));
    const __m128i q32b = _mm_unpackhi_epi16(q16a, _mm_cmpgt_epi16(zero, q16a));
    const __m128i q32c = _mm_unpacklo_epi16(q16b, _mm_cmpgt_epi16(zero, q16b));
    const __m128i q32d = _mm_unpackhi_epi16(q16b, _mm_cmpgt_epi16(zero, q16b));
    const __m128i q32e = _mm_unpacklo_epi16(q16c, _mm_cmpgt_epi16(zero, q16c));
    const __m128i q32f = _mm_unpackhi_epi16(q16c, _mm_cmpgt_epi16(zero, q16c));
    const __m128i q32g = _mm_unpacklo_epi16(q16d, _mm_cmpgt_epi16(zero, q16d));
    const __m128i q32h = _mm_unpackhi_epi16(q16d, _mm_cmpgt_epi16(zero, q16d));

    float *yb = y + i * QK8_0;
    _mm_storeu_ps(yb + 0, _mm_mul_ps(_mm_cvtepi32_ps(q32a), scale));
    _mm_storeu_ps(yb + 4, _mm_mul_ps(_mm_cvtepi32_ps(q32b), scale));
    _mm_storeu_ps(yb + 8, _mm_mul_ps(_mm_cvtepi32_ps(q32c), scale));
    _mm_storeu_ps(yb + 12, _mm_mul_ps(_mm_cvtepi32_ps(q32d), scale));
    _mm_storeu_ps(yb + 16, _mm_mul_ps(_mm_cvtepi32_ps(q32e), scale));
    _mm_storeu_ps(yb + 20, _mm_mul_ps(_mm_cvtepi32_ps(q32f), scale));
    _mm_storeu_ps(yb + 24, _mm_mul_ps(_mm_cvtepi32_ps(q32g), scale));
    _mm_storeu_ps(yb + 28, _mm_mul_ps(_mm_cvtepi32_ps(q32h), scale));
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}

