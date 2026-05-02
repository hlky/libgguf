#include "common/libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

extern "C" void dequantize_row_q4_0_sse2(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK4_0 == 0);
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  const int nb = k / QK4_0;
  const __m128i low_mask = _mm_set1_epi8(0x0f);
  const __m128i zero = _mm_setzero_si128();
  const __m128i eight = _mm_set1_epi16(8);

  for (int i = 0; i < nb; ++i)
  {
    const __m128 scale = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    const __m128i packed = _mm_loadu_si128((const __m128i *)x[i].qs);
    const __m128i lo8 = _mm_and_si128(packed, low_mask);
    const __m128i hi8 = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);

    const __m128i lo16a = _mm_sub_epi16(_mm_unpacklo_epi8(lo8, zero), eight);
    const __m128i lo16b = _mm_sub_epi16(_mm_unpackhi_epi8(lo8, zero), eight);
    const __m128i hi16a = _mm_sub_epi16(_mm_unpacklo_epi8(hi8, zero), eight);
    const __m128i hi16b = _mm_sub_epi16(_mm_unpackhi_epi8(hi8, zero), eight);

    const __m128i lo32a = _mm_unpacklo_epi16(lo16a, _mm_cmpgt_epi16(zero, lo16a));
    const __m128i lo32b = _mm_unpackhi_epi16(lo16a, _mm_cmpgt_epi16(zero, lo16a));
    const __m128i lo32c = _mm_unpacklo_epi16(lo16b, _mm_cmpgt_epi16(zero, lo16b));
    const __m128i lo32d = _mm_unpackhi_epi16(lo16b, _mm_cmpgt_epi16(zero, lo16b));
    const __m128i hi32a = _mm_unpacklo_epi16(hi16a, _mm_cmpgt_epi16(zero, hi16a));
    const __m128i hi32b = _mm_unpackhi_epi16(hi16a, _mm_cmpgt_epi16(zero, hi16a));
    const __m128i hi32c = _mm_unpacklo_epi16(hi16b, _mm_cmpgt_epi16(zero, hi16b));
    const __m128i hi32d = _mm_unpackhi_epi16(hi16b, _mm_cmpgt_epi16(zero, hi16b));

    float *yb = y + i * QK4_0;
    _mm_storeu_ps(yb + 0, _mm_mul_ps(_mm_cvtepi32_ps(lo32a), scale));
    _mm_storeu_ps(yb + 4, _mm_mul_ps(_mm_cvtepi32_ps(lo32b), scale));
    _mm_storeu_ps(yb + 8, _mm_mul_ps(_mm_cvtepi32_ps(lo32c), scale));
    _mm_storeu_ps(yb + 12, _mm_mul_ps(_mm_cvtepi32_ps(lo32d), scale));
    _mm_storeu_ps(yb + 16, _mm_mul_ps(_mm_cvtepi32_ps(hi32a), scale));
    _mm_storeu_ps(yb + 20, _mm_mul_ps(_mm_cvtepi32_ps(hi32b), scale));
    _mm_storeu_ps(yb + 24, _mm_mul_ps(_mm_cvtepi32_ps(hi32c), scale));
    _mm_storeu_ps(yb + 28, _mm_mul_ps(_mm_cvtepi32_ps(hi32d), scale));
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}

