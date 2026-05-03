#include "common/libgguf_common.h"

#if defined(_MSC_VER)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

extern "C" void dequantize_row_q4_1_sse4_1(const block_q4_1 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK4_1 == 0);
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  const int nb = (int)(k / QK4_1);
  const __m128i low_mask = _mm_set1_epi8(0x0f);

  for (int i = 0; i < nb; ++i)
  {
    const __m128 scale = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    const __m128 minv = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].m));
    const __m128i packed = _mm_loadu_si128((const __m128i *)x[i].qs);
    const __m128i lo8 = _mm_and_si128(packed, low_mask);
    const __m128i hi8 = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);

    const __m128i lo16a = _mm_cvtepu8_epi16(lo8);
    const __m128i lo16b = _mm_cvtepu8_epi16(_mm_srli_si128(lo8, 8));
    const __m128i hi16a = _mm_cvtepu8_epi16(hi8);
    const __m128i hi16b = _mm_cvtepu8_epi16(_mm_srli_si128(hi8, 8));

    const __m128i lo32a = _mm_cvtepi16_epi32(lo16a);
    const __m128i lo32b = _mm_cvtepi16_epi32(_mm_srli_si128(lo16a, 8));
    const __m128i lo32c = _mm_cvtepi16_epi32(lo16b);
    const __m128i lo32d = _mm_cvtepi16_epi32(_mm_srli_si128(lo16b, 8));
    const __m128i hi32a = _mm_cvtepi16_epi32(hi16a);
    const __m128i hi32b = _mm_cvtepi16_epi32(_mm_srli_si128(hi16a, 8));
    const __m128i hi32c = _mm_cvtepi16_epi32(hi16b);
    const __m128i hi32d = _mm_cvtepi16_epi32(_mm_srli_si128(hi16b, 8));

    float *yb = y + i * QK4_1;
    _mm_storeu_ps(yb + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(lo32a), scale), minv));
    _mm_storeu_ps(yb + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(lo32b), scale), minv));
    _mm_storeu_ps(yb + 8, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(lo32c), scale), minv));
    _mm_storeu_ps(yb + 12, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(lo32d), scale), minv));
    _mm_storeu_ps(yb + 16, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(hi32a), scale), minv));
    _mm_storeu_ps(yb + 20, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(hi32b), scale), minv));
    _mm_storeu_ps(yb + 24, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(hi32c), scale), minv));
    _mm_storeu_ps(yb + 28, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(hi32d), scale), minv));
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
