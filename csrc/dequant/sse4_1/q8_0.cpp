#include "common/libgguf_common.h"

#if defined(_MSC_VER)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

extern "C" void dequantize_row_q8_0_sse4_1(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK8_0 == 0);
  static_assert(QK8_0 == 32, "QK8_0 must be 32");

  const int nb = k / QK8_0;
  for (int i = 0; i < nb; ++i)
  {
    const __m128 scale = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    const __m128i q0 = _mm_loadu_si128((const __m128i *)(x[i].qs + 0));
    const __m128i q1 = _mm_loadu_si128((const __m128i *)(x[i].qs + 16));
    const __m128i q32a = _mm_cvtepi8_epi32(q0);
    const __m128i q32b = _mm_cvtepi8_epi32(_mm_srli_si128(q0, 4));
    const __m128i q32c = _mm_cvtepi8_epi32(_mm_srli_si128(q0, 8));
    const __m128i q32d = _mm_cvtepi8_epi32(_mm_srli_si128(q0, 12));
    const __m128i q32e = _mm_cvtepi8_epi32(q1);
    const __m128i q32f = _mm_cvtepi8_epi32(_mm_srli_si128(q1, 4));
    const __m128i q32g = _mm_cvtepi8_epi32(_mm_srli_si128(q1, 8));
    const __m128i q32h = _mm_cvtepi8_epi32(_mm_srli_si128(q1, 12));

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
