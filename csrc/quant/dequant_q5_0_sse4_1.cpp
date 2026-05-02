#include "common/libgguf_common.h"

#if defined(_MSC_VER)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

extern "C" void dequantize_row_q5_0_sse4_1(const block_q5_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK5_0 == 0);
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  const int nb = (int)(k / QK5_0);
  const __m128i low_mask = _mm_set1_epi8(0x0f);
  const __m128i sixteen = _mm_set1_epi16(16);

  for (int i = 0; i < nb; ++i)
  {
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    uint8_t high_lo[16];
    uint8_t high_hi[16];
    for (int j = 0; j < 16; ++j)
    {
      high_lo[j] = (uint8_t)((((qh >> j) << 4) & 0x10));
      high_hi[j] = (uint8_t)(((qh >> (j + 12)) & 0x10));
    }

    const __m128 scale = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    const __m128i packed = _mm_loadu_si128((const __m128i *)x[i].qs);
    const __m128i lo8 = _mm_or_si128(_mm_and_si128(packed, low_mask), _mm_loadu_si128((const __m128i *)high_lo));
    const __m128i hi8 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(packed, 4), low_mask), _mm_loadu_si128((const __m128i *)high_hi));

    const __m128i lo16a = _mm_sub_epi16(_mm_cvtepu8_epi16(lo8), sixteen);
    const __m128i lo16b = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(lo8, 8)), sixteen);
    const __m128i hi16a = _mm_sub_epi16(_mm_cvtepu8_epi16(hi8), sixteen);
    const __m128i hi16b = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(hi8, 8)), sixteen);

    const __m128i lo32a = _mm_cvtepi16_epi32(lo16a);
    const __m128i lo32b = _mm_cvtepi16_epi32(_mm_srli_si128(lo16a, 8));
    const __m128i lo32c = _mm_cvtepi16_epi32(lo16b);
    const __m128i lo32d = _mm_cvtepi16_epi32(_mm_srli_si128(lo16b, 8));
    const __m128i hi32a = _mm_cvtepi16_epi32(hi16a);
    const __m128i hi32b = _mm_cvtepi16_epi32(_mm_srli_si128(hi16a, 8));
    const __m128i hi32c = _mm_cvtepi16_epi32(hi16b);
    const __m128i hi32d = _mm_cvtepi16_epi32(_mm_srli_si128(hi16b, 8));

    float *yb = y + i * QK5_0;
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
