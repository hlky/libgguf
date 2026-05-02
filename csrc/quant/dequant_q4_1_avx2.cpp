#include "common/libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

extern "C" void dequantize_row_q4_1_avx2(const block_q4_1 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK4_1 == 0);
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  const int nb = (int)(k / QK4_1);
  const __m128i low_mask = _mm_set1_epi8(0x0f);

  for (int i = 0; i < nb; ++i)
  {
    const __m256 scale = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    const __m256 minv = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].m));
    const __m128i packed = _mm_loadu_si128((const __m128i *)x[i].qs);
    const __m128i lo8 = _mm_and_si128(packed, low_mask);
    const __m128i hi8 = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
    const __m256i lo16 = _mm256_cvtepu8_epi16(lo8);
    const __m256i hi16 = _mm256_cvtepu8_epi16(hi8);

    const __m256i lo32a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo16));
    const __m256i lo32b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo16, 1));
    const __m256i hi32a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi16));
    const __m256i hi32b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi16, 1));

    float *yb = y + i * QK4_1;
    _mm256_storeu_ps(yb + 0, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo32a), scale), minv));
    _mm256_storeu_ps(yb + 8, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo32b), scale), minv));
    _mm256_storeu_ps(yb + 16, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi32a), scale), minv));
    _mm256_storeu_ps(yb + 24, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi32b), scale), minv));
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
