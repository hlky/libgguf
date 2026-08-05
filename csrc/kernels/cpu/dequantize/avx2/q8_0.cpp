#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <immintrin.h>

extern "C" void dequantize_block_q8_0(const block_q8_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  static_assert(QK8_0 == 32, "QK8_0 must be 32");

  const int nb = k / QK8_0;
  for (int i = 0; i < nb; ++i)
  {
    const __m256 scale = _mm256_set1_ps(LIBGGUF_FP16_TO_FP32(x[i].d));
    float *yb = y + i * QK8_0;

    const __m256i q0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x[i].qs + 0)));
    const __m256i q1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x[i].qs + 8)));
    const __m256i q2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x[i].qs + 16)));
    const __m256i q3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x[i].qs + 24)));

    _mm256_storeu_ps(yb + 0, _mm256_mul_ps(_mm256_cvtepi32_ps(q0), scale));
    _mm256_storeu_ps(yb + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(q1), scale));
    _mm256_storeu_ps(yb + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(q2), scale));
    _mm256_storeu_ps(yb + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(q3), scale));
  }
}


void dequantize_q8_0(const block_q8_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_q8_0(x, y, k);
}

