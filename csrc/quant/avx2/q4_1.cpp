#include "libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_AVX2)
static inline __m256i q4_1_quantize_8_avx2(const float *RESTRICT x, __m256 minv, __m256 id, __m256 half, __m256i max_q)
{
  const __m256 scaled = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x), minv), id), half);
  return _mm256_min_epi32(_mm256_cvttps_epi32(scaled), max_q);
}
#endif

extern "C" void quantize_row_q4_1_avx2(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK4_1 == 0);
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  const int nb = (int)(k / QK4_1);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256i max_q = _mm256_set1_epi32(15);
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

    const __m256 minv = _mm256_set1_ps(min);
    const __m256 idv = _mm256_set1_ps(id);
    for (int j = 0; j < QK4_1 / 2; j += 8)
    {
      alignas(32) int32_t lo[8];
      alignas(32) int32_t hi[8];
      _mm256_store_si256((__m256i *)lo, q4_1_quantize_8_avx2(xb + j, minv, idv, half, max_q));
      _mm256_store_si256((__m256i *)hi, q4_1_quantize_8_avx2(xb + QK4_1 / 2 + j, minv, idv, half, max_q));
      for (int lane = 0; lane < 8; ++lane)
      {
        y[i].qs[j + lane] = (uint8_t)lo[lane] | (uint8_t)(hi[lane] << 4);
      }
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
