#include "libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_AVX2)
static inline __m256i q5_1_quantize_8_avx2(const float *RESTRICT x, __m256 minv, __m256 id, __m256 half)
{
  return _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x), minv), id), half));
}
#endif

extern "C" void quantize_row_q5_1_avx2(const float *RESTRICT x, block_q5_1 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK5_1 == 0);
  static_assert(QK5_1 == 32, "QK5_1 must be 32");

  const int nb = (int)(k / QK5_1);
  const __m256 half = _mm256_set1_ps(0.5f);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK5_1;
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int j = 0; j < QK5_1; ++j)
    {
      const float v = xb[j];
      if (v < min)
        min = v;
      if (v > max)
        max = v;
    }

    const float d = (max - min) / ((1 << 5) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = GGML_FP32_TO_FP16(d);
    y[i].m = GGML_FP32_TO_FP16(min);

    const __m256 minv = _mm256_set1_ps(min);
    const __m256 idv = _mm256_set1_ps(id);
    uint32_t qh = 0;
    for (int j = 0; j < QK5_1 / 2; j += 8)
    {
      alignas(32) int32_t lo[8];
      alignas(32) int32_t hi[8];
      _mm256_store_si256((__m256i *)lo, q5_1_quantize_8_avx2(xb + j, minv, idv, half));
      _mm256_store_si256((__m256i *)hi, q5_1_quantize_8_avx2(xb + QK5_1 / 2 + j, minv, idv, half));
      for (int lane = 0; lane < 8; ++lane)
      {
        const uint8_t xi0 = (uint8_t)lo[lane];
        const uint8_t xi1 = (uint8_t)hi[lane];
        y[i].qs[j + lane] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + lane);
        qh |= ((xi1 & 0x10u) >> 4) << (j + lane + QK5_1 / 2);
      }
    }
    memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
