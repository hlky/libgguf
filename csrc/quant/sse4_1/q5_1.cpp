#include "libgguf_common.h"

#if defined(__SSE4_1__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

#if defined(LIBGGUF_BUILD_SSE4_1)
static inline __m128i q5_1_quantize_4_i32_sse4_1(const float *RESTRICT x, __m128 minv, __m128 id, __m128 half)
{
  return _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(x), minv), id), half));
}
#endif

extern "C" void quantize_row_q5_1_sse4_1(const float *RESTRICT x, block_q5_1 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK5_1 == 0);
  static_assert(QK5_1 == 32, "QK5_1 must be 32");

  const int nb = (int)(k / QK5_1);
  const __m128 half = _mm_set1_ps(0.5f);
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

    const __m128 minv = _mm_set1_ps(min);
    const __m128 idv = _mm_set1_ps(id);
    uint32_t qh = 0;
    for (int j = 0; j < QK5_1 / 2; j += 4)
    {
      alignas(16) int32_t lo[4];
      alignas(16) int32_t hi[4];
      _mm_store_si128((__m128i *)lo, q5_1_quantize_4_i32_sse4_1(xb + j, minv, idv, half));
      _mm_store_si128((__m128i *)hi, q5_1_quantize_4_i32_sse4_1(xb + QK5_1 / 2 + j, minv, idv, half));
      for (int lane = 0; lane < 4; ++lane)
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
