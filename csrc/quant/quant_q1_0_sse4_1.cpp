#include "libgguf_common.h"

#if defined(__SSE4_1__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

extern "C" void quantize_row_q1_0_sse4_1(const float *RESTRICT x, block_q1_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK1_0 == 0);
  static_assert(QK1_0 == 128, "QK1_0 must be 128");

  const int nb = (int)(k / QK1_0);
  const __m128 zero = _mm_setzero_ps();
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK1_0;

    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; ++j)
    {
      sum_abs += fabsf(xb[j]);
    }
    y[i].d = GGML_FP32_TO_FP16(sum_abs / QK1_0);

    for (int j = 0; j < QK1_0; j += 8)
    {
      const int lo = _mm_movemask_ps(_mm_cmpge_ps(_mm_loadu_ps(xb + j), zero));
      const int hi = _mm_movemask_ps(_mm_cmpge_ps(_mm_loadu_ps(xb + j + 4), zero));
      y[i].qs[j / 8] = (uint8_t)(lo | (hi << 4));
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
