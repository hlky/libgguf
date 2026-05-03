#include "common/libgguf_common.h"

#if defined(_MSC_VER)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

static inline void libgguf_q1_0_store4_sse4_1(float *RESTRICT y, uint8_t byte, __m128 d, __m128 neg_d, __m128i masks)
{
  const __m128i bits = _mm_and_si128(_mm_set1_epi32(byte), masks);
  const __m128i zeros = _mm_cmpeq_epi32(bits, _mm_setzero_si128());
  const __m128 values = _mm_blendv_ps(d, neg_d, _mm_castsi128_ps(zeros));
  _mm_storeu_ps(y, values);
}

extern "C" void dequantize_row_q1_0_sse4_1(const block_q1_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK1_0 == 0);
  static_assert(QK1_0 == 128, "QK1_0 must be 128");

  const int nb = (int)(k / QK1_0);
  const __m128i masks_lo = _mm_set_epi32(8, 4, 2, 1);
  const __m128i masks_hi = _mm_set_epi32(128, 64, 32, 16);

  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    const __m128 neg_d = _mm_sub_ps(_mm_setzero_ps(), d);
    float *RESTRICT yb = y + i * QK1_0;

    for (int j = 0; j < QK1_0 / 8; ++j)
    {
      libgguf_q1_0_store4_sse4_1(yb + 8 * j + 0, x[i].qs[j], d, neg_d, masks_lo);
      libgguf_q1_0_store4_sse4_1(yb + 8 * j + 4, x[i].qs[j], d, neg_d, masks_hi);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
