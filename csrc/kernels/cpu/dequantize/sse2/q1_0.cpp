#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <emmintrin.h>

static inline void libgguf_q1_0_store4(float *LIBGGUF_RESTRICT y, uint8_t byte, __m128 d, __m128 neg_d, __m128i masks)
{
  const __m128i bits = _mm_and_si128(_mm_set1_epi32(byte), masks);
  const __m128i zeros = _mm_cmpeq_epi32(bits, _mm_setzero_si128());
  const __m128 values = _mm_or_ps(_mm_andnot_ps(_mm_castsi128_ps(zeros), d), _mm_and_ps(_mm_castsi128_ps(zeros), neg_d));
  _mm_storeu_ps(y, values);
}

extern "C" void dequantize_block_q1_0(const block_q1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK1_0 == 0);
  static_assert(QK1_0 == 128, "QK1_0 must be 128");

  const int nb = (int)(k / QK1_0);
  const __m128i masks_lo = _mm_set_epi32(8, 4, 2, 1);
  const __m128i masks_hi = _mm_set_epi32(128, 64, 32, 16);

  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(LIBGGUF_FP16_TO_FP32(x[i].d));
    const __m128 neg_d = _mm_sub_ps(_mm_setzero_ps(), d);
    float *LIBGGUF_RESTRICT yb = y + i * QK1_0;

    for (int j = 0; j < QK1_0 / 8; ++j)
    {
      libgguf_q1_0_store4(yb + 8 * j + 0, x[i].qs[j], d, neg_d, masks_lo);
      libgguf_q1_0_store4(yb + 8 * j + 4, x[i].qs[j], d, neg_d, masks_hi);
    }
  }
}


void dequantize_q1_0(const block_q1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_q1_0(x, y, k);
}

