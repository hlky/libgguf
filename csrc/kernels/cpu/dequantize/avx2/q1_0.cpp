#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <immintrin.h>

extern "C" void dequantize_block_q1_0(const block_q1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK1_0 == 0);
  static_assert(QK1_0 == 128, "QK1_0 must be 128");

  const int nb = (int)(k / QK1_0);
  const __m256i masks = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

  for (int i = 0; i < nb; ++i)
  {
    const __m256 d = _mm256_set1_ps(LIBGGUF_FP16_TO_FP32(x[i].d));
    const __m256 neg_d = _mm256_sub_ps(_mm256_setzero_ps(), d);
    float *LIBGGUF_RESTRICT yb = y + i * QK1_0;

    for (int j = 0; j < QK1_0 / 8; ++j)
    {
      const __m256i bits = _mm256_and_si256(_mm256_set1_epi32(x[i].qs[j]), masks);
      const __m256 cmp = _mm256_castsi256_ps(_mm256_cmpeq_epi32(bits, _mm256_setzero_si256()));
      _mm256_storeu_ps(yb + 8 * j, _mm256_blendv_ps(d, neg_d, cmp));
    }
  }
}


void dequantize_q1_0(const block_q1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_q1_0(x, y, k);
}

