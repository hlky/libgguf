#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <xmmintrin.h>

extern "C" void quantize_block_q1_0(const float *LIBGGUF_RESTRICT x, block_q1_0 *LIBGGUF_RESTRICT y, int64_t k)
{
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
    y[i].d = LIBGGUF_FP32_TO_FP16(sum_abs / QK1_0);

    for (int j = 0; j < QK1_0; j += 8)
    {
      const int lo = _mm_movemask_ps(_mm_cmpge_ps(_mm_loadu_ps(xb + j), zero));
      const int hi = _mm_movemask_ps(_mm_cmpge_ps(_mm_loadu_ps(xb + j + 4), zero));
      y[i].qs[j / 8] = (uint8_t)(lo | (hi << 4));
    }
  }
}


size_t quantize_q1_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  LIBGGUF_UNUSED(quant_weights);
  quantize_block_q1_0(src, (block_q1_0 *)dst, nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_Q1_0, n_per_row);
}

