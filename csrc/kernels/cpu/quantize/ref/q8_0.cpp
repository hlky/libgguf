#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


void quantize_block_q8_0(const float *LIBGGUF_RESTRICT x, block_q8_0 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++)
    {
      const float v = x[i * QK8_0 + j];
      amax = libgguf_max(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = LIBGGUF_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j)
    {
      const float x0 = x[i * QK8_0 + j] * id;
      y[i].qs[j] = roundf(x0);
    }
  }
}


size_t quantize_q8_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  quantize_block_q8_0(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_Q8_0, n_per_row);
}

