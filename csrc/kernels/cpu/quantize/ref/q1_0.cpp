#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


void quantize_block_q1_0(const float *LIBGGUF_RESTRICT x, block_q1_0 *LIBGGUF_RESTRICT y, int64_t k)
{
  static const int qk = QK1_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float sum_abs = 0.0f;
    for (int j = 0; j < qk; j++)
    {
      sum_abs += fabsf(x[i * qk + j]);
    }
    const float d = sum_abs / qk;

    y[i].d = LIBGGUF_FP32_TO_FP16(d);

    // Clear all bits first
    for (int j = 0; j < qk / 8; ++j)
    {
      y[i].qs[j] = 0;
    }

    // Just store sign of each weight directly (no normalization)
    for (int j = 0; j < qk; ++j)
    {
      const int bit_index = j;
      const int byte_index = bit_index / 8;
      const int bit_offset = bit_index % 8;

      if (x[i * qk + j] >= 0.0f)
      {
        y[i].qs[byte_index] |= (1 << bit_offset);
      }
    }
  }
}


size_t quantize_q1_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  LIBGGUF_UNUSED(quant_weights);
  quantize_block_q1_0(src, (block_q1_0 *)dst, nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_Q1_0, n_per_row);
}

