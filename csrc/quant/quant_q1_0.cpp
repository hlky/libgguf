#include "libgguf_common.h"

void quantize_row_q1_0_ref(const float *RESTRICT x, block_q1_0 *RESTRICT y, int64_t k)
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

    y[i].d = GGML_FP32_TO_FP16(d);

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

// reference implementation for deterministic creation of model files

size_t quantize_q1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    quantize_row_q1_0_ref(src, (block_q1_0 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q1_0, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q1_0, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_q1_0_ref(src, (block_q1_0 *)qrow, n_per_row);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

