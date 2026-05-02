#include "libgguf_common.h"
#include "libgguf_tables.h"

void quantize_row_mxfp4_ref(const float *RESTRICT x, block_mxfp4 *RESTRICT y, int64_t k)
{
  static const int qk = QK_MXFP4;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < qk; j++)
    {
      const float v = x[i * qk + j];

      if (amax < fabsf(v))
      {
        amax = fabsf(v);
      }
    }

    const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2 + 127) : 0;

    const float d = GGML_E8M0_TO_FP32_HALF(e);

    y[i].e = e;

    for (int j = 0; j < qk / 2; ++j)
    {
      const uint8_t x0 = best_index_mxfp4(x[i * qk + 0 + j], d);
      const uint8_t x1 = best_index_mxfp4(x[i * qk + qk / 2 + j], d);

      y[i].qs[j] = x0;
      y[i].qs[j] |= x1 << 4;
    }
  }
}


size_t quantize_mxfp4(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  GGML_UNUSED(quant_weights);
  quantize_row_mxfp4_ref(src, (block_mxfp4 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_MXFP4, n_per_row);
}

