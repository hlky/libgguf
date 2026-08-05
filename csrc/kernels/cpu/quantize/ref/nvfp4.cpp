#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


void quantize_block_nvfp4(const float *LIBGGUF_RESTRICT x, block_nvfp4 *LIBGGUF_RESTRICT y, int64_t k)
{
  static const int qk = QK_NVFP4;
  static const int qk_sub = QK_NVFP4_SUB;
  static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    for (int s = 0; s < n_sub; s++)
    {
      const float *xb = x + i * qk + s * qk_sub;

      float amax = 0.0f;
      for (int j = 0; j < qk_sub; j++)
      {
        if (amax < fabsf(xb[j]))
        {
          amax = fabsf(xb[j]);
        }
      }

      // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
      const uint8_t ue = libgguf_fp32_to_ue4m3(amax / 6.0f);
      y[i].d[s] = ue;
      const float d = libgguf_ue4m3_to_fp32(ue);

      for (int j = 0; j < qk_sub / 2; ++j)
      {
        const uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
        const uint8_t x1 = best_index_mxfp4(xb[qk_sub / 2 + j], d);

        y[i].qs[s * (qk_sub / 2) + j] = x0 | (x1 << 4);
      }
    }
  }
}


size_t quantize_nvfp4(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  LIBGGUF_UNUSED(quant_weights);
  quantize_block_nvfp4(src, (block_nvfp4 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_NVFP4, n_per_row);
}

