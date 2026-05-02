#include "libgguf_common.h"

void quantize_row_tq1_0_ref(const float *RESTRICT x, block_tq1_0 *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int64_t i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK_K; j++)
    {
      const float v = x[j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    // 5 elements per byte, along 32 bytes
    for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32)
    {
      for (size_t m = 0; m < 32; ++m)
      {
        uint8_t q = 0;
        for (size_t n = 0; n < 5; ++n)
        {
          int xi = lroundf(x[m + n * 32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
          q *= 3;
          q += xi;
        }
        // ceiling division (243 == pow(3, 5))
        q = ((uint16_t)q * 256 + (243 - 1)) / 243;
        y[i].qs[j + m] = q;
      }
      x += 5 * 32;
    }
    // along 16 bytes
    for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16)
    {
      for (size_t m = 0; m < 16; ++m)
      {
        uint8_t q = 0;
        for (size_t n = 0; n < 5; ++n)
        {
          int xi = lroundf(x[m + n * 16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
          q *= 3;
          q += xi;
        }
        // ceiling division (243 == pow(3, 5))
        q = ((uint16_t)q * 256 + (243 - 1)) / 243;
        y[i].qs[j + m] = q;
      }
      x += 5 * 16;
    }
    // 4 elements per byte
    for (size_t j = 0; j < sizeof(y->qh); ++j)
    {
      uint8_t q = 0;
      for (size_t m = 0; m < 4; ++m)
      {
        // -1, 0, 1 -> 0, 1, 2
        int xi = lroundf(x[j + m * sizeof(y->qh)] * id) + 1;
        q *= 3;
        q += xi;
      }
      // shift the first value to the most significant trit
      q *= 3;
      // ceiling division (243 == pow(3, 5))
      q = ((uint16_t)q * 256 + (243 - 1)) / 243;
      y[i].qh[j] = q;
    }
    x += 4 * sizeof(y->qh);
  }
}

size_t quantize_tq1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  (void)quant_weights; // not used
  const size_t row_size = libgguf_row_size(GGML_TYPE_TQ1_0, n_per_row);
  quantize_row_tq1_0_ref(src, (block_tq1_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

