#include "libgguf_common.h"

void quantize_row_q5_1_ref(const float *RESTRICT x, block_q5_1 *RESTRICT y, int64_t k)
{
  const int qk = QK5_1;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < qk; j++)
    {
      const float v = x[i * qk + j];

      if (v < min)
        min = v;
      if (v > max)
        max = v;
    }

    const float d = (max - min) / ((1 << 5) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);
    y[i].m = GGML_FP32_TO_FP16(min);

    uint32_t qh = 0;

    for (int j = 0; j < qk / 2; ++j)
    {
      const float x0 = (x[i * qk + 0 + j] - min) * id;
      const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

      const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
      const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

      y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10u) >> 4) << (j + qk / 2);
    }

    memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
  }
}

// reference implementation for deterministic creation of model files

static void quantize_row_q5_1_impl(const float *RESTRICT x, block_q5_1 *RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK5_1 == 32, "QK5_1 must be 32");

  if (!quant_weights)
  {
    quantize_row_q5_1_ref(x, y, n_per_row);
    return;
  }

  float weight[QK5_1];
  uint8_t L[QK5_1], Laux[QK5_1];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK5_1;
  for (int ib = 0; ib < nb; ++ib)
  {
    const float *xb = x + QK5_1 * ib;
    const float *qw = quant_weights + QK5_1 * ib;
    for (int j = 0; j < QK5_1; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float min;
    float d = make_qkx3_quants(QK5_1, 31, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
    y[ib].d = GGML_FP32_TO_FP16(d);
    y[ib].m = GGML_FP32_TO_FP16(-min);

    uint32_t qh = 0;
    for (int j = 0; j < 16; ++j)
    {
      const uint8_t xi0 = L[j];
      const uint8_t xi1 = L[j + 16];
      y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(&y[ib].qh, &qh, sizeof(qh));
  }
}

size_t quantize_q5_1(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    quantize_row_q5_1_ref(src, (block_q5_1 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q5_1, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q5_1, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_q5_1_impl(src, (block_q5_1 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

