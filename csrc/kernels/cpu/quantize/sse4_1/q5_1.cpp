#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <smmintrin.h>

static inline __m128i q5_1_quantize_4_i32(const float *LIBGGUF_RESTRICT x, __m128 minv, __m128 id, __m128 half)
{
  return _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(x), minv), id), half));
}

extern "C" void quantize_block_q5_1(const float *LIBGGUF_RESTRICT x, block_q5_1 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK5_1 == 0);
  static_assert(QK5_1 == 32, "QK5_1 must be 32");

  const int nb = (int)(k / QK5_1);
  const __m128 half = _mm_set1_ps(0.5f);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK5_1;
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int j = 0; j < QK5_1; ++j)
    {
      const float v = xb[j];
      if (v < min)
        min = v;
      if (v > max)
        max = v;
    }

    const float d = (max - min) / ((1 << 5) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = LIBGGUF_FP32_TO_FP16(d);
    y[i].m = LIBGGUF_FP32_TO_FP16(min);

    const __m128 minv = _mm_set1_ps(min);
    const __m128 idv = _mm_set1_ps(id);
    uint32_t qh = 0;
    for (int j = 0; j < QK5_1 / 2; j += 4)
    {
      alignas(16) int32_t lo[4];
      alignas(16) int32_t hi[4];
      _mm_store_si128((__m128i *)lo, q5_1_quantize_4_i32(xb + j, minv, idv, half));
      _mm_store_si128((__m128i *)hi, q5_1_quantize_4_i32(xb + QK5_1 / 2 + j, minv, idv, half));
      for (int lane = 0; lane < 4; ++lane)
      {
        const uint8_t xi0 = (uint8_t)lo[lane];
        const uint8_t xi1 = (uint8_t)hi[lane];
        y[i].qs[j + lane] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + lane);
        qh |= ((xi1 & 0x10u) >> 4) << (j + lane + QK5_1 / 2);
      }
    }
    memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
  }
}


static void quantize_weighted_q5_1(const float *LIBGGUF_RESTRICT x, block_q5_1 *LIBGGUF_RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK5_1 == 32, "QK5_1 must be 32");

  if (!quant_weights)
  {
    quantize_block_q5_1(x, y, n_per_row);
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
    y[ib].d = LIBGGUF_FP32_TO_FP16(d);
    y[ib].m = LIBGGUF_FP32_TO_FP16(-min);

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

size_t quantize_q5_1(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    quantize_block_q5_1(src, (block_q5_1 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q5_1, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q5_1, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_weighted_q5_1(src, (block_q5_1 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

