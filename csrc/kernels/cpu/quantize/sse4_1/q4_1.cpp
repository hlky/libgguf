#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <smmintrin.h>

static inline __m128i q4_1_quantize_4_i32(const float *LIBGGUF_RESTRICT x, __m128 minv, __m128 id, __m128 half, __m128i max_q)
{
  const __m128 scaled = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(x), minv), id), half);
  return _mm_min_epi32(_mm_cvttps_epi32(scaled), max_q);
}

extern "C" void quantize_block_q4_1(const float *LIBGGUF_RESTRICT x, block_q4_1 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK4_1 == 0);
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  const int nb = (int)(k / QK4_1);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128i max_q = _mm_set1_epi32(15);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK4_1;
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int j = 0; j < QK4_1; ++j)
    {
      const float v = xb[j];
      if (v < min)
        min = v;
      if (v > max)
        max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = LIBGGUF_FP32_TO_FP16(d);
    y[i].m = LIBGGUF_FP32_TO_FP16(min);

    const __m128 minv = _mm_set1_ps(min);
    const __m128 idv = _mm_set1_ps(id);
    for (int j = 0; j < QK4_1 / 2; j += 8)
    {
      const __m128i lo0 = q4_1_quantize_4_i32(xb + j, minv, idv, half, max_q);
      const __m128i lo1 = q4_1_quantize_4_i32(xb + j + 4, minv, idv, half, max_q);
      const __m128i hi0 = q4_1_quantize_4_i32(xb + QK4_1 / 2 + j, minv, idv, half, max_q);
      const __m128i hi1 = q4_1_quantize_4_i32(xb + QK4_1 / 2 + j + 4, minv, idv, half, max_q);
      const __m128i lo = _mm_packs_epi32(lo0, lo1);
      const __m128i hi = _mm_packs_epi32(hi0, hi1);
      const __m128i packed = _mm_or_si128(lo, _mm_slli_epi16(hi, 4));
      const __m128i bytes = _mm_packus_epi16(packed, _mm_setzero_si128());
      _mm_storel_epi64((__m128i *)(y[i].qs + j), bytes);
    }
  }
}


static void quantize_weighted_q4_1(const float *LIBGGUF_RESTRICT x, block_q4_1 *LIBGGUF_RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  if (!quant_weights)
  {
    quantize_block_q4_1(x, y, n_per_row);
    return;
  }

  float weight[QK4_1];
  uint8_t L[QK4_1], Laux[QK4_1];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK4_1;
  for (int ib = 0; ib < nb; ++ib)
  {
    const float *xb = x + QK4_1 * ib;
    const float *qw = quant_weights + QK4_1 * ib;
    for (int j = 0; j < QK4_1; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float min;
    float d = make_qkx3_quants(QK4_1, 15, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
    y[ib].d = LIBGGUF_FP32_TO_FP16(d);
    y[ib].m = LIBGGUF_FP32_TO_FP16(-min);
    for (int j = 0; j < 16; ++j)
    {
      y[ib].qs[j] = L[j] | (L[j + 16] << 4);
    }
  }
}

size_t quantize_q4_1(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    quantize_block_q4_1(src, (block_q4_1 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q4_1, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q4_1, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_weighted_q4_1(src, (block_q4_1 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

