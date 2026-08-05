#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <emmintrin.h>
#include <xmmintrin.h>

static inline __m128 q5_0_blendv_ps(__m128 a, __m128 b, __m128 mask)
{
  return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}

static inline float q5_0_signed_absmax(const float *LIBGGUF_RESTRICT x)
{
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 step = _mm_set1_ps(4.0f);
  __m128 idx = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
  __m128 best_idx = _mm_set1_ps(2147483648.0f);
  __m128 best_abs = _mm_setzero_ps();
  __m128 best_val = _mm_setzero_ps();

  for (int j = 0; j < QK5_0; j += 4)
  {
    const __m128 v = _mm_loadu_ps(x + j);
    const __m128 av = _mm_andnot_ps(sign_mask, v);
    const __m128 mask = _mm_cmpgt_ps(av, best_abs);
    best_abs = q5_0_blendv_ps(best_abs, av, mask);
    best_val = q5_0_blendv_ps(best_val, v, mask);
    best_idx = q5_0_blendv_ps(best_idx, idx, mask);
    idx = _mm_add_ps(idx, step);
  }

  alignas(16) float abs_parts[4];
  alignas(16) float val_parts[4];
  alignas(16) float idx_parts[4];
  _mm_store_ps(abs_parts, best_abs);
  _mm_store_ps(val_parts, best_val);
  _mm_store_ps(idx_parts, best_idx);

  float amax = abs_parts[0];
  float max = val_parts[0];
  float min_idx = idx_parts[0];
  for (int lane = 1; lane < 4; ++lane)
  {
    if (abs_parts[lane] > amax || (abs_parts[lane] == amax && idx_parts[lane] < min_idx))
    {
      amax = abs_parts[lane];
      max = val_parts[lane];
      min_idx = idx_parts[lane];
    }
  }
  return max;
}

static inline __m128i q5_0_quantize_8_i16(const float *LIBGGUF_RESTRICT x, __m128 id, __m128 offset, __m128i max_q)
{
  const __m128i q0 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_loadu_ps(x), id), offset));
  const __m128i q1 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_loadu_ps(x + 4), id), offset));
  return _mm_min_epi16(_mm_packs_epi32(q0, q1), max_q);
}

extern "C" void quantize_block_q5_0(const float *LIBGGUF_RESTRICT x, block_q5_0 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK5_0 == 0);
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  const int nb = (int)(k / QK5_0);
  const __m128 offset = _mm_set1_ps(16.5f);
  const __m128i max_q = _mm_set1_epi16(31);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK5_0;
    const float max = q5_0_signed_absmax(xb);
    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = LIBGGUF_FP32_TO_FP16(d);

    const __m128 idv = _mm_set1_ps(id);
    uint32_t qh = 0;
    for (int j = 0; j < QK5_0 / 2; j += 8)
    {
      alignas(16) uint16_t lo[8];
      alignas(16) uint16_t hi[8];
      _mm_store_si128((__m128i *)lo, q5_0_quantize_8_i16(xb + j, idv, offset, max_q));
      _mm_store_si128((__m128i *)hi, q5_0_quantize_8_i16(xb + QK5_0 / 2 + j, idv, offset, max_q));
      for (int lane = 0; lane < 8; ++lane)
      {
        const uint8_t xi0 = (uint8_t)lo[lane];
        const uint8_t xi1 = (uint8_t)hi[lane];
        y[i].qs[j + lane] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + lane);
        qh |= ((xi1 & 0x10u) >> 4) << (j + lane + QK5_0 / 2);
      }
    }
    memcpy(&y[i].qh, &qh, sizeof(qh));
  }
}


static void quantize_weighted_q5_0(const float *LIBGGUF_RESTRICT x, block_q5_0 *LIBGGUF_RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  if (!quant_weights)
  {
    quantize_block_q5_0(x, y, n_per_row);
    return;
  }

  float weight[QK5_0];
  int8_t L[QK5_0];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK5_0;
  for (int ib = 0; ib < nb; ++ib)
  {
    const float *xb = x + QK5_0 * ib;
    const float *qw = quant_weights + QK5_0 * ib;
    for (int j = 0; j < QK5_0; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float d = make_qx_quants(QK5_0, 16, xb, L, 1, weight);
    y[ib].d = LIBGGUF_FP32_TO_FP16(d);

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

size_t quantize_q5_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    quantize_block_q5_0(src, (block_q5_0 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q5_0, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q5_0, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_weighted_q5_0(src, (block_q5_0 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

