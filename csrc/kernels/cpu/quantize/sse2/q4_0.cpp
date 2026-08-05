#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <emmintrin.h>
#include <xmmintrin.h>

static inline __m128 q4_0_blendv_ps(__m128 a, __m128 b, __m128 mask)
{
  return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}

static inline float q4_0_signed_absmax(const float *LIBGGUF_RESTRICT x)
{
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 step = _mm_set1_ps(4.0f);
  __m128 idx = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
  __m128 best_idx = _mm_set1_ps(2147483648.0f);
  __m128 best_abs = _mm_setzero_ps();
  __m128 best_val = _mm_setzero_ps();

  for (int j = 0; j < QK4_0; j += 4)
  {
    const __m128 v = _mm_loadu_ps(x + j);
    const __m128 av = _mm_andnot_ps(sign_mask, v);
    const __m128 mask = _mm_cmpgt_ps(av, best_abs);
    best_abs = q4_0_blendv_ps(best_abs, av, mask);
    best_val = q4_0_blendv_ps(best_val, v, mask);
    best_idx = q4_0_blendv_ps(best_idx, idx, mask);
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

static inline __m128i q4_0_quantize_4_i32(const float *LIBGGUF_RESTRICT x, __m128 id, __m128 offset)
{
  const __m128 scaled = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(x), id), offset);
  return _mm_cvttps_epi32(scaled);
}

extern "C" void quantize_block_q4_0(const float *LIBGGUF_RESTRICT x, block_q4_0 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK4_0 == 0);
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  const int nb = k / QK4_0;
  const __m128 offset = _mm_set1_ps(8.5f);
  const __m128i max_q = _mm_set1_epi16(15);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK4_0;
    const float max = q4_0_signed_absmax(xb);
    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;
    const __m128 idv = _mm_set1_ps(id);
    y[i].d = LIBGGUF_FP32_TO_FP16(d);

    for (int j = 0; j < QK4_0 / 2; j += 8)
    {
      const __m128i lo0 = q4_0_quantize_4_i32(xb + j, idv, offset);
      const __m128i lo1 = q4_0_quantize_4_i32(xb + j + 4, idv, offset);
      const __m128i hi0 = q4_0_quantize_4_i32(xb + QK4_0 / 2 + j, idv, offset);
      const __m128i hi1 = q4_0_quantize_4_i32(xb + QK4_0 / 2 + j + 4, idv, offset);
      const __m128i lo = _mm_min_epi16(_mm_packs_epi32(lo0, lo1), max_q);
      const __m128i hi = _mm_min_epi16(_mm_packs_epi32(hi0, hi1), max_q);
      const __m128i packed = _mm_or_si128(lo, _mm_slli_epi16(hi, 4));
      const __m128i bytes = _mm_packus_epi16(packed, _mm_setzero_si128());
      _mm_storel_epi64((__m128i *)(y[i].qs + j), bytes);
    }
  }
}


static void quantize_weighted_q4_0(const float *LIBGGUF_RESTRICT x, block_q4_0 *LIBGGUF_RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  if (!quant_weights)
  {
    quantize_block_q4_0(x, y, n_per_row);
    return;
  }

  float weight[QK4_0];
  int8_t L[QK4_0];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK4_0;
  for (int ib = 0; ib < nb; ++ib)
  {
    const float *xb = x + QK4_0 * ib;
    const float *qw = quant_weights + QK4_0 * ib;
    for (int j = 0; j < QK4_0; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float d = make_qx_quants(QK4_0, 8, xb, L, 1, weight);
    y[ib].d = LIBGGUF_FP32_TO_FP16(d);
    for (int j = 0; j < 16; ++j)
    {
      y[ib].qs[j] = L[j] | (L[j + 16] << 4);
    }
  }
}

size_t quantize_q4_0(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  if (!quant_weights)
  {
    quantize_block_q4_0(src, (block_q4_0 *)dst, (int64_t)nrow * n_per_row);
    return nrow * libgguf_row_size(GGML_TYPE_Q4_0, n_per_row);
  }
  size_t row_size = libgguf_row_size(GGML_TYPE_Q4_0, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_weighted_q4_0(src, (block_q4_0 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

