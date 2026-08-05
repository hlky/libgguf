#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <immintrin.h>

static inline float q4_0_signed_absmax(const float *LIBGGUF_RESTRICT x)
{
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 step = _mm256_set1_ps(8.0f);
  __m256 idx = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
  __m256 best_idx = _mm256_set1_ps(2147483648.0f);
  __m256 best_abs = _mm256_setzero_ps();
  __m256 best_val = _mm256_setzero_ps();

  for (int j = 0; j < QK4_0; j += 8)
  {
    const __m256 v = _mm256_loadu_ps(x + j);
    const __m256 av = _mm256_andnot_ps(sign_mask, v);
    const __m256 mask = _mm256_cmp_ps(av, best_abs, _CMP_GT_OQ);
    best_abs = _mm256_blendv_ps(best_abs, av, mask);
    best_val = _mm256_blendv_ps(best_val, v, mask);
    best_idx = _mm256_blendv_ps(best_idx, idx, mask);
    idx = _mm256_add_ps(idx, step);
  }

  alignas(32) float abs_parts[8];
  alignas(32) float val_parts[8];
  alignas(32) float idx_parts[8];
  _mm256_store_ps(abs_parts, best_abs);
  _mm256_store_ps(val_parts, best_val);
  _mm256_store_ps(idx_parts, best_idx);

  float amax = abs_parts[0];
  float max = val_parts[0];
  float min_idx = idx_parts[0];
  for (int lane = 1; lane < 8; ++lane)
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

static inline __m256i q4_0_quantize_8(const float *LIBGGUF_RESTRICT x, __m256 id, __m256 offset, __m256i max_q)
{
  const __m256 scaled = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x), id), offset);
  const __m256i q = _mm256_cvttps_epi32(scaled);
  return _mm256_min_epi32(q, max_q);
}

extern "C" void quantize_block_q4_0(const float *LIBGGUF_RESTRICT x, block_q4_0 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK4_0 == 0);
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  const int nb = k / QK4_0;
  const __m256 offset = _mm256_set1_ps(8.5f);
  const __m256i max_q = _mm256_set1_epi32(15);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK4_0;
    const float max = q4_0_signed_absmax(xb);
    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;
    const __m256 idv = _mm256_set1_ps(id);
    y[i].d = LIBGGUF_FP32_TO_FP16(d);

    for (int j = 0; j < QK4_0 / 2; j += 8)
    {
      const __m256i lo32 = q4_0_quantize_8(xb + j, idv, offset, max_q);
      const __m256i hi32 = q4_0_quantize_8(xb + QK4_0 / 2 + j, idv, offset, max_q);
      const __m256i byte32 = _mm256_or_si256(lo32, _mm256_slli_epi32(hi32, 4));
      const __m256i packed16 = _mm256_packs_epi32(byte32, _mm256_setzero_si256());
      const __m256i packed8 = _mm256_packus_epi16(packed16, _mm256_setzero_si256());
      const uint32_t out0 = (uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(packed8));
      const uint32_t out1 = (uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(packed8, 1));
      const uint64_t out = (uint64_t)out0 | ((uint64_t)out1 << 32);
      memcpy(y[i].qs + j, &out, sizeof(out));
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

