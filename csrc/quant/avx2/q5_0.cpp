#include "libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_AVX2)
static inline float q5_0_signed_absmax_avx2(const float *RESTRICT x)
{
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 step = _mm256_set1_ps(8.0f);
  __m256 idx = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
  __m256 best_idx = _mm256_set1_ps(2147483648.0f);
  __m256 best_abs = _mm256_setzero_ps();
  __m256 best_val = _mm256_setzero_ps();

  for (int j = 0; j < QK5_0; j += 8)
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

static inline __m256i q5_0_quantize_8_avx2(const float *RESTRICT x, __m256 id, __m256 offset, __m256i max_q)
{
  const __m256 scaled = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x), id), offset);
  return _mm256_min_epi32(_mm256_cvttps_epi32(scaled), max_q);
}
#endif

extern "C" void quantize_row_q5_0_avx2(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK5_0 == 0);
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  const int nb = (int)(k / QK5_0);
  const __m256 offset = _mm256_set1_ps(16.5f);
  const __m256i max_q = _mm256_set1_epi32(31);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK5_0;
    const float max = q5_0_signed_absmax_avx2(xb);
    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = GGML_FP32_TO_FP16(d);

    const __m256 idv = _mm256_set1_ps(id);
    uint32_t qh = 0;
    for (int j = 0; j < QK5_0 / 2; j += 8)
    {
      alignas(32) int32_t lo[8];
      alignas(32) int32_t hi[8];
      _mm256_store_si256((__m256i *)lo, q5_0_quantize_8_avx2(xb + j, idv, offset, max_q));
      _mm256_store_si256((__m256i *)hi, q5_0_quantize_8_avx2(xb + QK5_0 / 2 + j, idv, offset, max_q));
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
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
