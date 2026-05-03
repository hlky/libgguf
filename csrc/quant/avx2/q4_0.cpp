#include "libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_AVX2)
static inline float q4_0_signed_absmax_avx2(const float *RESTRICT x)
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

static inline __m256i q4_0_quantize_8_avx2(const float *RESTRICT x, __m256 id, __m256 offset, __m256i max_q)
{
  const __m256 scaled = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x), id), offset);
  const __m256i q = _mm256_cvttps_epi32(scaled);
  return _mm256_min_epi32(q, max_q);
}
#endif

extern "C" void quantize_row_q4_0_avx2(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK4_0 == 0);
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  const int nb = k / QK4_0;
  const __m256 offset = _mm256_set1_ps(8.5f);
  const __m256i max_q = _mm256_set1_epi32(15);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK4_0;
    const float max = q4_0_signed_absmax_avx2(xb);
    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;
    const __m256 idv = _mm256_set1_ps(id);
    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK4_0 / 2; j += 8)
    {
      const __m256i lo32 = q4_0_quantize_8_avx2(xb + j, idv, offset, max_q);
      const __m256i hi32 = q4_0_quantize_8_avx2(xb + QK4_0 / 2 + j, idv, offset, max_q);
      const __m256i byte32 = _mm256_or_si256(lo32, _mm256_slli_epi32(hi32, 4));
      const __m256i packed16 = _mm256_packs_epi32(byte32, _mm256_setzero_si256());
      const __m256i packed8 = _mm256_packus_epi16(packed16, _mm256_setzero_si256());
      const uint32_t out0 = (uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(packed8));
      const uint32_t out1 = (uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(packed8, 1));
      const uint64_t out = (uint64_t)out0 | ((uint64_t)out1 << 32);
      memcpy(y[i].qs + j, &out, sizeof(out));
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
