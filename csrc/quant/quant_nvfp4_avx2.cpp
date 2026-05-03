#include "libgguf_common.h"
#include "libgguf_tables.h"

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_NVFP4_AVX2 1
#endif

#if defined(LIBGGUF_BUILD_NVFP4_AVX2)
static inline float nvfp4_hmax_avx2(__m256 v)
{
  __m128 maxv = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  maxv = _mm_max_ps(maxv, _mm_movehl_ps(maxv, maxv));
  maxv = _mm_max_ss(maxv, _mm_shuffle_ps(maxv, maxv, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(maxv);
}

static inline bool nvfp4_amax_sub_avx2(const float *RESTRICT x, float *RESTRICT amax)
{
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 maxv = _mm256_setzero_ps();
  for (int i = 0; i < QK_NVFP4_SUB; i += 8)
  {
    const __m256 v = _mm256_loadu_ps(x + i);
    if (_mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q)))
    {
      return false;
    }
    maxv = _mm256_max_ps(maxv, _mm256_and_ps(v, abs_mask));
  }

  const float m = nvfp4_hmax_avx2(maxv);
  if (!isfinite(m))
  {
    return false;
  }
  *amax = m;
  return true;
}

static inline __m128i nvfp4_indices8_avx2(const float *RESTRICT x, float d)
{
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256 xv = _mm256_loadu_ps(x);
  __m256 best = _mm256_and_ps(xv, abs_mask);
  __m256i best_index = _mm256_setzero_si256();

  for (int i = 1; i < 16; ++i)
  {
    const __m256 qv = _mm256_set1_ps((float)kvalues_mxfp4[i] * d);
    const __m256 err = _mm256_and_ps(_mm256_sub_ps(qv, xv), abs_mask);
    const __m256 mask_ps = _mm256_cmp_ps(err, best, _CMP_LT_OQ);
    const __m256i mask = _mm256_castps_si256(mask_ps);
    best = _mm256_or_ps(_mm256_and_ps(mask_ps, err), _mm256_andnot_ps(mask_ps, best));
    best_index = _mm256_or_si256(_mm256_and_si256(mask, _mm256_set1_epi32(i)),
                                 _mm256_andnot_si256(mask, best_index));
  }

  const __m128i zero = _mm_setzero_si128();
  __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(best_index), _mm256_extracti128_si256(best_index, 1));
  packed = _mm_packus_epi16(packed, zero);
  return packed;
}

static inline void nvfp4_pack8_avx2(const float *RESTRICT lo_x, const float *RESTRICT hi_x, float d,
                                    uint8_t *RESTRICT q)
{
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(nvfp4_indices8_avx2(lo_x, d), low_mask);
  const __m128i hi = _mm_slli_epi16(_mm_and_si128(nvfp4_indices8_avx2(hi_x, d), low_mask), 4);
  _mm_storel_epi64((__m128i *)q, _mm_or_si128(lo, hi));
}

static inline bool nvfp4_quantize_subblock_avx2(const float *RESTRICT x, uint8_t *RESTRICT d_out,
                                                uint8_t *RESTRICT qs)
{
  float amax = 0.0f;
  if (!nvfp4_amax_sub_avx2(x, &amax))
  {
    return false;
  }

  const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
  const float d = ggml_ue4m3_to_fp32(ue);
  *d_out = ue;

  nvfp4_pack8_avx2(x, x + QK_NVFP4_SUB / 2, d, qs);
  return true;
}
#endif

extern "C" void quantize_row_nvfp4_avx2(const float *RESTRICT x, block_nvfp4 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_NVFP4_AVX2)
  assert(k % QK_NVFP4 == 0);
  const int64_t nb = k / QK_NVFP4;
  for (int64_t i = 0; i < nb; ++i)
  {
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s)
    {
      const float *RESTRICT xb = x + i * QK_NVFP4 + s * QK_NVFP4_SUB;
      uint8_t *RESTRICT qs = y[i].qs + s * (QK_NVFP4_SUB / 2);
      if (!nvfp4_quantize_subblock_avx2(xb, y[i].d + s, qs))
      {
        quantize_row_nvfp4_ref(x, y, k);
        return;
      }
    }
  }
#else
  quantize_row_nvfp4_ref(x, y, k);
#endif
}
