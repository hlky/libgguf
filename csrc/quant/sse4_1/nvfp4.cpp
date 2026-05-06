#include "libgguf_common.h"
#include "libgguf_tables.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_NVFP4_SSE4_1 1
#endif

#if defined(LIBGGUF_BUILD_NVFP4_SSE4_1)
static inline float nvfp4_hmax_sse4_1(__m128 v)
{
  v = _mm_max_ps(v, _mm_movehl_ps(v, v));
  v = _mm_max_ss(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(v);
}

static inline bool nvfp4_amax_sub_sse4_1(const float *RESTRICT x, float *RESTRICT amax)
{
  const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
  __m128 maxv = _mm_setzero_ps();
  for (int i = 0; i < QK_NVFP4_SUB; i += 4)
  {
    const __m128 v = _mm_loadu_ps(x + i);
    if (_mm_movemask_ps(_mm_cmpneq_ps(v, v)))
    {
      return false;
    }
    maxv = _mm_max_ps(maxv, _mm_and_ps(v, abs_mask));
  }

  const float m = nvfp4_hmax_sse4_1(maxv);
  if (!std::isfinite(m))
  {
    return false;
  }
  *amax = m;
  return true;
}

static inline __m128i nvfp4_indices4_sse4_1(const float *RESTRICT x, float d)
{
  const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
  const __m128 xv = _mm_loadu_ps(x);
  __m128 best = _mm_and_ps(xv, abs_mask);
  __m128i best_index = _mm_setzero_si128();

  for (int i = 1; i < 16; ++i)
  {
    const __m128 qv = _mm_set1_ps((float)kvalues_mxfp4[i] * d);
    const __m128 err = _mm_and_ps(_mm_sub_ps(qv, xv), abs_mask);
    const __m128 mask_ps = _mm_cmplt_ps(err, best);
    const __m128i mask = _mm_castps_si128(mask_ps);
    best = _mm_blendv_ps(best, err, mask_ps);
    best_index = _mm_blendv_epi8(best_index, _mm_set1_epi32(i), mask);
  }

  const __m128i zero = _mm_setzero_si128();
  const __m128i packed16 = _mm_packs_epi32(best_index, zero);
  return _mm_packus_epi16(packed16, zero);
}

static inline void nvfp4_pack4_sse4_1(const float *RESTRICT lo_x, const float *RESTRICT hi_x, float d,
                                      uint8_t *RESTRICT q)
{
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(nvfp4_indices4_sse4_1(lo_x, d), low_mask);
  const __m128i hi = _mm_slli_epi16(_mm_and_si128(nvfp4_indices4_sse4_1(hi_x, d), low_mask), 4);
  const int packed = _mm_cvtsi128_si32(_mm_or_si128(lo, hi));
  memcpy(q, &packed, sizeof(packed));
}

static inline bool nvfp4_quantize_subblock_sse4_1(const float *RESTRICT x, uint8_t *RESTRICT d_out,
                                                  uint8_t *RESTRICT qs)
{
  float amax = 0.0f;
  if (!nvfp4_amax_sub_sse4_1(x, &amax))
  {
    return false;
  }

  const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
  const float d = ggml_ue4m3_to_fp32(ue);
  *d_out = ue;

  for (int j = 0; j < QK_NVFP4_SUB / 2; j += 4)
  {
    nvfp4_pack4_sse4_1(x + j, x + QK_NVFP4_SUB / 2 + j, d, qs + j);
  }
  return true;
}
#endif

extern "C" void quantize_row_nvfp4_sse4_1(const float *RESTRICT x, block_nvfp4 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_NVFP4_SSE4_1)
  assert(k % QK_NVFP4 == 0);
  const int64_t nb = k / QK_NVFP4;
  for (int64_t i = 0; i < nb; ++i)
  {
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s)
    {
      const float *RESTRICT xb = x + i * QK_NVFP4 + s * QK_NVFP4_SUB;
      uint8_t *RESTRICT qs = y[i].qs + s * (QK_NVFP4_SUB / 2);
      if (!nvfp4_quantize_subblock_sse4_1(xb, y[i].d + s, qs))
      {
        quantize_row_nvfp4(x, y, k);
        return;
      }
    }
  }
#else
  quantize_row_nvfp4(x, y, k);
#endif
}
