#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>

#include "kernels/cpu/common/tables.h"

#include <emmintrin.h>
#include <xmmintrin.h>

static inline float nvfp4_hmax(__m128 v)
{
  v = _mm_max_ps(v, _mm_movehl_ps(v, v));
  v = _mm_max_ss(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(v);
}

static inline bool nvfp4_amax_sub(const float *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT amax)
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

  const float m = nvfp4_hmax(maxv);
  if (!std::isfinite(m))
  {
    return false;
  }
  *amax = m;
  return true;
}

static inline __m128i nvfp4_indices4(const float *LIBGGUF_RESTRICT x, float d)
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
    best = _mm_or_ps(_mm_and_ps(mask_ps, err), _mm_andnot_ps(mask_ps, best));
    best_index = _mm_or_si128(_mm_and_si128(mask, _mm_set1_epi32(i)), _mm_andnot_si128(mask, best_index));
  }

  const __m128i zero = _mm_setzero_si128();
  const __m128i packed16 = _mm_packs_epi32(best_index, zero);
  return _mm_packus_epi16(packed16, zero);
}

static inline void nvfp4_pack4(const float *LIBGGUF_RESTRICT lo_x, const float *LIBGGUF_RESTRICT hi_x, float d,
                                    uint8_t *LIBGGUF_RESTRICT q)
{
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(nvfp4_indices4(lo_x, d), low_mask);
  const __m128i hi = _mm_slli_epi16(_mm_and_si128(nvfp4_indices4(hi_x, d), low_mask), 4);
  const int packed = _mm_cvtsi128_si32(_mm_or_si128(lo, hi));
  memcpy(q, &packed, sizeof(packed));
}

static inline bool nvfp4_quantize_subblock(const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT d_out,
                                                uint8_t *LIBGGUF_RESTRICT qs)
{
  float amax = 0.0f;
  if (!nvfp4_amax_sub(x, &amax))
  {
    return false;
  }

  const uint8_t ue = libgguf_fp32_to_ue4m3(amax / 6.0f);
  const float d = libgguf_ue4m3_to_fp32(ue);
  *d_out = ue;

  for (int j = 0; j < QK_NVFP4_SUB / 2; j += 4)
  {
    nvfp4_pack4(x + j, x + QK_NVFP4_SUB / 2 + j, d, qs + j);
  }
  return true;
}

extern "C" void quantize_block_nvfp4(const float *LIBGGUF_RESTRICT x, block_nvfp4 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_NVFP4 == 0);
  const int64_t nb = k / QK_NVFP4;
  for (int64_t i = 0; i < nb; ++i)
  {
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s)
    {
      const float *LIBGGUF_RESTRICT xb = x + i * QK_NVFP4 + s * QK_NVFP4_SUB;
      uint8_t *LIBGGUF_RESTRICT qs = y[i].qs + s * (QK_NVFP4_SUB / 2);
      if (!nvfp4_quantize_subblock(xb, y[i].d + s, qs))
      {
        quantize_block_nvfp4(x, y, k);
        return;
      }
    }
  }
}


size_t quantize_nvfp4(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  LIBGGUF_UNUSED(quant_weights);
  quantize_block_nvfp4(src, (block_nvfp4 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_NVFP4, n_per_row);
}

