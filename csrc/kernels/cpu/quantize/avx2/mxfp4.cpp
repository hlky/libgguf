#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>

#include "kernels/cpu/common/tables.h"

#include <immintrin.h>

static inline float mxfp4_hmax(__m256 v)
{
  __m128 maxv = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  maxv = _mm_max_ps(maxv, _mm_movehl_ps(maxv, maxv));
  maxv = _mm_max_ss(maxv, _mm_shuffle_ps(maxv, maxv, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(maxv);
}

static inline bool mxfp4_amax(const float *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT amax)
{
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 maxv = _mm256_setzero_ps();
  for (int i = 0; i < QK_MXFP4; i += 8)
  {
    const __m256 v = _mm256_loadu_ps(x + i);
    if (_mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q)))
    {
      return false;
    }
    maxv = _mm256_max_ps(maxv, _mm256_and_ps(v, abs_mask));
  }

  const float m = mxfp4_hmax(maxv);
  if (!std::isfinite(m))
  {
    return false;
  }
  *amax = m;
  return true;
}

static inline __m128i mxfp4_indices8(const float *LIBGGUF_RESTRICT x, float d)
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

static inline void mxfp4_pack8(const float *LIBGGUF_RESTRICT lo_x, const float *LIBGGUF_RESTRICT hi_x, float d,
                                    uint8_t *LIBGGUF_RESTRICT q)
{
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(mxfp4_indices8(lo_x, d), low_mask);
  const __m128i hi = _mm_slli_epi16(_mm_and_si128(mxfp4_indices8(hi_x, d), low_mask), 4);
  _mm_storel_epi64((__m128i *)q, _mm_or_si128(lo, hi));
}

static inline bool mxfp4_quantize_block(const float *LIBGGUF_RESTRICT x, block_mxfp4 *LIBGGUF_RESTRICT y)
{
  float amax = 0.0f;
  if (!mxfp4_amax(x, &amax))
  {
    return false;
  }

  const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2 + 127) : 0;
  const float d = LIBGGUF_E8M0_TO_FP32_HALF(e);
  y->e = e;

  for (int j = 0; j < QK_MXFP4 / 2; j += 8)
  {
    mxfp4_pack8(x + j, x + QK_MXFP4 / 2 + j, d, y->qs + j);
  }
  return true;
}

extern "C" void quantize_block_mxfp4(const float *LIBGGUF_RESTRICT x, block_mxfp4 *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_MXFP4 == 0);
  const int64_t nb = k / QK_MXFP4;
  for (int64_t i = 0; i < nb; ++i)
  {
    if (!mxfp4_quantize_block(x + i * QK_MXFP4, y + i))
    {
      quantize_block_mxfp4(x, y, k);
      return;
    }
  }
}


size_t quantize_mxfp4(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  LIBGGUF_UNUSED(quant_weights);
  quantize_block_mxfp4(src, (block_mxfp4 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_MXFP4, n_per_row);
}

