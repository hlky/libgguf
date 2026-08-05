#include "kernels/cpu/common/common.h"

#include <immintrin.h>

static float make_qx_quants_scalar(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, int rmse_type,
                                        const float *LIBGGUF_RESTRICT qw)
{
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i)
  {
    float ax = fabsf(x[i]);
    if (ax > amax)
    {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS)
  { // all zero
    for (int i = 0; i < n; ++i)
    {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (rmse_type == 0)
  {
    for (int i = 0; i < n; ++i)
    {
      int l = nearest_int(iscale * x[i]);
      L[i] = nmax + libgguf_max(-nmax, libgguf_min(nmax - 1, l));
    }
    return 1 / iscale;
  }
  bool return_early = false;
  if (rmse_type < 0)
  {
    rmse_type = -rmse_type;
    return_early = true;
  }
  float sumlx = 0;
  float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 0; i < n; ++i)
  {
#else
  for (int i = 0; i < n; ++i)
  {
#endif
    int l = nearest_int(iscale * x[i]);
    l = libgguf_max(-nmax, libgguf_min(nmax - 1, l));
    L[i] = l + nmax;
    float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i]
                       : rmse_type == 2   ? 1
                       : rmse_type == 3   ? fabsf(x[i])
                                          : sqrtf(fabsf(x[i]));
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  float scale = suml2 ? sumlx / suml2 : 0.0f;
  if (return_early)
    return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; ++is)
  {
    if (is == 0)
    {
      continue;
    }
    iscale = -(nmax + 0.1f * is) / max;
    sumlx = suml2 = 0;
    for (int i = 0; i < n; ++i)
    {
      int l = nearest_int(iscale * x[i]);
      l = libgguf_max(-nmax, libgguf_min(nmax - 1, l));
      float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i]
                         : rmse_type == 2   ? 1
                         : rmse_type == 3   ? fabsf(x[i])
                                            : sqrtf(fabsf(x[i]));
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2)
    {
      for (int i = 0; i < n; ++i)
      {
        int l = nearest_int(iscale * x[i]);
        L[i] = nmax + libgguf_max(-nmax, libgguf_min(nmax - 1, l));
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

static float make_q3_quants_scalar(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, bool do_rmse)
{
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i)
  {
    float ax = fabsf(x[i]);
    if (ax > amax)
    {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS)
  { // all zero
    for (int i = 0; i < n; ++i)
    {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (do_rmse)
  {
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i)
    {
      int l = nearest_int(iscale * x[i]);
      l = libgguf_max(-nmax, libgguf_min(nmax - 1, l));
      L[i] = l;
      float w = x[i] * x[i];
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry)
    {
      int n_changed = 0;
      for (int i = 0; i < n; ++i)
      {
        float w = x[i] * x[i];
        float slx = sumlx - w * x[i] * L[i];
        if (slx > 0)
        {
          float sl2 = suml2 - w * L[i] * L[i];
          int new_l = nearest_int(x[i] * sl2 / slx);
          new_l = libgguf_max(-nmax, libgguf_min(nmax - 1, new_l));
          if (new_l != L[i])
          {
            slx += w * x[i] * new_l;
            sl2 += w * new_l * new_l;
            if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2)
            {
              L[i] = new_l;
              sumlx = slx;
              suml2 = sl2;
              ++n_changed;
            }
          }
        }
      }
      if (!n_changed)
      {
        break;
      }
    }
    for (int i = 0; i < n; ++i)
    {
      L[i] += nmax;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
  }
  for (int i = 0; i < n; ++i)
  {
    int l = nearest_int(iscale * x[i]);
    l = libgguf_max(-nmax, libgguf_min(nmax - 1, l));
    L[i] = l + nmax;
  }
  return 1 / iscale;
}

static float make_qp_quants_scalar(int n, int nmax, const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT L,
                                        const float *quant_weights)
{
  float max = 0;
  for (int i = 0; i < n; ++i)
  {
    max = libgguf_max(max, x[i]);
  }
  if (max < GROUP_MAX_EPS)
  { // all zero
    for (int i = 0; i < n; ++i)
    {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = nmax / max;
  for (int i = 0; i < n; ++i)
  {
    L[i] = nearest_int(iscale * x[i]);
  }
  float scale = 1 / iscale;
  float best_mse = 0;
  for (int i = 0; i < n; ++i)
  {
    float diff = x[i] - scale * L[i];
    float w = quant_weights[i];
    best_mse += w * diff * diff;
  }
  for (int is = -4; is <= 4; ++is)
  {
    if (is == 0)
      continue;
    float iscale_is = (0.1f * is + nmax) / max;
    float scale_is = 1 / iscale_is;
    float mse = 0;
    for (int i = 0; i < n; ++i)
    {
      int l = nearest_int(iscale_is * x[i]);
      l = libgguf_min(nmax, l);
      float diff = x[i] - scale_is * l;
      float w = quant_weights[i];
      mse += w * diff * diff;
    }
    if (mse < best_mse)
    {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i)
  {
    int l = nearest_int(iscale * x[i]);
    l = libgguf_min(nmax, l);
    L[i] = l;
    float w = quant_weights[i];
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  for (int itry = 0; itry < 5; ++itry)
  {
    int n_changed = 0;
    for (int i = 0; i < n; ++i)
    {
      float w = quant_weights[i];
      float slx = sumlx - w * x[i] * L[i];
      float sl2 = suml2 - w * L[i] * L[i];
      if (slx > 0 && sl2 > 0)
      {
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = libgguf_min(nmax, new_l);
        if (new_l != L[i])
        {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (slx * slx * suml2 > sumlx * sumlx * sl2)
          {
            L[i] = new_l;
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed)
    {
      break;
    }
  }
  return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}


int nearest_int(float fval)
{
  assert(fabsf(fval) <= 4194303.f);
  const __m128 val = _mm_add_ss(_mm_set_ss(fval), _mm_set_ss(12582912.f));
  const int bits = _mm_cvtsi128_si32(_mm_castps_si128(val));
  return (bits & 0x007fffff) - 0x00400000;
}

static inline __m256i libgguf_nearest_i32x8(__m256 fval)
{
  const __m256 val = _mm256_add_ps(fval, _mm256_set1_ps(12582912.f));
  const __m256i bits = _mm256_castps_si256(val);
  return _mm256_sub_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x00400000));
}

static inline __m128i libgguf_pack_i32x8_to_i8(__m256i q)
{
  const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
  return _mm_packs_epi16(q16, _mm_setzero_si128());
}

static inline __m128i libgguf_pack_u32x8_to_u8(__m256i q)
{
  const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
  return _mm_packus_epi16(q16, _mm_setzero_si128());
}

static inline void libgguf_store_qx_i8x8(const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, __m256 iscale, int nmax,
                                              int bias)
{
  __m256i q = libgguf_nearest_i32x8(_mm256_mul_ps(_mm256_loadu_ps(x), iscale));
  q = _mm256_max_epi32(q, _mm256_set1_epi32(-nmax));
  q = _mm256_min_epi32(q, _mm256_set1_epi32(nmax - 1));
  q = _mm256_add_epi32(q, _mm256_set1_epi32(bias));
  _mm_storel_epi64((__m128i *)L, libgguf_pack_i32x8_to_i8(q));
}

static inline void libgguf_fill_i8(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, float iscale,
                                        int bias)
{
  const __m256 iscalev = _mm256_set1_ps(iscale);
  int i = 0;
  for (; i + 8 <= n; i += 8)
  {
    libgguf_store_qx_i8x8(x + i, L + i, iscalev, nmax, bias);
  }
  for (; i < n; ++i)
  {
    int l = nearest_int(iscale * x[i]);
    L[i] = (int8_t)(bias + libgguf_max(-nmax, libgguf_min(nmax - 1, l)));
  }
}

static inline void libgguf_fill_u8_min(int n, int nmax, const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT L,
                                            float iscale, float min)
{
  const __m256 iscalev = _mm256_set1_ps(iscale);
  const __m256 minv = _mm256_set1_ps(min);
  int i = 0;
  for (; i + 8 <= n; i += 8)
  {
    __m256i q = libgguf_nearest_i32x8(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), minv), iscalev));
    q = _mm256_max_epi32(q, _mm256_setzero_si256());
    q = _mm256_min_epi32(q, _mm256_set1_epi32(nmax));
    _mm_storel_epi64((__m128i *)(L + i), libgguf_pack_u32x8_to_u8(q));
  }
  for (; i < n; ++i)
  {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = (uint8_t)libgguf_max(0, libgguf_min(nmax, l));
  }
}

static inline void libgguf_fill_u8_qp(int n, int nmax, const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT L,
                                           float iscale, bool clamp_max)
{
  const __m256 iscalev = _mm256_set1_ps(iscale);
  int i = 0;
  for (; i + 8 <= n; i += 8)
  {
    __m256i q = libgguf_nearest_i32x8(_mm256_mul_ps(_mm256_loadu_ps(x + i), iscalev));
    if (clamp_max)
    {
      q = _mm256_min_epi32(q, _mm256_set1_epi32(nmax));
    }
    _mm_storel_epi64((__m128i *)(L + i), libgguf_pack_u32x8_to_u8(q));
  }
  for (; i < n; ++i)
  {
    int l = nearest_int(iscale * x[i]);
    if (clamp_max)
    {
      l = libgguf_min(nmax, l);
    }
    L[i] = (uint8_t)l;
  }
}

static inline float make_qx_quants_no_rmse(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L)
{
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i)
  {
    float ax = fabsf(x[i]);
    if (ax > amax)
    {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS)
  {
    memset(L, 0, n);
    return 0.f;
  }

  const float iscale_scalar = -nmax / max;
  const __m256 iscale = _mm256_set1_ps(iscale_scalar);
  int i = 0;
  for (; i + 8 <= n; i += 8)
  {
    libgguf_store_qx_i8x8(x + i, L + i, iscale, nmax, nmax);
  }
  for (; i < n; ++i)
  {
    int l = nearest_int(iscale_scalar * x[i]);
    L[i] = (int8_t)(nmax + libgguf_max(-nmax, libgguf_min(nmax - 1, l)));
  }
  return 1 / iscale_scalar;
}

int best_index_mxfp4(float x, float e)
{
  alignas(32) float err[16];
  const __m256 xv = _mm256_set1_ps(x);
  const __m256 ev = _mm256_set1_ps(e);
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  for (int i = 0; i < 16; i += 8)
  {
    const __m128i kv8 = _mm_loadl_epi64((const __m128i *)(kvalues_mxfp4 + i));
    const __m256 kv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(kv8));
    _mm256_store_ps(err + i, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(_mm256_mul_ps(kv, ev), xv)));
  }
  int best_index = 0;
  float best_err = err[0];
  for (int i = 1; i < 16; ++i)
  {
    if (err[i] < best_err)
    {
      best_index = i;
      best_err = err[i];
    }
  }
  return best_index;
}

int best_index_int8(int n, const int8_t *val, float x)
{
  if (n != 16 || x <= val[0] || x >= val[n - 1])
  {
    return best_index_int8_scalar(n, val, x);
  }

  alignas(32) float err[16];
  const __m256 xv = _mm256_set1_ps(x);
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  for (int i = 0; i < 16; i += 8)
  {
    const __m128i v = _mm_loadl_epi64((const __m128i *)(val + i));
    const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
    _mm256_store_ps(err + i, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(xv, vf)));
  }

  int best = 0;
  float best_err = err[0];
  for (int i = 1; i < 16; ++i)
  {
    if (err[i] <= best_err)
    {
      best = i;
      best_err = err[i];
    }
  }
  return best;
}












float make_qx_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L,
                                                int rmse_type, const float *LIBGGUF_RESTRICT qw)
{
  if (n > 256)
  {
    return make_qx_quants_scalar(n, nmax, x, L, rmse_type, qw);
  }

  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i)
  {
    float ax = fabsf(x[i]);
    if (ax > amax)
    {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS)
  {
    memset(L, 0, n);
    return 0.f;
  }

  float iscale = -nmax / max;
  libgguf_fill_i8(n, nmax, x, L, iscale, nmax);
  if (rmse_type == 0)
  {
    return 1 / iscale;
  }

  bool return_early = false;
  if (rmse_type < 0)
  {
    rmse_type = -rmse_type;
    return_early = true;
  }

  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i)
  {
    const int l = (int)L[i] - nmax;
    float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i]
                       : rmse_type == 2   ? 1
                       : rmse_type == 3   ? fabsf(x[i])
                                          : sqrtf(fabsf(x[i]));
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  float scale = suml2 ? sumlx / suml2 : 0.0f;
  if (return_early)
  {
    return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
  }

  int8_t Ltmp[256];
  float best = scale * sumlx;
  for (int is = -9; is <= 9; ++is)
  {
    if (is == 0)
    {
      continue;
    }
    iscale = -(nmax + 0.1f * is) / max;
    libgguf_fill_i8(n, nmax, x, Ltmp, iscale, nmax);
    sumlx = suml2 = 0;
    for (int i = 0; i < n; ++i)
    {
      const int l = (int)Ltmp[i] - nmax;
      float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i]
                         : rmse_type == 2   ? 1
                         : rmse_type == 3   ? fabsf(x[i])
                                            : sqrtf(fabsf(x[i]));
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2)
    {
      memcpy(L, Ltmp, n);
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

float make_q3_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L,
                                                bool do_rmse)
{
  if (n > 256)
  {
    return make_q3_quants_scalar(n, nmax, x, L, do_rmse);
  }

  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i)
  {
    float ax = fabsf(x[i]);
    if (ax > amax)
    {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS)
  {
    memset(L, 0, n);
    return 0.f;
  }

  const float iscale = -nmax / max;
  if (!do_rmse)
  {
    libgguf_fill_i8(n, nmax, x, L, iscale, nmax);
    return 1 / iscale;
  }

  libgguf_fill_i8(n, nmax, x, L, iscale, 0);
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i)
  {
    float w = x[i] * x[i];
    sumlx += w * x[i] * L[i];
    suml2 += w * L[i] * L[i];
  }
  for (int itry = 0; itry < 5; ++itry)
  {
    int n_changed = 0;
    for (int i = 0; i < n; ++i)
    {
      float w = x[i] * x[i];
      float slx = sumlx - w * x[i] * L[i];
      if (slx > 0)
      {
        float sl2 = suml2 - w * L[i] * L[i];
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = libgguf_max(-nmax, libgguf_min(nmax - 1, new_l));
        if (new_l != L[i])
        {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2)
          {
            L[i] = new_l;
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed)
    {
      break;
    }
  }
  for (int i = 0; i < n; ++i)
  {
    L[i] += nmax;
  }
  return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

float make_qkx2_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x,
                                                  const float *LIBGGUF_RESTRICT weights, uint8_t *LIBGGUF_RESTRICT L,
                                                  float *LIBGGUF_RESTRICT the_min, uint8_t *LIBGGUF_RESTRICT Laux, float rmin,
                                                  float rdelta, int nstep, bool use_mad)
{
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i)
  {
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0)
    min = 0;
  if (max == min)
  {
    memset(L, 0, n);
    *the_min = -min;
    return 0.f;
  }

  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_error = 0;
  libgguf_fill_u8_min(n, nmax, x, L, iscale, min);
  for (int i = 0; i < n; ++i)
  {
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    best_error += weights[i] * diff;
  }
  if (nstep < 1)
  {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is)
  {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    libgguf_fill_u8_min(n, nmax, x, Laux, iscale, min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i)
    {
      const int l = Laux[i];
      float w = weights[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0)
    {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0)
      {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float cur_error = 0;
      for (int i = 0; i < n; ++i)
      {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        cur_error += weights[i] * diff;
      }
      if (cur_error < best_error)
      {
        memcpy(L, Laux, n);
        best_error = cur_error;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

float make_qkx3_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x,
                                                  const float *LIBGGUF_RESTRICT weights, uint8_t *LIBGGUF_RESTRICT L,
                                                  float *LIBGGUF_RESTRICT the_min, uint8_t *LIBGGUF_RESTRICT Laux, float rmin,
                                                  float rdelta, int nstep, bool use_mad)
{
  float min = x[0];
  float max = x[0];
  float sum_w = weights ? weights[0] : x[0] * x[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i)
  {
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights ? weights[i] : x[i] * x[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0)
  {
    min = 0;
  }
  if (max <= min)
  {
    memset(L, 0, n);
    *the_min = -min;
    return 0.f;
  }

  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  libgguf_fill_u8_min(n, nmax, x, L, iscale, min);
  for (int i = 0; i < n; ++i)
  {
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights ? weights[i] : x[i] * x[i];
    best_mad += w * diff;
  }
  if (nstep < 1)
  {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is)
  {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    libgguf_fill_u8_min(n, nmax, x, Laux, iscale, min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i)
    {
      const int l = Laux[i];
      float w = weights ? weights[i] : x[i] * x[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0)
    {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0)
      {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i)
      {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        mad += w * diff;
      }
      if (mad < best_mad)
      {
        memcpy(L, Laux, n);
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

float make_qp_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT L,
                                                const float *quant_weights)
{
  if (n > 256)
  {
    return make_qp_quants_scalar(n, nmax, x, L, quant_weights);
  }

  float max = 0;
  for (int i = 0; i < n; ++i)
  {
    if (x[i] < 0)
    {
      return make_qp_quants_scalar(n, nmax, x, L, quant_weights);
    }
    max = libgguf_max(max, x[i]);
  }
  if (max < GROUP_MAX_EPS)
  {
    memset(L, 0, n);
    return 0.f;
  }

  float iscale = nmax / max;
  libgguf_fill_u8_qp(n, nmax, x, L, iscale, false);
  float scale = 1 / iscale;
  float best_mse = 0;
  for (int i = 0; i < n; ++i)
  {
    float diff = x[i] - scale * L[i];
    float w = quant_weights[i];
    best_mse += w * diff * diff;
  }

  uint8_t Ltmp[256];
  for (int is = -4; is <= 4; ++is)
  {
    if (is == 0)
      continue;
    float iscale_is = (0.1f * is + nmax) / max;
    float scale_is = 1 / iscale_is;
    float mse = 0;
    libgguf_fill_u8_qp(n, nmax, x, Ltmp, iscale_is, true);
    for (int i = 0; i < n; ++i)
    {
      float diff = x[i] - scale_is * Ltmp[i];
      float w = quant_weights[i];
      mse += w * diff * diff;
    }
    if (mse < best_mse)
    {
      best_mse = mse;
      iscale = iscale_is;
    }
  }

  libgguf_fill_u8_qp(n, nmax, x, L, iscale, true);
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i)
  {
    int l = L[i];
    float w = quant_weights[i];
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  for (int itry = 0; itry < 5; ++itry)
  {
    int n_changed = 0;
    for (int i = 0; i < n; ++i)
    {
      float w = quant_weights[i];
      float slx = sumlx - w * x[i] * L[i];
      float sl2 = suml2 - w * L[i] * L[i];
      if (slx > 0 && sl2 > 0)
      {
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = libgguf_min(nmax, new_l);
        if (new_l != L[i])
        {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (slx * slx * suml2 > sumlx * sumlx * sl2)
          {
            L[i] = new_l;
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed)
    {
      break;
    }
  }
  return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}





