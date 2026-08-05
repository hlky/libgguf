#include "kernels/cpu/common/common.h"

int best_index_int8(int n, const int8_t *val, float x)
{
  if (x <= val[0])
    return 0;
  if (x >= val[n - 1])
    return n - 1;
  int ml = 0, mu = n - 1;
  while (mu - ml > 1)
  {
    int mav = (ml + mu) / 2;
    if (x < val[mav])
      mu = mav;
    else
      ml = mav;
  }
  return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

int best_index_mxfp4(float x, float e)
{
  int best_index = 0;
  float best_err = fabsf(kvalues_mxfp4[0] * e - x);
  for (int i = 1; i < 16; i++)
  {
    float err = fabsf(kvalues_mxfp4[i] * e - x);
    if (err < best_err)
    {
      best_index = i;
      best_err = err;
    }
  }
  return best_index;
}

int nearest_int(float fval)
{
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

float make_qx_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, int rmse_type,
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

float make_q3_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, bool do_rmse)
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

float make_qkx2_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, const float *LIBGGUF_RESTRICT weights,
                                          uint8_t *LIBGGUF_RESTRICT L, float *LIBGGUF_RESTRICT the_min, uint8_t *LIBGGUF_RESTRICT Laux,
                                          float rmin, float rdelta, int nstep, bool use_mad)
{
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 1; i < n; ++i)
  {
#else
  for (int i = 1; i < n; ++i)
  {
#endif
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
    for (int i = 0; i < n; ++i)
      L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_error = 0;
  for (int i = 0; i < n; ++i)
  {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = libgguf_max(0, libgguf_min(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights[i];
    best_error += w * diff;
  }
  if (nstep < 1)
  {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is)
  {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i)
    {
      int l = nearest_int(iscale * (x[i] - min));
      l = libgguf_max(0, libgguf_min(nmax, l));
      Laux[i] = l;
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
        float w = weights[i];
        cur_error += w * diff;
      }
      if (cur_error < best_error)
      {
        for (int i = 0; i < n; ++i)
        {
          L[i] = Laux[i];
        }
        best_error = cur_error;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

float make_qkx3_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, const float *LIBGGUF_RESTRICT weights,
                                          uint8_t *LIBGGUF_RESTRICT L, float *LIBGGUF_RESTRICT the_min, uint8_t *LIBGGUF_RESTRICT Laux,
                                          float rmin, float rdelta, int nstep, bool use_mad)
{
  float min = x[0];
  float max = x[0];
  float sum_w = weights ? weights[0] : x[0] * x[0];
  float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 1; i < n; ++i)
  {
#else
  for (int i = 1; i < n; ++i)
  {
#endif
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
  for (int i = 0; i < n; ++i)
  {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = libgguf_max(0, libgguf_min(nmax, l));
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
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i)
    {
      int l = nearest_int(iscale * (x[i] - min));
      l = libgguf_max(0, libgguf_min(nmax, l));
      Laux[i] = l;
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
        for (int i = 0; i < n; ++i)
        {
          L[i] = Laux[i];
        }
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

