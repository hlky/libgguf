#pragma once

#include "libgguf_internal.h"
#include "libgguf_tables.h"

static inline float fp32_from_bits(uint32_t w)
{
  union
  {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f)
{
  union
  {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h)
{
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__GNUC__) && !defined(__STRICT_ANSI__)) || __cplusplus >= 201703L
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
                          (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
#if (defined(__GNUC__) && !defined(__STRICT_ANSI__)) || __cplusplus >= 201703L
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000))
  {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

static inline float ggml_e8m0_to_fp32(uint8_t x)
{
  uint32_t bits; // Stores the raw bit representation of the float

  // Handle special case for minimum exponent (denormalized float)
  if (x == 0)
  {
    // Bit pattern for 2^(-127):
    // - Sign bit: 0 (positive)
    // - Exponent: 0 (denormalized number)
    // - Mantissa: 0x400000 (0.5 in fractional form)
    // Value = 0.5 * 2^(-126) = 2^(-127)
    bits = 0x00400000;
  }
  // note: disabled as we don't need to handle NaNs
  //// Handle special case for NaN (all bits set)
  // else if (x == 0xFF) {
  //     // Standard quiet NaN pattern:
  //     // - Sign bit: 0
  //     // - Exponent: all 1s (0xFF)
  //     // - Mantissa: 0x400000 (quiet NaN flag)
  //     bits = 0x7FC00000;
  // }
  //  Normalized values (most common case)
  else
  {
    // Construct normalized float by shifting exponent into position:
    // - Exponent field: 8 bits (positions 30-23)
    // - Mantissa: 0 (implicit leading 1)
    // Value = 2^(x - 127)
    bits = (uint32_t)x << 23;
  }

  float result; // Final float value
                // Safely reinterpret bit pattern as float without type-punning issues
  memcpy(&result, &bits, sizeof(float));
  return result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
static inline float ggml_e8m0_to_fp32_half(uint8_t x)
{
  uint32_t bits;

  // For x < 2: use precomputed denormal patterns
  if (x < 2)
  {
    // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
    bits = 0x00200000 << x;
  }
  // For x >= 2: normalized exponent adjustment
  else
  {
    // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
    bits = (uint32_t)(x - 1) << 23;
  }
  // Note: NaNs are not handled here

  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
static inline float ggml_ue4m3_to_fp32(uint8_t x)
{
  if (x == 0 || x == 0x7F)
  {
    return 0.0f;
  }
  int exp = (x >> 3) & 0xF;
  int man = x & 0x7;
  float raw;
  if (exp == 0)
  {
    raw = ldexpf((float)man, -9);
  }
  else
  {
    raw = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
  }
  return raw * 0.5f;
}

static inline uint8_t ggml_fp32_to_ue4m3(float x)
{
  if (!(x > 0.0f))
  {
    return 0;
  }
  if (x > 448.0f)
  {
    x = 448.0f;
  }
  uint32_t bits;
  memcpy(&bits, &x, 4);
  int fp32_exp = ((bits >> 23) & 0xFF) - 127;
  int fp32_man = (bits >> 20) & 0x7;
  int ue4m3_exp = fp32_exp + 7;
  if (ue4m3_exp <= 0)
  {
    // subnormal: value = man * 2^-9, man = round(x * 2^9)
    int man = (int)(x * 512.0f + 0.5f);
    if (man > 7)
    {
      man = 7;
    }
    if (man < 1)
    {
      return 0;
    }
    return (uint8_t)man;
  }
  if (ue4m3_exp >= 15)
  {
    return 0x7E;
  }
  int round_bit = (bits >> 19) & 1;
  int ue4m3_man = fp32_man + round_bit;
  if (ue4m3_man > 7)
  {
    ue4m3_man = 0;
    ue4m3_exp++;
    if (ue4m3_exp >= 15)
    {
      return 0x7E;
    }
  }
  return (uint8_t)((ue4m3_exp << 3) | ue4m3_man);
}

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h)
{
  union
  {
    float f;
    uint32_t i;
  } u;
  u.i = (uint32_t)h.bits << 16;
  return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s)
{
  ggml_bf16_t h;
  union
  {
    float f;
    uint32_t i;
  } u;
  u.f = s;
  if ((u.i & 0x7fffffff) > 0x7f800000)
  {                            /* nan */
    h.bits = (u.i >> 16) | 64; /* force to quiet */
    return h;
  }
  h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
  return h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

static inline int best_index_int8(int n, const int8_t *val, float x)
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

// reference implementation for deterministic creation of model files

static inline int best_index_mxfp4(float x, float e)
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

static inline int nearest_int(float fval)
{
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

typedef float (*libgguf_make_qx_quants_fn)(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L, int rmse_type,
                                           const float *RESTRICT qw);
typedef float (*libgguf_make_q3_quants_fn)(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L, bool do_rmse);
typedef float (*libgguf_make_qkx_quants_fn)(int n, int nmax, const float *RESTRICT x, const float *RESTRICT weights,
                                            uint8_t *RESTRICT L, float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                            float rmin, float rdelta, int nstep, bool use_mad);
typedef float (*libgguf_make_qp_quants_fn)(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                           const float *quant_weights);
typedef float (*libgguf_fp16_to_fp32_fn)(ggml_fp16_t h);
typedef ggml_fp16_t (*libgguf_fp32_to_fp16_fn)(float f);
typedef float (*libgguf_bf16_to_fp32_fn)(ggml_bf16_t h);
typedef ggml_bf16_t (*libgguf_fp32_to_bf16_fn)(float f);
typedef float (*libgguf_e8m0_to_fp32_fn)(uint8_t x);
typedef float (*libgguf_ue4m3_to_fp32_fn)(uint8_t x);
typedef uint8_t (*libgguf_fp32_to_ue4m3_fn)(float x);
typedef int (*libgguf_best_index_int8_fn)(int n, const int8_t *val, float x);
typedef int (*libgguf_best_index_mxfp4_fn)(float x, float e);
typedef int (*libgguf_nearest_int_fn)(float fval);
typedef void (*libgguf_get_scale_min_k4_fn)(int j, const uint8_t *RESTRICT q, uint8_t *RESTRICT d, uint8_t *RESTRICT m);

extern "C" const char *libgguf_common_quant_backend(void);
extern "C" int libgguf_common_quant_cpu_supports_backend(const char *backend);
extern "C" int libgguf_common_quant_set_backend(const char *backend);
extern "C" uint64_t libgguf_common_quant_probe_for_backend(const char *backend);
extern "C" float libgguf_fp16_to_fp32_dispatch(ggml_fp16_t h);
extern "C" ggml_fp16_t libgguf_fp32_to_fp16_dispatch(float f);
extern "C" float libgguf_bf16_to_fp32_dispatch(ggml_bf16_t h);
extern "C" ggml_bf16_t libgguf_fp32_to_bf16_dispatch(float f);
extern "C" float libgguf_e8m0_to_fp32_dispatch(uint8_t x);
extern "C" float libgguf_e8m0_to_fp32_half_dispatch(uint8_t x);
extern "C" float libgguf_ue4m3_to_fp32_dispatch(uint8_t x);
extern "C" uint8_t libgguf_fp32_to_ue4m3_dispatch(float x);
extern "C" int libgguf_best_index_int8_dispatch(int n, const int8_t *val, float x);
extern "C" int libgguf_best_index_mxfp4_dispatch(float x, float e);
extern "C" int libgguf_nearest_int_dispatch(float fval);
extern "C" void libgguf_get_scale_min_k4_dispatch(int j, const uint8_t *RESTRICT q, uint8_t *RESTRICT d,
                                                  uint8_t *RESTRICT m);
extern "C" float libgguf_make_qx_quants_dispatch(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                                 int rmse_type, const float *RESTRICT qw);
extern "C" float libgguf_make_q3_quants_dispatch(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                                 bool do_rmse);
extern "C" float libgguf_make_qkx2_quants_dispatch(int n, int nmax, const float *RESTRICT x,
                                                   const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                   float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                   float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qkx3_quants_dispatch(int n, int nmax, const float *RESTRICT x,
                                                   const float *RESTRICT weights, uint8_t *RESTRICT L,
                                                   float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                                   float rmin, float rdelta, int nstep, bool use_mad);
extern "C" float libgguf_make_qp_quants_dispatch(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                                 const float *quant_weights);

static float libgguf_make_qx_quants_ref(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L, int rmse_type,
                                        const float *RESTRICT qw)
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
      L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
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
    l = MAX(-nmax, MIN(nmax - 1, l));
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
      l = MAX(-nmax, MIN(nmax - 1, l));
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
        L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

static float libgguf_make_q3_quants_ref(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L, bool do_rmse)
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
      l = MAX(-nmax, MIN(nmax - 1, l));
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
          new_l = MAX(-nmax, MIN(nmax - 1, new_l));
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
    l = MAX(-nmax, MIN(nmax - 1, l));
    L[i] = l + nmax;
  }
  return 1 / iscale;
}

static float libgguf_make_qkx2_quants_ref(int n, int nmax, const float *RESTRICT x, const float *RESTRICT weights,
                                          uint8_t *RESTRICT L, float *RESTRICT the_min, uint8_t *RESTRICT Laux,
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
    L[i] = MAX(0, MIN(nmax, l));
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
      l = MAX(0, MIN(nmax, l));
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

static inline void get_scale_min_k4(int j, const uint8_t *RESTRICT q, uint8_t *RESTRICT d, uint8_t *RESTRICT m)
{
  if (j < 4)
  {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  }
  else
  {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

//========================- 2-bit (de)-quantization

static float libgguf_make_qkx3_quants_ref(int n, int nmax, const float *RESTRICT x, const float *RESTRICT weights,
                                          uint8_t *RESTRICT L, float *RESTRICT the_min, uint8_t *RESTRICT Laux,
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
    L[i] = MAX(0, MIN(nmax, l));
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
      l = MAX(0, MIN(nmax, l));
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

static float libgguf_make_qp_quants_ref(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                        const float *quant_weights)
{
  float max = 0;
  for (int i = 0; i < n; ++i)
  {
    max = MAX(max, x[i]);
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
      l = MIN(nmax, l);
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
    l = MIN(nmax, l);
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
        new_l = MIN(nmax, new_l);
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

static inline float make_qx_quants(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L, int rmse_type,
                                   const float *RESTRICT qw)
{
  return libgguf_make_qx_quants_dispatch(n, nmax, x, L, rmse_type, qw);
}

static inline float make_q3_quants(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L, bool do_rmse)
{
  return libgguf_make_q3_quants_dispatch(n, nmax, x, L, do_rmse);
}

static inline float make_qkx2_quants(int n, int nmax, const float *RESTRICT x, const float *RESTRICT weights,
                                     uint8_t *RESTRICT L, float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                     float rmin, float rdelta, int nstep, bool use_mad)
{
  return libgguf_make_qkx2_quants_dispatch(n, nmax, x, weights, L, the_min, Laux, rmin, rdelta, nstep, use_mad);
}

static inline float make_qkx3_quants(int n, int nmax, const float *RESTRICT x, const float *RESTRICT weights,
                                     uint8_t *RESTRICT L, float *RESTRICT the_min, uint8_t *RESTRICT Laux,
                                     float rmin, float rdelta, int nstep, bool use_mad)
{
  return libgguf_make_qkx3_quants_dispatch(n, nmax, x, weights, L, the_min, Laux, rmin, rdelta, nstep, use_mad);
}

static inline float make_qp_quants(int n, int nmax, const float *RESTRICT x, uint8_t *RESTRICT L,
                                   const float *quant_weights)
{
  return libgguf_make_qp_quants_dispatch(n, nmax, x, L, quant_weights);
}
