#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include <libgguf/formats.h>

#include "kernels/cpu/common/tables.h"

template<typename A, typename B>
static inline typename std::common_type<A, B>::type libgguf_min(A a, B b)
{
  using result_t = typename std::common_type<A, B>::type;
  return static_cast<result_t>(a) < static_cast<result_t>(b) ? static_cast<result_t>(a) : static_cast<result_t>(b);
}

template<typename A, typename B>
static inline typename std::common_type<A, B>::type libgguf_max(A a, B b)
{
  using result_t = typename std::common_type<A, B>::type;
  return static_cast<result_t>(a) > static_cast<result_t>(b) ? static_cast<result_t>(a) : static_cast<result_t>(b);
}

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

static inline float libgguf_compute_fp16_to_fp32(ggml_fp16_t h)
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

static inline ggml_fp16_t libgguf_compute_fp32_to_fp16(float f)
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

#define LIBGGUF_COMPUTE_FP16_TO_FP32(x) libgguf_compute_fp16_to_fp32(x)
#define LIBGGUF_COMPUTE_FP32_TO_FP16(x) libgguf_compute_fp32_to_fp16(x)

#define LIBGGUF_FP16_TO_FP32(x) LIBGGUF_COMPUTE_FP16_TO_FP32(x)
#define LIBGGUF_FP32_TO_FP16(x) LIBGGUF_COMPUTE_FP32_TO_FP16(x)

static inline float libgguf_e8m0_to_fp32(uint8_t x)
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

// Equal to libgguf_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
static inline float libgguf_e8m0_to_fp32_half(uint8_t x)
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

#define LIBGGUF_E8M0_TO_FP32(x) libgguf_e8m0_to_fp32(x)
#define LIBGGUF_E8M0_TO_FP32_HALF(x) libgguf_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
static inline float libgguf_ue4m3_to_fp32(uint8_t x)
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

static inline uint8_t libgguf_fp32_to_ue4m3(float x)
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
static inline float libgguf_compute_bf16_to_fp32(ggml_bf16_t h)
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
static inline ggml_bf16_t libgguf_compute_fp32_to_bf16(float s)
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

#define LIBGGUF_FP32_TO_BF16(x) libgguf_compute_fp32_to_bf16(x)
#define LIBGGUF_BF16_TO_FP32(x) libgguf_compute_bf16_to_fp32(x)

static inline int best_index_int8_scalar(int n, const int8_t *val, float x)
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

static inline int best_index_mxfp4_scalar(float x, float e)
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

static inline int nearest_int_scalar(float fval)
{
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}




static inline void get_scale_min_k4(int j, const uint8_t *LIBGGUF_RESTRICT q, uint8_t *LIBGGUF_RESTRICT d, uint8_t *LIBGGUF_RESTRICT m)
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

int best_index_int8(int n, const int8_t *val, float x);
int best_index_mxfp4(float x, float e);
int nearest_int(float fval);

float make_qx_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, int rmse_type,
                     const float *LIBGGUF_RESTRICT qw);
float make_q3_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, bool do_rmse);
float make_qkx2_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, const float *LIBGGUF_RESTRICT weights,
                       uint8_t *LIBGGUF_RESTRICT L, float *LIBGGUF_RESTRICT the_min, uint8_t *LIBGGUF_RESTRICT Laux,
                       float rmin, float rdelta, int nstep, bool use_mad);
float make_qkx3_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, const float *LIBGGUF_RESTRICT weights,
                       uint8_t *LIBGGUF_RESTRICT L, float *LIBGGUF_RESTRICT the_min, uint8_t *LIBGGUF_RESTRICT Laux,
                       float rmin, float rdelta, int nstep, bool use_mad);
float make_qp_quants(int n, int nmax, const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT L,
                     const float *quant_weights);
