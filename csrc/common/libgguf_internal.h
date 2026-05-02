#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "libgguf.h"


#define GGML_UNUSED(x) (void)(x)

#ifndef GGML_UNREACHABLE
#if defined(_MSC_VER)
#define GGML_UNREACHABLE() __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#define GGML_UNREACHABLE() __builtin_unreachable()
#else
#define GGML_UNREACHABLE() abort()
#endif
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

// std-c++ allow anonymous unions but some compiler warn on it
#define GGML_COMMON_AGGR_U
// std-c++ do not allow it.
#define GGML_COMMON_AGGR_S

// ieee 754-2008 half-precision float16
// todo: make this not an integral type
typedef uint16_t ggml_fp16_t;
// google brain half-precision bfloat16
typedef struct
{
  uint16_t bits;
} ggml_bf16_t;

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12

// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QI1_0 (QK1_0 / 32)
#define QR1_0 1

#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QR4_1 2

#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))
#define QR_MXFP4 2

#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))
#define QR_NVFP4 2

#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR5_0 2

#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QR5_1 2

#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI2_K (QK_K / (4 * QR2_K))
#define QR2_K 4

#define QI3_K (QK_K / (4 * QR3_K))
#define QR3_K 4

#define QI4_K (QK_K / (4 * QR4_K))
#define QR4_K 2

#define QI5_K (QK_K / (4 * QR5_K))
#define QR5_K 2

#define QI6_K (QK_K / (4 * QR6_K))
#define QR6_K 2

#define QI2_XXS (QK_K / (4 * QR2_XXS))
#define QR2_XXS 4

#define QI2_XS (QK_K / (4 * QR2_XS))
#define QR2_XS 4

#define QI2_S (QK_K / (4 * QR2_S))
#define QR2_S 4

#define QI3_XXS (QK_K / (4 * QR3_XXS))
#define QR3_XXS 4

#define QI3_XS (QK_K / (4 * QR3_XS))
#define QR3_XS 4

#define QI1_S (QK_K / (4 * QR1_S))
#define QR1_S 8

#define QI1_M (QK_K / (4 * QR1_M))
#define QR1_M 8

#define QI4_NL (QK4_NL / (4 * QR4_NL))
#define QR4_NL 2

#define QI4_XS (QK_K / (4 * QR4_XS))
#define QR4_XS 2

#define QI3_S (QK_K / (4 * QR3_S))
#define QR3_S 4

#ifdef _MSC_VER
#define GGML_EXTENSION
#else // _MSC_VER
#define GGML_EXTENSION __extension__
#endif // _MSC_VER

#define QK1_0 128
typedef struct
{
  ggml_half d;           // delta
  uint8_t qs[QK1_0 / 8]; // bits / quants
} block_q1_0;
static_assert(sizeof(block_q1_0) == sizeof(ggml_half) + QK1_0 / 8, "wrong q1_0 block size/padding");

#define QK4_0 32
typedef struct
{
  ggml_half d;           // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct
{
  GGML_EXTENSION union
  {
    struct
    {
      ggml_half d; // delta
      ggml_half m; // min
    } GGML_COMMON_AGGR_S;
    ggml_half2 dm;
  } GGML_COMMON_AGGR_U;
  uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK_MXFP4 32
typedef struct
{
  uint8_t e; // E8M0
  uint8_t qs[QK_MXFP4 / 2];
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4 / 2, "wrong mxfp4 block size/padding");

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16 // sub-block size for per-group scales
typedef struct
{
  uint8_t d[QK_NVFP4 / QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
  uint8_t qs[QK_NVFP4 / 2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;
static_assert(sizeof(block_nvfp4) == sizeof(uint8_t) * (QK_NVFP4 / QK_NVFP4_SUB) + QK_NVFP4 / 2, "wrong nvfp4 block size/padding");

#define QK5_0 32
typedef struct
{
  ggml_half d;           // delta
  uint8_t qh[4];         // 5-th bit of quants
  uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct
{
  GGML_EXTENSION union
  {
    struct
    {
      ggml_half d; // delta
      ggml_half m; // min
    } GGML_COMMON_AGGR_S;
    ggml_half2 dm;
  } GGML_COMMON_AGGR_U;
  uint8_t qh[4];         // 5-th bit of quants
  uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct
{
  ggml_half d;      // delta
  int8_t qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct
{
  GGML_EXTENSION union
  {
    struct
    {
      ggml_half d; // delta
      ggml_half s; // d * sum(qs[i])
    } GGML_COMMON_AGGR_S;
    ggml_half2 ds;
  } GGML_COMMON_AGGR_U;
  int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");

//
// Ternary quantization
//

// 1.6875 bpw
typedef struct
{
  uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
  uint8_t qh[QK_K / 64];                  // 4 elements per byte
  ggml_half d;
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == sizeof(ggml_half) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5, "wrong tq1_0 block size/padding");

// 2.0625 bpw
typedef struct
{
  uint8_t qs[QK_K / 4]; // 2 bits per element
  ggml_half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == sizeof(ggml_half) + QK_K / 4, "wrong tq2_0 block size/padding");

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct
{
  uint8_t scales[QK_K / 16]; // scales and mins, quantized with 4 bits
  uint8_t qs[QK_K / 4];      // quants
  GGML_EXTENSION union
  {
    struct
    {
      ggml_half d;    // super-block scale for quantized scales
      ggml_half dmin; // super-block scale for quantized mins
    } GGML_COMMON_AGGR_S;
    ggml_half2 dm;
  } GGML_COMMON_AGGR_U;
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2 * sizeof(ggml_half) + QK_K / 16 + QK_K / 4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
typedef struct
{
  uint8_t hmask[QK_K / 8]; // quants - high bit
  uint8_t qs[QK_K / 4];    // quants - low 2 bits
  uint8_t scales[12];      // scales, quantized with 6 bits
  ggml_half d;             // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct
{
  GGML_EXTENSION union
  {
    struct
    {
      ggml_half d;    // super-block scale for quantized scales
      ggml_half dmin; // super-block scale for quantized mins
    } GGML_COMMON_AGGR_S;
    ggml_half2 dm;
  } GGML_COMMON_AGGR_U;
  uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2];         // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2, "wrong q4_K block size/padding");

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
typedef struct
{
  GGML_EXTENSION union
  {
    struct
    {
      ggml_half d;    // super-block scale for quantized scales
      ggml_half dmin; // super-block scale for quantized mins
    } GGML_COMMON_AGGR_S;
    ggml_half2 dm;
  } GGML_COMMON_AGGR_U;
  uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  uint8_t qh[QK_K / 8];         // quants, high bit
  uint8_t qs[QK_K / 2];         // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8, "wrong q5_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct
{
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  ggml_half d;              // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct
{
  float d;                  // delta
  int8_t qs[QK_K];          // quants
  int16_t bsums[QK_K / 16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K / 16 * sizeof(int16_t), "wrong q8_K block size/padding");

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct
{
  ggml_half d;
  uint16_t qs[QK_K / 8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_half) + QK_K / 8 * sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
typedef struct
{
  ggml_half d;
  uint16_t qs[QK_K / 8];
  uint8_t scales[QK_K / 32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_half) + QK_K / 8 * sizeof(uint16_t) + QK_K / 32, "wrong iq2_xs block size/padding");

// 2.5625 bpw quants
typedef struct
{
  ggml_half d;
  uint8_t qs[QK_K / 4];
  uint8_t qh[QK_K / 32];
  uint8_t scales[QK_K / 32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == sizeof(ggml_half) + QK_K / 4 + QK_K / 16, "wrong iq2_s block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct
{
  ggml_half d;
  uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3 * (QK_K / 8), "wrong iq3_xxs block size/padding");

// 3.4375 bpw
#define IQ3S_N_SCALE QK_K / 64
typedef struct
{
  ggml_half d;
  uint8_t qs[QK_K / 4];
  uint8_t qh[QK_K / 32];
  uint8_t signs[QK_K / 8];
  uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) + 13 * (QK_K / 32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

// 1.5625 bpw
typedef struct
{
  ggml_half d;
  uint8_t qs[QK_K / 8];
  uint16_t qh[QK_K / 32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K / 8 + QK_K / 16, "wrong iq1_s block size/padding");

// 1.75 bpw
typedef struct
{
  uint8_t qs[QK_K / 8];      // grid index, low 8 bits
  uint8_t qh[QK_K / 16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
  uint8_t scales[QK_K / 32]; // 3-bit block scales (4-bit if QK_K == 64)
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == QK_K / 8 + QK_K / 16 + QK_K / 32, "wrong iq1_m block size/padding");

// Used by IQ1_M quants
typedef union
{
  ggml_half f16;
  uint16_t u16;
} iq1m_scale_t;

// Non-linear quants
#define QK4_NL 32
typedef struct
{
  ggml_half d;
  uint8_t qs[QK4_NL / 2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL / 2, "wrong iq4_nl block size/padding");

typedef struct
{
  ggml_half d;
  uint16_t scales_h;
  uint8_t scales_l[QK_K / 64];
  uint8_t qs[QK_K / 2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K / 64 + QK_K / 2, "wrong iq4_xs block size/padding");


// restrict not standard in C++
#if defined(__GNUC__)
#define RESTRICT __restrict__
#elif defined(__clang__)
#define RESTRICT __restrict
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

extern "C"
{
  // Quantization
  extern void quantize_row_q1_0_ref(const float *RESTRICT x, block_q1_0 *RESTRICT y, int64_t k);
  extern void quantize_row_q4_0_ref(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k);
  extern void quantize_row_q4_1_ref(const float *RESTRICT x, block_q4_1 *RESTRICT y, int64_t k);
  extern void quantize_row_q5_0_ref(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t k);
  extern void quantize_row_q5_1_ref(const float *RESTRICT x, block_q5_1 *RESTRICT y, int64_t k);
  extern void quantize_row_q8_0_ref(const float *RESTRICT x, block_q8_0 *RESTRICT y, int64_t k);

  extern void quantize_row_mxfp4_ref(const float *RESTRICT x, block_mxfp4 *RESTRICT y, int64_t k);
  extern void quantize_row_nvfp4_ref(const float *RESTRICT x, block_nvfp4 *RESTRICT y, int64_t k);

  extern void quantize_row_q2_K_ref(const float *RESTRICT x, block_q2_K *RESTRICT y, int64_t k);
  extern void quantize_row_q3_K_ref(const float *RESTRICT x, block_q3_K *RESTRICT y, int64_t k);
  extern void quantize_row_q4_K_ref(const float *RESTRICT x, block_q4_K *RESTRICT y, int64_t k);
  extern void quantize_row_q5_K_ref(const float *RESTRICT x, block_q5_K *RESTRICT y, int64_t k);
  extern void quantize_row_q6_K_ref(const float *RESTRICT x, block_q6_K *RESTRICT y, int64_t k);

  extern void quantize_row_tq1_0_ref(const float *RESTRICT x, block_tq1_0 *RESTRICT y, int64_t k);
  extern void quantize_row_tq2_0_ref(const float *RESTRICT x, block_tq2_0 *RESTRICT y, int64_t k);

  extern void quantize_row_iq3_xxs_ref(const float *RESTRICT x, block_iq3_xxs *RESTRICT y, int64_t k);
  extern void quantize_row_iq4_nl_ref(const float *RESTRICT x, block_iq4_nl *RESTRICT y, int64_t k);
  extern void quantize_row_iq4_xs_ref(const float *RESTRICT x, block_iq4_xs *RESTRICT y, int64_t k);
  extern void quantize_row_iq3_s_ref(const float *RESTRICT x, block_iq3_s *RESTRICT y, int64_t k);
  extern void quantize_row_iq2_s_ref(const float *RESTRICT x, block_iq2_s *RESTRICT y, int64_t k);

  // Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
  extern size_t quantize_iq2_xxs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq2_xs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq2_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq3_xxs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq1_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq1_m(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq4_nl(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq4_xs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_iq3_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);

  extern size_t quantize_tq1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_tq2_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);

  extern size_t quantize_q2_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q3_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q4_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q5_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q6_K(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q1_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q4_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q4_1(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q5_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q5_1(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_q8_0(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);

  extern size_t quantize_mxfp4(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
  extern size_t quantize_nvfp4(const float *RESTRICT src, void *RESTRICT dst, int64_t nrows, int64_t n_per_row, const float *imatrix);

  // Dequantization
  extern void dequantize_row_q1_0(const block_q1_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q1_0_ref(const block_q1_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q4_0(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q4_0_ref(const block_q4_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q4_1(const block_q4_1 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q4_1_ref(const block_q4_1 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q5_0(const block_q5_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q5_0_ref(const block_q5_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q5_1(const block_q5_1 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q5_1_ref(const block_q5_1 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q8_0(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q8_0_ref(const block_q8_0 *RESTRICT x, float *RESTRICT y, int64_t k);

  extern void dequantize_row_mxfp4(const block_mxfp4 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_mxfp4_ref(const block_mxfp4 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_nvfp4(const block_nvfp4 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_nvfp4_ref(const block_nvfp4 *RESTRICT x, float *RESTRICT y, int64_t k);

  extern void dequantize_row_q2_K(const block_q2_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q2_K_ref(const block_q2_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q3_K(const block_q3_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q3_K_ref(const block_q3_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q4_K(const block_q4_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q4_K_ref(const block_q4_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q5_K(const block_q5_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q5_K_ref(const block_q5_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q6_K(const block_q6_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q6_K_ref(const block_q6_K *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_q8_K(const block_q8_K *RESTRICT x, float *RESTRICT y, int64_t k);

  extern void dequantize_row_tq1_0(const block_tq1_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_tq1_0_ref(const block_tq1_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_tq2_0(const block_tq2_0 *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_tq2_0_ref(const block_tq2_0 *RESTRICT x, float *RESTRICT y, int64_t k);

  extern void dequantize_row_iq2_xxs(const block_iq2_xxs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq2_xxs_ref(const block_iq2_xxs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq2_xs(const block_iq2_xs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq2_xs_ref(const block_iq2_xs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq2_s(const block_iq2_s *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq2_s_ref(const block_iq2_s *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq3_xxs(const block_iq3_xxs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq3_xxs_ref(const block_iq3_xxs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq1_s(const block_iq1_s *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq1_s_ref(const block_iq1_s *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq1_m(const block_iq1_m *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq1_m_ref(const block_iq1_m *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq4_nl(const block_iq4_nl *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq4_nl_ref(const block_iq4_nl *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq4_xs(const block_iq4_xs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq4_xs_ref(const block_iq4_xs *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq3_s(const block_iq3_s *RESTRICT x, float *RESTRICT y, int64_t k);
  extern void dequantize_row_iq3_s_ref(const block_iq3_s *RESTRICT x, float *RESTRICT y, int64_t k);

  extern void iq2xs_init_impl(enum ggml_type type);
  extern void iq2xs_free_impl(enum ggml_type type);
  extern void iq3xs_init_impl(int grid_size);
  extern void iq3xs_free_impl(int grid_size);
}

bool ggml_validate_row_data(enum ggml_type type, const void *data, size_t nbytes);
