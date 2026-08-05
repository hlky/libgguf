#pragma once

#include <cstdint>
#include <cstdlib>

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

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

typedef uint16_t ggml_fp16_t;
typedef struct
{
  uint16_t bits;
} ggml_bf16_t;

#ifdef _MSC_VER
#define GGML_EXTENSION
#else
#define GGML_EXTENSION __extension__
#endif

#if defined(__GNUC__)
#define RESTRICT __restrict__
#elif defined(__clang__)
#define RESTRICT __restrict
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif
