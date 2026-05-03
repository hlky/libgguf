#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cerrno>
#include <thread>
#include <vector>

#include "libgguf.h"
#include "common/libgguf_internal.h"
#include "common/libgguf_storage.h"

extern "C" LIBGGUF_API size_t libgguf_row_size(enum ggml_type type, int64_t n_per_row)
{
  switch (type)
  {
  case GGML_TYPE_F32:
    return (size_t)n_per_row * sizeof(float);
  case GGML_TYPE_F16:
    return (size_t)n_per_row * sizeof(ggml_fp16_t);
  case GGML_TYPE_BF16:
    return (size_t)n_per_row * sizeof(ggml_bf16_t);
  case GGML_TYPE_Q1_0:
    return (size_t)n_per_row * sizeof(block_q1_0) / QK1_0;
  case GGML_TYPE_Q4_0:
    return (size_t)n_per_row * sizeof(block_q4_0) / QK4_0;
  case GGML_TYPE_Q4_1:
    return (size_t)n_per_row * sizeof(block_q4_1) / QK4_1;
  case GGML_TYPE_Q5_0:
    return (size_t)n_per_row * sizeof(block_q5_0) / QK5_0;
  case GGML_TYPE_Q5_1:
    return (size_t)n_per_row * sizeof(block_q5_1) / QK5_1;
  case GGML_TYPE_Q8_0:
    return (size_t)n_per_row * sizeof(block_q8_0) / QK8_0;
  case GGML_TYPE_Q2_K:
    return (size_t)n_per_row * sizeof(block_q2_K) / QK_K;
  case GGML_TYPE_Q3_K:
    return (size_t)n_per_row * sizeof(block_q3_K) / QK_K;
  case GGML_TYPE_Q4_K:
    return (size_t)n_per_row * sizeof(block_q4_K) / QK_K;
  case GGML_TYPE_Q5_K:
    return (size_t)n_per_row * sizeof(block_q5_K) / QK_K;
  case GGML_TYPE_Q6_K:
    return (size_t)n_per_row * sizeof(block_q6_K) / QK_K;
  case GGML_TYPE_IQ2_XXS:
    return (size_t)n_per_row * sizeof(block_iq2_xxs) / QK_K;
  case GGML_TYPE_IQ2_XS:
    return (size_t)n_per_row * sizeof(block_iq2_xs) / QK_K;
  case GGML_TYPE_IQ2_S:
    return (size_t)n_per_row * sizeof(block_iq2_s) / QK_K;
  case GGML_TYPE_IQ3_XXS:
    return (size_t)n_per_row * sizeof(block_iq3_xxs) / QK_K;
  case GGML_TYPE_IQ3_S:
    return (size_t)n_per_row * sizeof(block_iq3_s) / QK_K;
  case GGML_TYPE_IQ1_S:
    return (size_t)n_per_row * sizeof(block_iq1_s) / QK_K;
  case GGML_TYPE_IQ1_M:
    return (size_t)n_per_row * sizeof(block_iq1_m) / QK_K;
  case GGML_TYPE_IQ4_NL:
    return (size_t)n_per_row * sizeof(block_iq4_nl) / QK4_NL;
  case GGML_TYPE_IQ4_XS:
    return (size_t)n_per_row * sizeof(block_iq4_xs) / QK_K;
  case GGML_TYPE_TQ1_0:
    return (size_t)n_per_row * sizeof(block_tq1_0) / QK_K;
  case GGML_TYPE_TQ2_0:
    return (size_t)n_per_row * sizeof(block_tq2_0) / QK_K;
  case GGML_TYPE_MXFP4:
    return (size_t)n_per_row * sizeof(block_mxfp4) / QK_MXFP4;
  case GGML_TYPE_NVFP4:
    return (size_t)n_per_row * sizeof(block_nvfp4) / QK_NVFP4;
  default:
    return 0;
  }
}

extern "C" LIBGGUF_API size_t libgguf_type_size(enum ggml_type type)
{
  switch (type)
  {
  case GGML_TYPE_Q1_0:
    return sizeof(block_q1_0);
  case GGML_TYPE_Q4_0:
    return sizeof(block_q4_0);
  case GGML_TYPE_Q4_1:
    return sizeof(block_q4_1);
  case GGML_TYPE_Q5_0:
    return sizeof(block_q5_0);
  case GGML_TYPE_Q5_1:
    return sizeof(block_q5_1);
  case GGML_TYPE_Q8_0:
    return sizeof(block_q8_0);
  case GGML_TYPE_Q8_1:
    return sizeof(block_q8_1);
  case GGML_TYPE_Q2_K:
    return sizeof(block_q2_K);
  case GGML_TYPE_Q3_K:
    return sizeof(block_q3_K);
  case GGML_TYPE_Q4_K:
    return sizeof(block_q4_K);
  case GGML_TYPE_Q5_K:
    return sizeof(block_q5_K);
  case GGML_TYPE_Q6_K:
    return sizeof(block_q6_K);
  case GGML_TYPE_Q8_K:
    return sizeof(block_q8_K);
  case GGML_TYPE_IQ2_XXS:
    return sizeof(block_iq2_xxs);
  case GGML_TYPE_IQ2_XS:
    return sizeof(block_iq2_xs);
  case GGML_TYPE_IQ3_XXS:
    return sizeof(block_iq3_xxs);
  case GGML_TYPE_IQ1_S:
    return sizeof(block_iq1_s);
  case GGML_TYPE_IQ4_NL:
    return sizeof(block_iq4_nl);
  case GGML_TYPE_IQ3_S:
    return sizeof(block_iq3_s);
  case GGML_TYPE_IQ2_S:
    return sizeof(block_iq2_s);
  case GGML_TYPE_IQ4_XS:
    return sizeof(block_iq4_xs);
  case GGML_TYPE_IQ1_M:
    return sizeof(block_iq1_m);
  case GGML_TYPE_TQ1_0:
    return sizeof(block_tq1_0);
  case GGML_TYPE_TQ2_0:
    return sizeof(block_tq2_0);
  case GGML_TYPE_MXFP4:
    return sizeof(block_mxfp4);
  case GGML_TYPE_NVFP4:
    return sizeof(block_nvfp4);
  case GGML_TYPE_F16:
    return sizeof(ggml_fp16_t);
  case GGML_TYPE_BF16:
    return sizeof(ggml_bf16_t);
  case GGML_TYPE_F32:
    return sizeof(float);
  case GGML_TYPE_F64:
    return sizeof(double);
  case GGML_TYPE_I8:
    return sizeof(int8_t);
  case GGML_TYPE_I16:
    return sizeof(int16_t);
  case GGML_TYPE_I32:
    return sizeof(int32_t);
  case GGML_TYPE_I64:
    return sizeof(int64_t);
  default:
    return 0;
  }
}

extern "C" LIBGGUF_API const char *libgguf_type_name(enum ggml_type type)
{
  switch (type)
  {
  case GGML_TYPE_F32:
    return "f32";
  case GGML_TYPE_F16:
    return "f16";
  case GGML_TYPE_Q4_0:
    return "q4_0";
  case GGML_TYPE_Q4_1:
    return "q4_1";
  case GGML_TYPE_Q5_0:
    return "q5_0";
  case GGML_TYPE_Q5_1:
    return "q5_1";
  case GGML_TYPE_Q8_0:
    return "q8_0";
  case GGML_TYPE_Q8_1:
    return "q8_1";
  case GGML_TYPE_Q2_K:
    return "q2_K";
  case GGML_TYPE_Q3_K:
    return "q3_K";
  case GGML_TYPE_Q4_K:
    return "q4_K";
  case GGML_TYPE_Q5_K:
    return "q5_K";
  case GGML_TYPE_Q6_K:
    return "q6_K";
  case GGML_TYPE_Q8_K:
    return "q8_K";
  case GGML_TYPE_IQ2_XXS:
    return "iq2_xxs";
  case GGML_TYPE_IQ2_XS:
    return "iq2_xs";
  case GGML_TYPE_IQ3_XXS:
    return "iq3_xxs";
  case GGML_TYPE_IQ1_S:
    return "iq1_s";
  case GGML_TYPE_IQ4_NL:
    return "iq4_nl";
  case GGML_TYPE_IQ3_S:
    return "iq3_s";
  case GGML_TYPE_IQ2_S:
    return "iq2_s";
  case GGML_TYPE_IQ4_XS:
    return "iq4_xs";
  case GGML_TYPE_I8:
    return "i8";
  case GGML_TYPE_I16:
    return "i16";
  case GGML_TYPE_I32:
    return "i32";
  case GGML_TYPE_I64:
    return "i64";
  case GGML_TYPE_F64:
    return "f64";
  case GGML_TYPE_IQ1_M:
    return "iq1_m";
  case GGML_TYPE_BF16:
    return "bf16";
  case GGML_TYPE_TQ1_0:
    return "tq1_0";
  case GGML_TYPE_TQ2_0:
    return "tq2_0";
  case GGML_TYPE_MXFP4:
    return "mxfp4";
  case GGML_TYPE_NVFP4:
    return "nvfp4";
  case GGML_TYPE_Q1_0:
    return "q1_0";
  default:
    return "unknown";
  }
}

static void libgguf_quantize_init(enum ggml_type type)
{
  switch (type)
  {
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
    iq2xs_init_impl(type);
    break;
  case GGML_TYPE_IQ3_XXS:
    iq3xs_init_impl(256);
    break;
  case GGML_TYPE_IQ3_S:
    iq3xs_init_impl(512);
    break;
  default:
    break;
  }
}

extern "C" LIBGGUF_API void libgguf_quantize_free(void)
{
  iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
  iq2xs_free_impl(GGML_TYPE_IQ2_XS);
  iq2xs_free_impl(GGML_TYPE_IQ2_S);
  iq2xs_free_impl(GGML_TYPE_IQ1_S);
  iq2xs_free_impl(GGML_TYPE_IQ1_M);
  iq3xs_free_impl(256);
  iq3xs_free_impl(512);
}

extern "C" LIBGGUF_API bool libgguf_quantize_requires_imatrix(enum ggml_type type)
{
  return type == GGML_TYPE_IQ2_XXS ||
         type == GGML_TYPE_IQ2_XS ||
         type == GGML_TYPE_IQ1_S;
}

static unsigned int libgguf_quantize_thread_count(int64_t nrows)
{
  if (nrows < 64)
  {
    return 1;
  }

  const char *env = std::getenv("LIBGGUF_NUM_THREADS");
  if (env != nullptr && env[0] != '\0')
  {
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (errno == 0 && end != env && parsed > 0)
    {
      return (unsigned int)std::min<int64_t>((int64_t)parsed, nrows);
    }
    return 1;
  }

  const unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0)
  {
    return 1;
  }
  return std::min<unsigned int>(hardware, (unsigned int)nrows);
}

static size_t libgguf_quantize_chunk_serial(
    enum ggml_type type,
    const float *src,
    void *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row,
    const float *imatrix)
{
  const size_t row_size = libgguf_row_size(type, n_per_row);
  const size_t start_row = (size_t)(start / n_per_row);

  assert(row_size != 0);
  assert(start % n_per_row == 0);

  if (libgguf_quantize_requires_imatrix(type))
  {
    assert(imatrix != nullptr);
  }

  switch (type)
  {
  case GGML_TYPE_F32:
    return libgguf_store_f32(src + start, (char *)dst + start_row * row_size, (size_t)nrows * (size_t)n_per_row);
  case GGML_TYPE_F16:
    return libgguf_store_f16(src + start, (char *)dst + start_row * row_size, (size_t)nrows * (size_t)n_per_row);
  case GGML_TYPE_BF16:
    return libgguf_store_bf16(src + start, (char *)dst + start_row * row_size, (size_t)nrows * (size_t)n_per_row);
  case GGML_TYPE_Q1_0:
    return quantize_q1_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q4_0:
    return quantize_q4_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q4_1:
    return quantize_q4_1(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q5_0:
    return quantize_q5_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q5_1:
    return quantize_q5_1(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q8_0:
    return quantize_q8_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q2_K:
    return quantize_q2_K(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q3_K:
    return quantize_q3_K(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q4_K:
    return quantize_q4_K(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q5_K:
    return quantize_q5_K(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q6_K:
    return quantize_q6_K(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ2_XXS:
    return quantize_iq2_xxs(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ2_XS:
    return quantize_iq2_xs(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ2_S:
    return quantize_iq2_s(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ3_XXS:
    return quantize_iq3_xxs(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ3_S:
    return quantize_iq3_s(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ1_S:
    return quantize_iq1_s(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ1_M:
    return quantize_iq1_m(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ4_NL:
    return quantize_iq4_nl(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_IQ4_XS:
    return quantize_iq4_xs(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_TQ1_0:
    return quantize_tq1_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_TQ2_0:
    return quantize_tq2_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_MXFP4:
    return quantize_mxfp4(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_NVFP4:
    return quantize_nvfp4(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  default:
    assert(false && "unsupported quantization type");
    return 0;
  }
}

extern "C" LIBGGUF_API size_t libgguf_quantize_chunk(
    enum ggml_type type,
    const float *src,
    void *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row,
    const float *imatrix)
{
  if (nrows <= 0)
  {
    return 0;
  }

  const size_t row_size = libgguf_row_size(type, n_per_row);
  assert(row_size != 0);
  assert(start % n_per_row == 0);

  if (libgguf_quantize_requires_imatrix(type))
  {
    assert(imatrix != nullptr);
  }

  libgguf_quantize_init(type);

  const unsigned int nthreads = libgguf_quantize_thread_count(nrows);
  if (nthreads <= 1)
  {
    return libgguf_quantize_chunk_serial(type, src, dst, start, nrows, n_per_row, imatrix);
  }

  std::vector<std::thread> threads;
  std::vector<size_t> written(nthreads, 0);
  threads.reserve(nthreads);

  const int64_t rows_per_thread = (nrows + (int64_t)nthreads - 1) / (int64_t)nthreads;
  for (unsigned int thread_id = 0; thread_id < nthreads; ++thread_id)
  {
    const int64_t row_begin = (int64_t)thread_id * rows_per_thread;
    if (row_begin >= nrows)
    {
      written.resize(thread_id);
      break;
    }
    const int64_t row_count = std::min<int64_t>(rows_per_thread, nrows - row_begin);
    threads.emplace_back([=, &written]() {
      written[thread_id] = libgguf_quantize_chunk_serial(
          type,
          src,
          dst,
          start + row_begin * n_per_row,
          row_count,
          n_per_row,
          imatrix);
    });
  }

  for (std::thread &thread : threads)
  {
    thread.join();
  }

  size_t total = 0;
  for (size_t n : written)
  {
    total += n;
  }
  return total;
}

static size_t libgguf_dequantize_chunk_serial(
    enum ggml_type type,
    const void *src,
    float *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row)
{
  if (nrows <= 0)
  {
    return 0;
  }

  const size_t row_size = libgguf_row_size(type, n_per_row);
  if (row_size == 0 || n_per_row <= 0 || start % n_per_row != 0)
  {
    return 0;
  }

  const size_t start_row = (size_t)(start / n_per_row);
  const char *src_row = (const char *)src + start_row * row_size;
  float *dst_row = dst + start;
  const int64_t k = nrows * n_per_row;

  switch (type)
  {
  case GGML_TYPE_Q1_0:
    dequantize_row_q1_0((const block_q1_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q4_0:
    dequantize_row_q4_0((const block_q4_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q4_1:
    dequantize_row_q4_1((const block_q4_1 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q5_0:
    dequantize_row_q5_0((const block_q5_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q5_1:
    dequantize_row_q5_1((const block_q5_1 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q8_0:
    dequantize_row_q8_0((const block_q8_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q2_K:
    dequantize_row_q2_K((const block_q2_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q3_K:
    dequantize_row_q3_K((const block_q3_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q4_K:
    dequantize_row_q4_K((const block_q4_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q5_K:
    dequantize_row_q5_K((const block_q5_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_Q6_K:
    dequantize_row_q6_K((const block_q6_K *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ2_XXS:
    dequantize_row_iq2_xxs((const block_iq2_xxs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ2_XS:
    dequantize_row_iq2_xs((const block_iq2_xs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ2_S:
    dequantize_row_iq2_s((const block_iq2_s *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ3_XXS:
    dequantize_row_iq3_xxs((const block_iq3_xxs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ3_S:
    dequantize_row_iq3_s((const block_iq3_s *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ1_S:
    dequantize_row_iq1_s((const block_iq1_s *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ1_M:
    dequantize_row_iq1_m((const block_iq1_m *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ4_NL:
    dequantize_row_iq4_nl((const block_iq4_nl *)src_row, dst_row, k);
    break;
  case GGML_TYPE_IQ4_XS:
    dequantize_row_iq4_xs((const block_iq4_xs *)src_row, dst_row, k);
    break;
  case GGML_TYPE_TQ1_0:
    dequantize_row_tq1_0((const block_tq1_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_TQ2_0:
    dequantize_row_tq2_0((const block_tq2_0 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_MXFP4:
    dequantize_row_mxfp4((const block_mxfp4 *)src_row, dst_row, k);
    break;
  case GGML_TYPE_NVFP4:
    dequantize_row_nvfp4((const block_nvfp4 *)src_row, dst_row, k);
    break;
  default:
    return 0;
  }

  return (size_t)nrows * (size_t)n_per_row * sizeof(float);
}

extern "C" LIBGGUF_API size_t libgguf_dequantize_chunk(
    enum ggml_type type,
    const void *src,
    float *dst,
    int64_t start,
    int64_t nrows,
    int64_t n_per_row)
{
  if (nrows <= 0)
  {
    return 0;
  }

  const size_t row_size = libgguf_row_size(type, n_per_row);
  if (row_size == 0 || n_per_row <= 0 || start % n_per_row != 0)
  {
    return 0;
  }

  const unsigned int nthreads = libgguf_quantize_thread_count(nrows);
  if (nthreads <= 1)
  {
    return libgguf_dequantize_chunk_serial(type, src, dst, start, nrows, n_per_row);
  }

  std::vector<std::thread> threads;
  std::vector<size_t> written(nthreads, 0);
  threads.reserve(nthreads);

  const int64_t rows_per_thread = (nrows + (int64_t)nthreads - 1) / (int64_t)nthreads;
  for (unsigned int thread_id = 0; thread_id < nthreads; ++thread_id)
  {
    const int64_t row_begin = (int64_t)thread_id * rows_per_thread;
    if (row_begin >= nrows)
    {
      written.resize(thread_id);
      break;
    }
    const int64_t row_count = std::min<int64_t>(rows_per_thread, nrows - row_begin);
    threads.emplace_back([=, &written]() {
      written[thread_id] = libgguf_dequantize_chunk_serial(
          type,
          src,
          dst,
          start + row_begin * n_per_row,
          row_count,
          n_per_row);
    });
  }

  for (std::thread &thread : threads)
  {
    thread.join();
  }

  size_t total = 0;
  for (size_t n : written)
  {
    total += n;
  }
  return total;
}
