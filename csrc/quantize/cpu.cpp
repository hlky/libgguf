#include "quantize/cpu.h"

#include <algorithm>
#include <cassert>
#include <thread>
#include <vector>

#include <libgguf/libgguf.h>

#include "cpu/threading.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"

size_t libgguf_quantize_chunk_serial(
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
  if (row_size == 0 || start < 0 || start % n_per_row != 0)
  {
    return 0;
  }

  assert(row_size != 0);
  assert(start % n_per_row == 0);

  if (libgguf_quantize_requires_imatrix(type))
  {
    assert(imatrix != nullptr);
    if (imatrix == nullptr)
    {
      return 0;
    }
  }

  const size_t start_row = (size_t)(start / n_per_row);

  switch (type)
  {
  case GGML_TYPE_F32:
    return quantize_f32(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_F16:
    return quantize_f16(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_BF16:
    return quantize_bf16(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
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
    return quantize_q2_k(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q3_K:
    return quantize_q3_k(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q4_K:
    return quantize_q4_k(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q5_K:
    return quantize_q5_k(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
  case GGML_TYPE_Q6_K:
    return quantize_q6_k(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix);
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

extern "C" LIBGGUF_API void libgguf_quantize_free(void)
{
  iq_tables_free();
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
  if (row_size == 0 || start < 0 || start % n_per_row != 0)
  {
    return 0;
  }

  assert(row_size != 0);
  assert(start % n_per_row == 0);

  if (libgguf_quantize_requires_imatrix(type))
  {
    assert(imatrix != nullptr);
    if (imatrix == nullptr)
    {
      return 0;
    }
  }

  iq_tables_init(type);

  const unsigned int nthreads = libgguf_cpu_thread_count(nrows);
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
